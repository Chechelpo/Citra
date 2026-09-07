"""Model-facing Python environment tool.

Single Tool that owns the agent-managed Python environment living in the
copied workspace's ``env/python`` venv. ``uv`` is the only package manager
allowed (Constraint C1). All work happens through ``WorkspaceSandbox.run``
with provisioned command resolution and the isolated runtime PATH
(Constraint C3). The venv sits inside the model-facing copied workspace
filesystem; the controller-owned source path is never touched (Constraint
C4).

Actions
-------

* ``select_version`` — install a Python interpreter version with
  ``uv python install`` and recreate the ``env/python`` venv against it.
  The version string is forwarded to ``uv`` as-is, so any expression
  ``uv`` understands (``"3.12"``, ``"3.12.3"``, ``"cpython-3.13.7"``) is
  accepted.
* ``install`` / ``uninstall`` — add or remove PyPI dependencies with
  ``uv pip install`` / ``uv pip uninstall`` against ``env/python``.
* ``sync`` — reconcile ``env/python`` with the current project
  ``pyproject.toml`` via ``uv sync``.
* ``status`` — read the active interpreter, the venv root, and the
  installed package list without mutating anything.

Redirect
--------

The tool re-registers ``python`` and ``python3`` against
``env/python/bin`` after every mutation through
:meth:`WorkspaceContext.refresh_staged_command`, so Bash, Subprocess, and
direct ``sandbox.run`` invocations all resolve to the agent-managed
interpreter (Requirement R4). Bare ``python`` is also reachable through
the isolated runtime ``PATH`` (``env/python/bin`` is prepended in
:meth:`WorkspaceContext.environment`).

Author: brave-ferret
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..capabilities import ToolCapabilities
from ..tool import Tool, ToolDefinition
from .prompt_user import PromptUser

# Accept the same identifier shape ``uv`` itself accepts: dotted version
# strings ("3.12", "3.12.3"), implementation-qualified requests
# ("cpython-3.12.3"), or partial patches ("3"). Refuse anything else so a
# malformed request becomes a clean error rather than a silent uv failure.
_VERSION_PATTERN = re.compile(
    r"^(?:cpython|pypy|graalpy)-?\d+(?:\.\d+){0,2}$|^\d+(?:\.\d+){0,2}$"
)
_MAX_INSTALL_ARGUMENTS = 64
_DEFAULT_SYNC_TIMEOUT_SECONDS = 180
_DEFAULT_INSTALL_TIMEOUT_SECONDS = 240
_MIN_TIMEOUT_SECONDS = 10
_MAX_TIMEOUT_SECONDS = 1800


def _require_version(version: str) -> str:
    """Validate one user-supplied Python version identifier."""
    if not isinstance(version, str) or not version.strip():
        raise ValueError("'version' is required for action='select_version'.")
    candidate = version.strip()
    if not _VERSION_PATTERN.match(candidate):
        raise ValueError(
            "'version' must be a dotted Python version (for example '3.12' "
            "or '3.12.3') or an implementation-qualified request "
            "(for example 'cpython-3.13.7')."
        )
    return candidate


def _parse_package_list(
    raw: object,
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Parse a ``packages`` / ``package`` argument into a deduplicated tuple."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = [str(item) for item in raw if str(item).strip()]
    else:
        raise TypeError(f"'{field_name}' must be a string or an array of strings.")
    if not candidates:
        return ()
    seen: set[str] = set()
    ordered: list[str] = []
    for item in candidates:
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    if not ordered:
        return ()
    if len(ordered) > _MAX_INSTALL_ARGUMENTS:
        raise ValueError(
            f"'{field_name}' accepts at most {_MAX_INSTALL_ARGUMENTS} entries."
        )
    return tuple(ordered)


def _python_definition(
    *,
    name: str,
    description: str,
) -> ChatCompletionTool:
    """Build the canonical model-facing ``python`` tool definition."""
    return ChatCompletionTool(
        function=FunctionDefinition(
            name=name,
            description=description,
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Python environment action to perform.",
                            enum=(
                                "select_version",
                                "install",
                                "uninstall",
                                "sync",
                                "status",
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="version",
                        schema=JsonSchema.string(
                            description=(
                                "Python version to install for "
                                "action='select_version'. Accepts a dotted "
                                "version (for example '3.12' or '3.12.3') "
                                "or an implementation-qualified request "
                                "(for example 'cpython-3.13.7'). uv "
                                "resolves the version; an exact patch is "
                                "recommended for reproducibility."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="packages",
                        schema=JsonSchema.array(
                            JsonSchema.string(
                                description=(
                                    "PyPI requirement, for example 'numpy' "
                                    "or 'scikit-learn==1.5.0'."
                                ),
                            ),
                            description=(
                                "PyPI requirements to install or uninstall. "
                                "Required by 'install' and 'uninstall' "
                                "unless 'package' is supplied."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="package",
                        schema=JsonSchema.string(
                            description=(
                                "Single PyPI requirement shortcut for "
                                "'install' or 'uninstall'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="dev",
                        schema=JsonSchema.boolean(
                            description=(
                                "When true, target the optional dependency "
                                "group of the project (for example "
                                "'[project.optional-dependencies.dev]') for "
                                "action='install'. Defaults to false."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="timeout",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum execution time in seconds. "
                                "Defaults to 240 for install/uninstall and "
                                "180 for sync."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="reason",
                        schema=JsonSchema.string(
                            description=(
                                "Required when network access is needed. "
                                "uv downloads require network (for example, "
                                "sync or install); the "
                                "sandbox's network policy still applies."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )


class Python(Tool):
    """Model-facing tool for the agent-managed Python environment."""

    TOOL_ID = "python"

    ACTIONS: ClassVar[tuple[str, ...]] = (
        "select_version",
        "install",
        "uninstall",
        "sync",
        "status",
    )

    CAPABILITIES = ToolCapabilities(
        actions=ACTIONS,
        action_arguments=("action",),
    )

    # The tool never reads from the host environment, never re-uses a
    # previous answer, and the install / uninstall / select_version / sync
    # operations always invalidate the tool-result cache.
    CACHEABLE = False
    INVALIDATES_TOOL_CACHE = True

    CITRA_DEFINITION = _python_definition(
        name="python",
        description=(
            "Manage the agent-managed Python environment that backs the "
            "current project. The environment is a uv-managed venv under "
            "`./env/python`; `python` and `python3` from Bash, Subprocess, "
            "and any direct sandbox run resolve to it after the first "
            "action. Actions: 'select_version' to install a Python "
            "interpreter version, 'install' and 'uninstall' for PyPI "
            "operations through uv, 'sync' to reconcile the environment "
            "with the current project's `pyproject.toml`, and 'status' "
            "to inspect the active interpreter and packages."
        ),
    )

    # The Citra schema is reused for every known model family. No
    # harness-specific Python callable is currently documented; if one
    # is added later, register a new ``ToolDefinition`` entry here.
    CLAUDE_CODE_DEFINITION = CITRA_DEFINITION
    GEMINI_CLI_DEFINITION = CITRA_DEFINITION
    QWEN_CODE_DEFINITION = CITRA_DEFINITION
    KIMI_CODE_DEFINITION = CITRA_DEFINITION
    ZCODE_DEFINITION = CITRA_DEFINITION

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        """Return the model-facing definitions for the active context."""
        del context
        return (
            ToolDefinition(
                definition=cls.CLAUDE_CODE_DEFINITION,
                model_family_matchers=("claude",),
            ),
            ToolDefinition(
                definition=cls.GEMINI_CLI_DEFINITION,
                model_family_matchers=("gemini",),
            ),
            ToolDefinition(
                definition=cls.QWEN_CODE_DEFINITION,
                model_family_matchers=("qwen",),
            ),
            ToolDefinition(
                definition=cls.KIMI_CODE_DEFINITION,
                model_family_matchers=("kimi", "moonshot"),
            ),
            ToolDefinition(
                definition=cls.ZCODE_DEFINITION,
                model_family_matchers=("glm",),
            ),
            ToolDefinition(
                definition=cls.CITRA_DEFINITION,
            ),
        )

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize the instance."""
        super().__init__(context=context)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Dispatch one Python environment action."""
        if "action" not in arguments:
            raise ValueError(
                "'action' is required; supported actions: " + ", ".join(self.ACTIONS)
            )
        action = str(arguments["action"])
        if action not in self.ACTIONS:
            raise ValueError(
                f"Unsupported python action: {action!r}. Supported actions: "
                + ", ".join(self.ACTIONS)
            )

        workspace = self.context.workspace
        env_root = workspace.python_runtime()
        venv_python = workspace.python_bin() / "python"
        if not venv_python.is_file():
            raise RuntimeError(
                "Agent-managed Python venv is missing: "
                f"{venv_python}. The runtime was started without a usable "
                "Python interpreter; restart the Agent Runtime to recover."
            )

        if action == "status":
            return self._status(env_root=env_root, venv_python=venv_python)

        if action == "select_version":
            version = _require_version(str(arguments.get("version", "") or ""))
            denied = self._authorize_network(arguments, action="select_version")
            if denied is not None:
                return denied
            timeout = self._timeout(arguments, default=_DEFAULT_INSTALL_TIMEOUT_SECONDS)
            return self._select_version(
                version=version,
                env_root=env_root,
                timeout=timeout,
            )

        if action == "install":
            packages = self._packages(arguments)
            if not packages:
                raise ValueError(
                    "'packages' (or 'package') is required for action='install'."
                )

            denied = self._authorize_network(arguments, action="install")
            if denied is not None:
                return denied
            dev = bool(arguments.get("dev", False))
            timeout = self._timeout(
                arguments,
                default=_DEFAULT_INSTALL_TIMEOUT_SECONDS,
            )
            return self._pip(
                uv_subcommand="install",
                packages=packages,
                env_root=env_root,
                dev=dev,
                timeout=timeout,
            )

        if action == "uninstall":
            packages = self._packages(arguments)
            if not packages:
                raise ValueError(
                    "'packages' (or 'package') is required for action='uninstall'."
                )
            timeout = self._timeout(arguments, default=_DEFAULT_INSTALL_TIMEOUT_SECONDS)
            return self._pip(
                uv_subcommand="uninstall",
                packages=packages,
                env_root=env_root,
                dev=False,
                timeout=timeout,
            )

        # action == "sync"
        if self._find_pyproject() is None:
            raise FileNotFoundError(
                "No pyproject.toml was found at the project root: "
                f"{self.context.workspace.workspace / 'pyproject.toml'}"
            )
        denied = self._authorize_network(arguments, action="sync")
        if denied is not None:
            return denied

        timeout = self._timeout(
            arguments,
            default=_DEFAULT_SYNC_TIMEOUT_SECONDS,
        )

        return self._sync(
            env_root=env_root,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _status(
        self,
        *,
        env_root: Path,
        venv_python: Path,
    ) -> str:
        """Report the active interpreter, venv root, and installed packages."""
        version_output = self._run_captured(
            [str(venv_python), "--version"],
            env_root=env_root,
            timeout=30,
            network=False,
        )
        list_output = self._run_captured(
            ["uv", "pip", "list", "--format=json"],
            env_root=env_root,
            timeout=60,
            network=False,
        )
        pyproject = self._find_pyproject()
        return self._render_status(
            env_root=env_root,
            venv_python=venv_python,
            version_output=version_output,
            list_output=list_output,
            pyproject=pyproject,
        )

    def _select_version(
        self,
        *,
        version: str,
        env_root: Path,
        timeout: int,
    ) -> str:
        """Install ``version`` and rebuild the venv against it."""
        install_result = self._run_captured(
            ["uv", "python", "install", version],
            env_root=env_root,
            timeout=timeout,
            network=True,
        )
        # Rebuild the venv against the selected interpreter. ``uv venv
        # --seed --python <ver>`` makes pip/wheel/uv available without an
        # extra install step.
        venv_result = self._run_captured(
            ["uv", "venv", "--seed", "--python", version, str(env_root)],
            env_root=env_root,
            timeout=timeout,
            network=True,
        )
        # The new venv drops any packages that were in the previous
        # interpreter; ``uv pip sync`` re-installs the ones listed in
        # ``uv.lock`` when present, otherwise leaves the environment empty
        # for the model to reinstall as needed.
        if (env_root / ".." / "pyproject.toml").is_file() and (
            self.context.workspace.workspace / "uv.lock"
        ).is_file():
            self._run_captured(
                [
                    "uv",
                    "pip",
                    "sync",
                    str(self.context.workspace.workspace / "uv.lock"),
                ],
                env_root=env_root,
                timeout=timeout,
                network=True,
            )
        self._register_managed_commands()
        return self._render_select_version(
            version=version,
            install_output=install_result,
            venv_output=venv_result,
            env_root=env_root,
        )

    def _pip(
        self,
        *,
        uv_subcommand: str,
        packages: tuple[str, ...],
        env_root: Path,
        dev: bool,
        timeout: int,
    ) -> str:
        """Run ``uv pip install`` or ``uv pip uninstall`` against the venv."""
        command: list[str] = ["uv", "pip", uv_subcommand]
        if uv_subcommand == "install" and dev:
            command.append("--extra")
            command.append("dev")
        command.extend(packages)
        result = self._run_captured(
            command,
            env_root=env_root,
            timeout=timeout,
            network=uv_subcommand == "install",
        )
        self._register_managed_commands()
        return self._render_pip(
            uv_subcommand=uv_subcommand,
            packages=packages,
            dev=dev,
            output=result,
        )

    def _sync(
        self,
        *,
        env_root: Path,
        timeout: int,
    ) -> str:
        """Reconcile ``env/python`` with the current project pyproject."""
        project = self._find_pyproject()
        if project is None:
            raise FileNotFoundError(
                "No pyproject.toml was found at the project root: "
                f"{self.context.workspace.workspace / 'pyproject.toml'}"
            )
        result = self._run_captured(
            [
                "uv",
                "sync",
                "--active",
                "--project",
                str(self.context.workspace.workspace),
            ],
            env_root=env_root,
            timeout=timeout,
            network=True,
        )
        self._register_managed_commands()
        return self._render_sync(
            project=project,
            output=result,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _packages(
        self,
        arguments: dict[str, Any],
    ) -> tuple[str, ...]:
        """Resolve the user's ``packages``/``package`` arguments."""
        packages = _parse_package_list(
            arguments.get("packages"),
            field_name="packages",
        )
        if packages:
            return packages
        return _parse_package_list(
            arguments.get("package"),
            field_name="package",
        )

    def _timeout(
        self,
        arguments: dict[str, Any],
        *,
        default: int,
    ) -> int:
        """Read the per-action timeout or fall back to the action default."""
        raw = arguments.get("timeout")
        if raw is None:
            return default
        if (
            not isinstance(raw, int)
            or isinstance(raw, bool)
            or not _MIN_TIMEOUT_SECONDS <= raw <= _MAX_TIMEOUT_SECONDS
        ):
            raise ValueError(
                f"'timeout' must be an integer between {_MIN_TIMEOUT_SECONDS} "
                f"and {_MAX_TIMEOUT_SECONDS} seconds."
            )
        return raw

    def _find_pyproject(self) -> Path | None:
        """Return the project ``pyproject.toml`` when one exists."""
        candidate = self.context.workspace.workspace / "pyproject.toml"
        return candidate if candidate.is_file() else None

    def _run_captured(
        self,
        command: list[str],
        *,
        env_root: Path,
        timeout: int,
        network: bool,
    ) -> str:
        """Run ``command`` in the workspace cwd and return the captured text."""
        result = self.context.sandbox.run(
            command,
            cwd=self.context.workspace.workspace,
            timeout=timeout,
            network=network,
        )
        text = result.output.strip()
        if result.timed_out:
            return self._format_failure(
                command=command,
                output=text,
                marker=f"timed out after {timeout}s",
            )
        if result.returncode != 0:
            return self._format_failure(
                command=command,
                output=text,
                marker=f"exit code {result.returncode}",
            )
        return text

    def _register_managed_commands(self) -> None:
        """Re-publish ``python`` and ``python3`` as staged runtime commands."""
        for command in ("python", "python3"):
            self.context.workspace.refresh_staged_command(command)

    def _network_reason(
        self,
        arguments: dict[str, Any],
        *,
        action: str,
    ) -> str:
        """Require an explicit reason for a network-backed uv action."""

        reason = arguments.get("reason")

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                f"action='{action}' requires network access, but no 'reason' "
                "was provided. Include a reason to allow uv network operations."
            )
        return reason.strip()

    def _authorize_network(
        self,
        arguments: dict[str, Any],
        *,
        action: str,
    ) -> str | None:
        """Request approval for a network-backed uv action when configured."""
        reason = self._network_reason(arguments, action=action)
        config = self.context.config.python
        if config.always_allow_network:
            return None
        permission = PromptUser(self.context)._execute(
            {
                "question": (
                    "Allow the managed Python tool to access the network?\n\n"
                    f"Action: {action}\n"
                    f"Reason: {reason}"
                ),
                "options": ["Allow once", "Deny"],
                "timeout": config.permission_timeout,
            }
        )
        if permission == "Allow once":
            return None
        return (
            "permission-denied: managed Python network access was not "
            "granted; uv was not executed."
        )

    @staticmethod
    def _format_failure(
        *,
        command: list[str],
        output: str,
        marker: str,
    ) -> str:
        """Render a failed command with its captured output."""
        rendered_command = " ".join(command)
        if not output:
            return f"$ {rendered_command}\n({marker})"
        return f"$ {rendered_command}\n{output}\n({marker})"

    @staticmethod
    def _render_status(
        *,
        env_root: Path,
        venv_python: Path,
        version_output: str,
        list_output: str,
        pyproject: Path | None,
    ) -> str:
        """Render the human-readable status payload."""
        lines: list[str] = [
            f"venv: {env_root}",
            f"interpreter: {venv_python}",
        ]
        if version_output:
            lines.append(f"version: {version_output}")
        if pyproject is not None:
            lines.append(f"pyproject: {pyproject}")
        else:
            lines.append(
                "pyproject: (none — sync will fail until a pyproject.toml "
                "is present at the project root)"
            )
        if list_output:
            lines.extend(("", "packages:", list_output))
        else:
            lines.extend(("", "packages: (uv pip list returned no output)"))
        return "\n".join(lines)

    @staticmethod
    def _render_select_version(
        *,
        version: str,
        install_output: str,
        venv_output: str,
        env_root: Path,
    ) -> str:
        """Render the result of ``select_version``."""
        lines: list[str] = [
            f"Selected Python version: {version}",
            f"venv: {env_root}",
        ]
        if install_output:
            lines.extend(("", "uv python install:", install_output))
        if venv_output:
            lines.extend(("", "uv venv --seed:", venv_output))
        lines.append(
            "Bare `python` and `python3` from Bash, Subprocess, and direct "
            "sandbox runs now resolve to this venv."
        )
        return "\n".join(lines)

    @staticmethod
    def _render_pip(
        *,
        uv_subcommand: str,
        packages: tuple[str, ...],
        dev: bool,
        output: str,
    ) -> str:
        """Render the result of an install or uninstall action."""
        verb = "Installed" if uv_subcommand == "install" else "Uninstalled"
        head = f"{verb} {len(packages)} package(s): " + ", ".join(packages)
        if uv_subcommand == "install" and dev:
            head += " (extra=dev)"
        if output:
            return f"{head}\n{output}"
        return head

    @staticmethod
    def _render_sync(
        *,
        project: Path,
        output: str,
    ) -> str:
        """Render the result of a sync action."""
        lines: list[str] = [f"Synced {project}"]
        if output:
            lines.extend(("", output))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Render a compact, single-line call log."""
        action = str(arguments.get("action", "?"))
        parts: list[str] = [f"action={action}"]
        if action == "select_version":
            version = arguments.get("version")
            if version is not None:
                parts.append(f"version={version}")
        elif action in {"install", "uninstall"}:
            packages = self._packages(arguments)
            if packages:
                preview = ", ".join(packages[:3])
                if len(packages) > 3:
                    preview += f", +{len(packages) - 3} more"
                parts.append(f"packages=[{preview}]")
            if action == "install" and arguments.get("dev"):
                parts.append("dev=true")
        if "timeout" in arguments:
            parts.append(f"timeout={arguments['timeout']}s")
        return " | ".join(parts)

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        """Render a compact, single-line result log."""
        text = str(result)
        if not text:
            return "empty"
        lines = text.splitlines()
        return f"{len(lines)} lines | {len(text)} chars"
