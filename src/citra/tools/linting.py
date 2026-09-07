"""Configured lint checks for modified workspace files."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
import shlex
import tomllib
from typing import TYPE_CHECKING, Any

from citra.config import LintContextConfig, LintRuleConfig

if TYPE_CHECKING:
    from citra.context.session_context import WorkspaceContext
    from citra.sandbox.sandbox import WorkspaceSandbox


# Single source of truth for the supported lint-placeholder set. The
# ``{path}``, ``{relative_path}``, ``{project}``, and ``{workspace}`` tokens
# are documented in ``src/citra/context/AGENTS.md``. ``{workspace}`` is an
# exact alias for ``{project}`` (copied project root). Any other ``{...}``
# token is rejected at expansion time so the failure surfaces as a clear
# lint error instead of being joined onto the sandbox source directory as
# a relative path.
# Author: adamant-disaster
_SUPPORTED_PLACEHOLDERS: tuple[str, ...] = (
    "{path}",
    "{relative_path}",
    "{project}",
    "{workspace}",
)

_PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")

# Hardcoded post-edit pyrefly policy. This rule is Citra-owned and runs
# without any model-written ``pyrefly.toml`` or ``[tool.pyrefly]`` section.
# ``--preset default`` surfaces default-level errors (for example
# ``bad-assignment``) that the no-config ``basic`` preset hides, while
# ``--search-path {project}`` resolves sibling workspace imports from the
# copied project tree. Third-party imports resolve through the
# sandbox-provisioned interpreter site-packages (see PythonRuntimeDiscovery),
# so no host absolute paths are embedded here. ``cwd`` stays at ``{project}``
# so configuration-relative resolution keeps the copied-project shape.
# Author: liberating-potato
_HARDCODED_PYREFLY_RULE_NAME = "pyrefly"
_HARDCODED_PYREFLY_COMMAND: tuple[str, ...] = (
    "pyrefly",
    "check",
    "--preset",
    "default",
    "--search-path",
    "{project}",
    "{path}",
)
_HARDCODED_PYREFLY_CWD = "{project}"
_HARDCODED_PYREFLY_INCLUDE: tuple[str, ...] = (
    "**/*.py",
    "**/*.pyi",
    "**/*.pyw",
)
_MISSING_PYREFLY_MARKERS: tuple[str, ...] = (
    "no such file",
    "not found",
    "execvp",
    "command not found",
    "enoent",
)

_PROJECT_RUFF_FILES = (
    "**/*.py",
    "**/*.pyi",
    "**/*.pyw",
    "**/*.ipynb",
    "pyproject.toml",
    "**/pyproject.toml",
)


class LintRunner:
    """Run the highest-precedence lint policy against one modified file.

    Policy precedence is intentionally exclusive rather than additive:

    1. A supported lint policy declared by the current project's
       ``pyproject.toml``.
    2. Citra's global ``linting.toml`` policy.
    3. No linting.

    Project policy is read from the same project tree the model edits.
    """

    def __init__(
        self,
        workspace: WorkspaceContext,
        sandbox: WorkspaceSandbox,
        config: LintContextConfig,
    ) -> None:
        """Initialize the instance."""
        self.workspace = workspace
        self.sandbox = sandbox
        self.config = config

    def lint_for_path(self, path_raw: str) -> str | None:
        """Return lint failures for a project file, or ``None`` when clean/inactive."""
        # ``lint.enabled`` is the master switch for post-edit linting. Project
        # auto-detection must not bypass an operator who explicitly disabled
        # lint enforcement in the global Citra configuration.
        if not self.config.enabled:
            return None

        path = self.workspace.resolve_path(path_raw)
        project_root = self.workspace.workspace

        try:
            relative = path.relative_to(project_root)
        except ValueError:
            # Lint policy applies only to the current project, not lifecycle
            # scratch/config directories.
            return None

        config = self._effective_config(relative)
        if not config.enabled:
            return None

        rules = self._effective_rules(config)
        if not rules:
            return None

        relative_path = relative.as_posix()
        failures: list[str] = []

        for rule in rules:
            is_hardcoded_pyrefly = self._is_hardcoded_pyrefly_rule(rule)
            if not self._matches(rule, relative_path):
                continue

            command = tuple(
                self._expand(argument, path, relative_path)
                for argument in rule.command
            )
            cwd_raw = self._expand(rule.cwd, path, relative_path)

            unsupported_argument: str | None = next(
                (
                    token
                    for token in (
                        self._unsupported_placeholder(argument)
                        for argument in command
                    )
                    if token is not None
                ),
                None,
            )
            if unsupported_argument is not None:
                failures.append(
                    self._format_failure(
                        rule,
                        command,
                        "lint execution failed: lint rule "
                        f"'{rule.name}' uses unsupported placeholder "
                        f"'{unsupported_argument}' in a command argument. "
                        "Supported placeholders: "
                        + ", ".join(_SUPPORTED_PLACEHOLDERS)
                        + ".",
                    )
                )
                continue

            unsupported_cwd = self._unsupported_placeholder(cwd_raw)
            if unsupported_cwd is not None:
                failures.append(
                    self._format_failure(
                        rule,
                        command,
                        "lint execution failed: lint rule "
                        f"'{rule.name}' uses unsupported placeholder "
                        f"'{unsupported_cwd}' in cwd. "
                        "Supported placeholders: "
                        + ", ".join(_SUPPORTED_PLACEHOLDERS)
                        + ".",
                    )
                )
                continue

            try:
                cwd = self.workspace.resolve_path(cwd_raw)
                result = self.sandbox.run(
                    command,
                    cwd=cwd,
                    timeout=config.timeout,
                    network=False,
                )
            except Exception as error:
                if is_hardcoded_pyrefly and self._is_missing_pyrefly_error(error):
                    continue
                failures.append(
                    self._format_failure(
                        rule,
                        command,
                        f"lint execution failed: {error}",
                    )
                )
                continue

            if result.returncode == 0 and not result.timed_out:
                continue

            output = result.output.strip()
            if is_hardcoded_pyrefly and self._is_missing_pyrefly_output(output):
                continue
            if result.timed_out:
                marker = f"timed out after {config.timeout}s"
            else:
                marker = f"exit code {result.returncode}"

            details = marker if not output else f"{output}\n({marker})"
            failures.append(
                self._format_failure(
                    rule,
                    command,
                    details,
                )
            )

        if not failures:
            return None

        text = f"Lint violations for {relative_path}:\n" + "\n\n".join(failures)
        return self._truncate(text, config.max_output_length)

    def _effective_config(self, relative: Path) -> LintContextConfig:
        """Handle effective config."""
        project = self._project_config(relative)
        if project is not None:
            return project
        return self.config

    def _effective_rules(self, config: LintContextConfig) -> tuple[LintRuleConfig, ...]:
        """Return configured rules plus the hardcoded pyrefly rule.

        Parameters:
            config: Effective lint config selected for the edited file.

        What it does:
            Merges the Citra-owned pyrefly single-file check additively
            after operator-configured rules. The hardcoded rule is skipped
            when the effective config already invokes pyrefly or when the
            isolated runtime does not provide a pyrefly executable, so
            operator intent and offline environments keep prior behavior.

        Returns:
            Configured rules with the hardcoded pyrefly rule appended when
            applicable.

        Author: liberating-potato
        """
        if self._has_pyrefly_rule(config.rules):
            return config.rules
        if not self._is_pyrefly_available():
            return config.rules
        return (*config.rules, self._hardcoded_pyrefly_rule())

    @staticmethod
    def _hardcoded_pyrefly_rule() -> LintRuleConfig:
        """Return the Citra-owned single-file pyrefly rule.

        What it does:
            Builds the workspace-relative pyrefly invocation that type
            checks one edited Python file with default-level diagnostics
            and project-root import resolution.

        Returns:
            Lint rule running ``pyrefly check --preset default
            --search-path {project} {path}`` from ``{project}`` for
            Python sources only.

        Author: liberating-potato
        """
        return LintRuleConfig(
            name=_HARDCODED_PYREFLY_RULE_NAME,
            command=_HARDCODED_PYREFLY_COMMAND,
            include=_HARDCODED_PYREFLY_INCLUDE,
            exclude=(),
            cwd=_HARDCODED_PYREFLY_CWD,
        )

    @staticmethod
    def _has_pyrefly_rule(rules: tuple[LintRuleConfig, ...]) -> bool:
        """Return whether any configured rule already invokes pyrefly.

        Parameters:
            rules: Effective configured lint rules for the edited file.

        What it does:
            Inspects configured command tokens for a pyrefly executable so
            the hardcoded check does not duplicate operator policy.

        Returns:
            True when a configured command already covers pyrefly.

        Author: liberating-potato
        """
        for rule in rules:
            for part in rule.command:
                if "pyrefly" in part:
                    return True
        return False

    def _is_hardcoded_pyrefly_rule(self, rule: LintRuleConfig) -> bool:
        """Return whether a rule is the Citra-owned pyrefly check.

        Parameters:
            rule: Lint rule selected for the edited file.

        What it does:
            Matches the hardcoded rule by name, command, and cwd so an
            operator rule with a colliding name is not mistaken for the
            Citra-owned check.

        Returns:
            True only for the hardcoded pyrefly rule shape.

        Author: liberating-potato
        """
        return (
            rule.name == _HARDCODED_PYREFLY_RULE_NAME
            and tuple(rule.command) == _HARDCODED_PYREFLY_COMMAND
            and rule.cwd == _HARDCODED_PYREFLY_CWD
        )

    def _is_pyrefly_available(self) -> bool:
        """Return whether the isolated runtime provides pyrefly.

        What it does:
            Resolves ``pyrefly`` through the workspace or sandbox command
            resolver. Fixtures without a resolver assume availability so
            the hardcoded command still executes under test doubles.

        Returns:
            False when a resolver affirmatively reports pyrefly missing;
            otherwise True.

        Author: liberating-potato
        """
        resolver = getattr(self.workspace, "resolve_command", None)
        if callable(resolver):
            resolved = resolver("pyrefly")
            return resolved is not None
        sandbox_resolver = getattr(self.sandbox, "resolve_command", None)
        if callable(sandbox_resolver):
            resolved = sandbox_resolver("pyrefly")
            return resolved is not None
        return True

    @staticmethod
    def _is_missing_pyrefly_output(output: str) -> bool:
        """Return whether sandbox output reports a missing pyrefly binary.

        Parameters:
            output: Stripped combined stdout/stderr from the sandbox run.

        What it does:
            Requires both a pyrefly token and an executable-missing marker
            (for example Bubblewrap ``execvp`` output) so genuine
            ``missing-import`` diagnostics are never mistaken for a
            missing binary.

        Returns:
            True when the output indicates pyrefly itself is unavailable.

        Author: liberating-potato
        """
        lowered = output.lower()
        if "pyrefly" not in lowered:
            return False
        return any(marker in lowered for marker in _MISSING_PYREFLY_MARKERS)

    @staticmethod
    def _is_missing_pyrefly_error(error: Exception) -> bool:
        """Return whether a run exception reports a missing pyrefly binary.

        Parameters:
            error: Exception raised while launching the sandbox command.

        What it does:
            Applies the same pyrefly-plus-marker test used for sandbox
            output to exception text, so a missing launcher skips the
            hardcoded check instead of failing the edit.

        Returns:
            True when the exception indicates pyrefly itself is missing.

        Author: liberating-potato
        """
        lowered = f"{type(error).__name__}: {error}".lower()
        if "pyrefly" not in lowered:
            return False
        return any(marker in lowered for marker in _MISSING_PYREFLY_MARKERS)

    def _project_config(
        self,
        relative: Path,
    ) -> LintContextConfig | None:
        """Detect the nearest supported project ``pyproject.toml`` policy.

        The editable workspace may represent an assistant filesystem rather than
        one repository. Detection therefore starts beside the corresponding
        project file and walks toward the project root. Ruff is currently the
        auto-detected project linter. An explicit ``[tool.ruff.lint]`` table is
        required; ``[tool.ruff.format]`` additionally enables format checking.

        This read-only policy detection is intentionally separate from
        Python dependency management: the model-facing
        :class:`citra.tools.transient.python.Python` tool owns the
        ``env/python`` venv and the ``pyproject.toml`` install/sync path,
        while :class:`LintRunner` only inspects project policy for static
        analysis. There is no duplicate install or venv logic here.
        """
        project_root = self.workspace.workspace.resolve()
        project_path = (project_root / relative).resolve()
        start = project_path if project_path.is_dir() else project_path.parent

        for candidate_root in self._ancestors_within(start, project_root):
            pyproject = candidate_root / "pyproject.toml"
            if not pyproject.is_file():
                continue

            try:
                with pyproject.open("rb") as file:
                    raw: dict[str, Any] = tomllib.load(file)
            except (OSError, tomllib.TOMLDecodeError):
                continue

            tool = raw.get("tool")
            if not isinstance(tool, dict):
                continue
            ruff = tool.get("ruff")
            if not isinstance(ruff, dict):
                continue
            ruff_lint = ruff.get("lint")
            if not isinstance(ruff_lint, dict):
                continue

            return self._ruff_project_config(
                pyproject=pyproject,
                project_root=candidate_root,
                ruff=ruff,
            )

        return None

    @staticmethod
    def _ancestors_within(start: Path, root: Path) -> tuple[Path, ...]:
        """Handle ancestors within."""
        ancestors: list[Path] = []
        current = start.resolve()
        root = root.resolve()

        while True:
            try:
                current.relative_to(root)
            except ValueError:
                break
            ancestors.append(current)
            if current == root:
                break
            current = current.parent

        return tuple(ancestors)

    def _ruff_project_config(
        self,
        *,
        pyproject: Path,
        project_root: Path,
        ruff: dict[str, Any],
    ) -> LintContextConfig:
        # With an explicit --config, Ruff resolves configuration-relative paths
        # against cwd. Use the corresponding writable project root so project
        # globs and src paths keep the same relative shape while Ruff checks the
        # file in the current project.
        """Handle ruff project config."""
        writable_project_root = project_root.resolve()
        config_path = str(pyproject.resolve())
        rules = [
            LintRuleConfig(
                name="ruff-project",
                command = (
                    "ruff",
                    "format",
                    "--check",
                    "--force-exclude",
                    "{path}",
                ),
                include=_PROJECT_RUFF_FILES,
                cwd=str(writable_project_root),
            )
        ]

        if isinstance(ruff.get("format"), dict):
            rules.append(
                LintRuleConfig(
                    name="ruff-format-project",
                    command=(
                        "ruff",
                        "format",
                        "--check",
                        "--force-exclude",
                        "--config",
                        config_path,
                        "{path}",
                    ),
                    include=_PROJECT_RUFF_FILES,
                    cwd=str(writable_project_root),
                )
            )

        return LintContextConfig(
            enabled=True,
            rules=tuple(rules),
        )

    def _expand(
        self,
        template: str,
        path: Path,
        relative_path: str,
    ) -> str:
        """Substitute supported ``{...}`` tokens in ``template``.

        The supported set is :data:`_SUPPORTED_PLACEHOLDERS`. ``{workspace}``
        is an exact alias for ``{project}``. Unsupported tokens are
        intentionally left in place so :meth:`_unsupported_placeholder` can
        report them.

        Author: adamant-disaster
        """
        project_root = str(self.workspace.workspace)
        values = {
            "{path}": str(path),
            "{relative_path}": relative_path,
            "{project}": project_root,
            "{workspace}": project_root,
        }
        expanded = template
        for placeholder in _SUPPORTED_PLACEHOLDERS:
            expanded = expanded.replace(placeholder, values[placeholder])
        return expanded

    @staticmethod
    def _unsupported_placeholder(expanded: str) -> str | None:
        """Return the first unsupported ``{...}`` token in ``expanded``, if any.

        The expansion stage is the only stage that can recognise project
        paths; reporting the offending token here keeps unsupported
        placeholders from being silently passed to the sandbox as relative
        paths.
        """
        match = _PLACEHOLDER_RE.search(expanded)
        if match is None:
            return None
        token = match.group(0)
        if token in _SUPPORTED_PLACEHOLDERS:
            return None
        return token

    @staticmethod
    def _glob_match(relative_path: str, pattern: str) -> bool:
        """Handle glob match."""
        path = PurePosixPath(relative_path)
        normalized = pattern.removeprefix("./")
        candidates = {normalized}
        pending = [normalized]

        # pathlib treats **/ as one-or-more directories in several useful
        # cases. Project glob configuration conventionally expects **/ to also
        # match zero directories, so test those reduced variants explicitly.
        while pending:
            candidate = pending.pop()
            marker = candidate.find("**/")
            if marker == -1:
                continue
            reduced = candidate[:marker] + candidate[marker + 3 :]
            if reduced not in candidates:
                candidates.add(reduced)
                pending.append(reduced)

        if any(path.match(candidate) for candidate in candidates):
            return True

        if normalized.endswith("/**"):
            prefix = normalized[:-3].rstrip("/")
            return relative_path.startswith(prefix + "/")

        return False

    def _matches(
        self,
        rule: LintRuleConfig,
        relative_path: str,
    ) -> bool:
        """Handle matches."""
        included = any(
            self._glob_match(relative_path, pattern) for pattern in rule.include
        )
        if not included:
            return False
        return not any(
            self._glob_match(relative_path, pattern) for pattern in rule.exclude
        )

    @staticmethod
    def _format_failure(
        rule: LintRuleConfig,
        command: tuple[str, ...],
        details: str,
    ) -> str:
        """Handle format failure."""
        return f"[{rule.name}] $ {shlex.join(command)}\n{details}"

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        """Handle truncate."""
        if len(text) <= limit:
            return text
        omitted = len(text) - limit
        return text[:limit] + f"\n... <truncated {omitted} characters>"
