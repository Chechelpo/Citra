"""Configured lint checks for modified workspace files."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import shlex
import tomllib
from typing import TYPE_CHECKING, Any

from citra.context.config_loader import LintContextConfig, LintRuleConfig

if TYPE_CHECKING:
    from citra.context.turn_workspace import WorkspaceContext
    from citra.utils.sandbox import WorkspaceSandbox


_PLACEHOLDERS = (
    ("{path}", "path"),
    ("{relative_path}", "relative_path"),
    ("{workspace}", "workspace"),
    ("{source}", "source"),
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

    1. A supported lint policy declared by the permanent source project's
       ``pyproject.toml``.
    2. Citra's global ``linting.toml`` policy.
    3. No linting.

    Source policy is read from ``@source`` so an agent cannot weaken lint
    enforcement merely by editing a staged project configuration.
    """

    def __init__(
        self,
        workspace: WorkspaceContext,
        sandbox: WorkspaceSandbox,
        config: LintContextConfig,
    ) -> None:
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
            # Lint policy applies only to the editable workspace, not @source
            # or lifecycle scratch/config directories.
            return None

        config = self._effective_config(relative)
        if not config.enabled or not config.rules:
            return None

        relative_path = relative.as_posix()
        failures: list[str] = []

        for rule in config.rules:
            if not self._matches(rule, relative_path):
                continue

            command = tuple(
                self._expand(argument, path, relative_path)
                for argument in rule.command
            )
            cwd_raw = self._expand(rule.cwd, path, relative_path)

            try:
                cwd = self.workspace.resolve_path(cwd_raw)
                result = self.sandbox.run(
                    command,
                    cwd=cwd,
                    timeout=config.timeout,
                    network=False,
                )
            except Exception as error:
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
        project = self._source_project_config(relative)
        if project is not None:
            return project
        return self.config

    def _source_project_config(
        self,
        relative: Path,
    ) -> LintContextConfig | None:
        """Detect the nearest supported ``pyproject.toml`` policy in ``@source``.

        The editable workspace may represent an assistant filesystem rather than
        one repository. Detection therefore starts beside the corresponding
        source file and walks toward the source root. Ruff is currently the
        auto-detected project linter. An explicit ``[tool.ruff.lint]`` table is
        required; ``[tool.ruff.format]`` additionally enables format checking.
        """
        source_root = self.workspace.source_workspace.resolve()
        source_path = (source_root / relative).resolve()
        start = source_path if source_path.is_dir() else source_path.parent

        for project_root in self._ancestors_within(start, source_root):
            pyproject = project_root / "pyproject.toml"
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
                project_root=project_root,
                ruff=ruff,
            )

        return None

    @staticmethod
    def _ancestors_within(start: Path, root: Path) -> tuple[Path, ...]:
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
        # staged file rather than the permanent source copy.
        project_relative = project_root.relative_to(
            self.workspace.source_workspace.resolve()
        )
        writable_project_root = (
            self.workspace.workspace / project_relative
        ).resolve()
        config_path = str(pyproject.resolve())
        rules = [
            LintRuleConfig(
                name="ruff-project",
                command=(
                    "ruff",
                    "check",
                    "--force-exclude",
                    "--config",
                    config_path,
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
        values = {
            "path": str(path),
            "relative_path": relative_path,
            "workspace": str(self.workspace.workspace),
            "source": str(self.workspace.source_workspace),
        }
        expanded = template
        for placeholder, name in _PLACEHOLDERS:
            expanded = expanded.replace(placeholder, values[name])
        return expanded

    @staticmethod
    def _glob_match(relative_path: str, pattern: str) -> bool:
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
        return f"[{rule.name}] $ {shlex.join(command)}\n{details}"

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        omitted = len(text) - limit
        return text[:limit] + f"\n... <truncated {omitted} characters>"
