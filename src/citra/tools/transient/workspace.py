"""Safe, project-scoped workspace recovery operations."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
from typing import Any, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..tool import Tool, ToolDefinition


_PATH_MAGIC = re.compile(r"[*?\[\]{}]")


class Workspace(Tool):
    """Restore explicitly selected tracked files without creating a commit."""

    TOOL_ID = "workspace"
    MAX_PATHS = 50

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="workspace",
            description=(
                "Manage recoverable state in the current project. The rollback "
                "operation restores only the explicitly named tracked files to "
                "HEAD, including their staged and working-tree changes. It does "
                "not create commits and never accepts directories or globs."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="operation",
                        schema=JsonSchema.string(
                            description="Workspace operation to perform.",
                            enum=("rollback",),
                        ),
                    ),
                    JsonProperty(
                        name="paths",
                        schema=JsonSchema.array(
                            JsonSchema.string(
                                description=(
                                    "Exact project-relative path of a tracked "
                                    "file to restore."
                                )
                            ),
                            description=(
                                "One or more exact tracked file paths. "
                                "Directories, globs, and untracked files are "
                                "rejected."
                            ),
                        ),
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        del context
        return (ToolDefinition(definition=cls.DEFINITION),)

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        operation = arguments.get("operation")
        if operation != "rollback":
            raise ValueError("'operation' must be 'rollback'.")

        paths = arguments.get("paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError("'paths' must be a non-empty array.")
        if len(paths) > self.MAX_PATHS:
            raise ValueError(
                f"At most {self.MAX_PATHS} files may be rolled back at once."
            )
        if not self.context.has_command("git"):
            raise RuntimeError("Git is not available in the current project.")

        project_root = self.context.workspace.workspace.resolve()
        repository_root = self._repository_root(project_root)
        normalized = self._normalize_paths(
            paths,
            project_root=project_root,
            repository_root=repository_root,
        )
        pathspecs = [f":(literal){path}" for path in normalized]

        tracked = self._run_git(
            repository_root,
            ("ls-files", "--error-unmatch", "--", *pathspecs),
        )
        if tracked.timed_out:
            raise TimeoutError("Workspace rollback validation timed out.")
        if tracked.returncode != 0:
            raise ValueError(
                "Rollback accepts tracked files only; at least one selected "
                "path is untracked or unknown."
            )

        result = self._run_git(
            repository_root,
            (
                "restore",
                "--source=HEAD",
                "--staged",
                "--worktree",
                "--",
                *pathspecs,
            ),
        )
        if result.timed_out:
            raise TimeoutError("Workspace rollback timed out.")
        if result.returncode != 0:
            detail = result.output.strip() or "git restore failed"
            raise RuntimeError(f"Workspace rollback failed: {detail}")

        shown = [
            self.context.workspace.display_path(repository_root / path)
            for path in normalized
        ]
        return (
            f"Rolled back {len(shown)} tracked "
            f"{'file' if len(shown) == 1 else 'files'} to HEAD:\n"
            + "\n".join(f"- {path}" for path in shown)
        )

    def _repository_root(self, project_root: Path) -> Path:
        result = self._run_git(
            project_root,
            ("rev-parse", "--show-toplevel"),
        )
        if result.timed_out:
            raise TimeoutError("Git repository discovery timed out.")
        if result.returncode != 0:
            raise RuntimeError("The current project is not a Git repository.")

        raw = result.output.strip().splitlines()
        if not raw:
            raise RuntimeError("Git did not return a repository root.")
        repository_root = Path(raw[-1]).resolve()
        try:
            project_root.relative_to(repository_root)
        except ValueError as error:
            raise RuntimeError(
                "Git repository root is outside the current project."
            ) from error
        return repository_root

    def _normalize_paths(
        self,
        values: list[Any],
        *,
        project_root: Path,
        repository_root: Path,
    ) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Rollback paths must be non-empty strings.")
            raw = value.strip()
            if any(character in raw for character in ("\x00", "\n", "\r", "\\")):
                raise ValueError("Rollback paths contain invalid characters.")
            candidate = PurePosixPath(raw)
            if (
                candidate.is_absolute()
                or candidate in {PurePosixPath("."), PurePosixPath("..")}
                or ".." in candidate.parts
                or ".git" in candidate.parts
                or _PATH_MAGIC.search(raw)
            ):
                raise ValueError(
                    "Rollback requires exact project-relative file paths."
                )

            resolved = self.context.workspace.resolve_path(candidate.as_posix())
            try:
                resolved.relative_to(project_root)
            except ValueError as error:
                raise ValueError("Rollback path escapes the current project.") from error
            if resolved.is_dir():
                raise ValueError("Rollback paths must identify files, not directories.")

            relative = resolved.relative_to(repository_root).as_posix()
            if relative not in normalized:
                normalized.append(relative)

        return tuple(normalized)

    def _run_git(self, cwd: Path, arguments: tuple[str, ...]):
        return self.context.sandbox.run(
            [
                "git",
                "--no-pager",
                "--no-optional-locks",
                "-c",
                "credential.helper=",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                *arguments,
            ],
            cwd=cwd,
            timeout=30,
            network=False,
            environment={
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_ASKPASS": "/bin/false",
                "SSH_ASKPASS": "/bin/false",
            },
        )

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        paths = arguments.get("paths")
        count = len(paths) if isinstance(paths, list) else 0
        return f"operation={arguments.get('operation', '?')} | files={count}"
