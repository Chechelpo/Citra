"""User-facing access to the copied project checkout."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

from .command import Command, CommandResult


class WorkspaceCommand(Command):
    """Show or enter the copied project checkout."""

    id = "workspace"
    description = "Show or enter the project checkout."

    def _run(self, args: str) -> CommandResult:
        raw = args.strip()
        action, _, remainder = raw.partition(" ")
        action = action.lower() or "path"
        if action in {"path", "show"}:
            if remainder:
                return self._usage()
            return CommandResult(output=self._path_output())
        if action == "shell":
            if remainder:
                return self._usage()
            return self._open_shell()
        return self._usage()

    @property
    def _project(self) -> Path:
        project = Path(self.context.workspace.workspace).resolve()
        if not project.is_dir():
            raise NotADirectoryError(
                f"Project checkout does not exist: {project}"
            )
        return project

    def _path_output(self) -> str:
        project = self._project
        return "\n".join(
            (
                f"Project checkout: {project}",
                "",
                "Open a user shell here with:",
                "  /workspace shell",
                "",
                "Preview and apply changes to the original source with:",
                "  /apply",
                "",
                "Or enter it from another terminal:",
                f"  cd -- {shlex.quote(str(project))}",
            )
        )

    def _open_shell(self) -> CommandResult:
        project = self._project
        shell = os.environ.get("SHELL")
        if not shell:
            shell = shutil.which("bash") or shutil.which("sh")
        if not shell:
            raise RuntimeError("No interactive shell is available.")

        environment = os.environ.copy()
        environment["CITRA_PROJECT_ROOT"] = str(project)
        completed = subprocess.run(
            [shell, "-i"],
            cwd=project,
            env=environment,
            check=False,
        )
        return CommandResult(
            output=(
                "Returned from the project checkout shell "
                f"(exit code {completed.returncode})."
            )
        )

    @staticmethod
    def _usage() -> CommandResult:
        return CommandResult(
            output="Usage: /workspace [path|shell]"
        )


__all__ = ["WorkspaceCommand"]
