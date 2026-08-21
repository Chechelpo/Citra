from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import subprocess
from typing import Mapping, Sequence

from ..context.turn_workspace import WorkspaceContext


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    output: str
    timed_out: bool


class WorkspaceSandbox:
    def __init__(
        self,
        workspace: WorkspaceContext,
    ) -> None:
        self.__workspace = workspace

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout: int,
        network: bool,
        environment: Mapping[str, str] | None = None,
    ) -> SandboxResult:
        bwrap = shutil.which(
            "bwrap"
        )

        if bwrap is None:
            raise RuntimeError(
                "Bubblewrap is required for sandboxed "
                "execution but 'bwrap' was not found in PATH."
            )

        if not command:
            raise ValueError(
                "Sandbox command cannot be empty."
            )

        workspace = self.__workspace

        if cwd is None:
            cwd_path = workspace.workspace
        else:
            cwd_path = workspace.resolve_path(
                cwd
            )

        if not cwd_path.is_dir():
            raise NotADirectoryError(
                f"Sandbox working directory does not exist: {cwd_path}"
            )

        env = workspace.environment(
            environment
        )

        bwrap_command = [
            bwrap,

            "--die-with-parent",

            # Host filesystem is visible but immutable.
            "--ro-bind",
            "/",
            "/",

            # Entire turn-scoped agent environment is writable. The active
            # workspace is a child of this root.
            "--bind",
            str(workspace.root),
            str(workspace.root),

            # The original project is readable but never generally writable.
            "--ro-bind",
            str(workspace.source_workspace),
            str(workspace.source_workspace),

            # Give shell commands a real @source path from the default cwd.
            # A filesystem mount is reliable for quoting and compound shell
            # commands; rewriting command strings is not.
            "--ro-bind",
            str(workspace.source_workspace),
            str(workspace.workspace / "@source"),

            # Isolate process-visible host state.
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",

            "--proc",
            "/proc",

            "--dev",
            "/dev",

            "--chdir",
            str(cwd_path),
        ]

        if not network:
            bwrap_command.append(
                "--unshare-net"
            )

        bwrap_command.extend(
            command
        )

        proc = subprocess.Popen(
            bwrap_command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        try:
            output, _ = proc.communicate(
                timeout=timeout,
            )

            return SandboxResult(
                returncode=proc.returncode,
                output=output,
                timed_out=False,
            )

        except subprocess.TimeoutExpired:
            try:
                os.killpg(
                    proc.pid,
                    signal.SIGKILL,
                )
            except ProcessLookupError:
                pass

            output, _ = proc.communicate()

            return SandboxResult(
                returncode=proc.returncode,
                output=output,
                timed_out=True,
            )
