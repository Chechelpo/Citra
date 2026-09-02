from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import subprocess
from typing import Iterable, Mapping, Sequence

from citra.config import SandboxPolicy
from citra.sandbox.sandbox_mode import SandboxMode


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    output: str
    timed_out: bool


class WorkspaceSandbox:
    """
    Execute commands using a finalized SandboxPolicy.

    The sandbox has exactly two construction inputs:
    - source
    - policy

    It has no knowledge of modes, workflows, config files, or runtime
    discovery. Those concerns must already have been folded into policy.
    """

    def __init__(
        self,
        source: str | Path,
        policy: SandboxPolicy,
        *,
        base_environment: Mapping[str, str] | None = None,
    ) -> None:
        self.__source = Path(
            source
        ).expanduser().resolve()

        self.__policy = policy
        self.__base_environment = dict(
            os.environ if base_environment is None else base_environment
        )

        if not self.__source.exists():
            raise FileNotFoundError(
                f"Sandbox source does not exist: {self.__source}"
            )

        if not self.__source.is_dir():
            raise NotADirectoryError(
                f"Sandbox source is not a directory: {self.__source}"
            )

    @property
    def source(self) -> Path:
        return self.__source

    @property
    def policy(self) -> SandboxPolicy:
        return self.__policy

    @property
    def mode(self) -> SandboxMode:
        return self.__policy.mode

    def allows_network(
        self,
        requested: bool,
    ) -> bool:
        if not isinstance(requested, bool):
            raise TypeError(
                "requested network access must be boolean"
            )

        return (
            requested
            and not self.__policy.global_disallow_network
        )

    def readonly_binds(self) -> tuple[Path, ...]:
        return self.__policy.readonly_binds

    def writable_binds(self) -> tuple[Path, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.__source,
                    *self.__policy.writable_binds,
                )
            )
        )

    def masked_dirs(self) -> tuple[Path, ...]:
        return tuple(
            self.__policy.masked_host_dirs
        )

    def masked_files(self) -> tuple[Path, ...]:
        return tuple(
            self.__policy.masked_host_files
        )

    def device_binds(self) -> tuple[Path, ...]:
        return tuple(
            self.__policy.extra_device_binds
        )

    def private_files(self) -> tuple[Path, ...]:
        return tuple(
            self.__policy.private_files
        )

    def build_environment(
        self,
        base: Mapping[str, str] | None = None,
        *,
        overrides: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        environment = dict(
            self.__base_environment
            if base is None
            else base
        )

        explicitly_set = set(
            overrides or ()
        )

        if overrides is not None:
            environment.update(
                overrides
            )

        for name in (
            self.__policy.drop_environment_variables
        ):
            if name not in explicitly_set:
                environment.pop(
                    name,
                    None,
                )

        prefixes = (
            self.__policy.drop_environment_prefixes
        )

        if prefixes:
            for name in tuple(environment):
                if name in explicitly_set:
                    continue

                if any(
                    name.startswith(prefix)
                    for prefix in prefixes
                ):
                    environment.pop(
                        name,
                        None,
                    )

        environment["CITRA_PROJECT_ROOT"] = str(self.__source)
        environment.pop("CITRA_SOURCE", None)
        environment.pop("CITRA_WORKSPACE", None)

        return environment

    def iter_readonly_binds(
        self,
    ) -> Iterable[Path]:
        yield from self.readonly_binds()

    def iter_writable_binds(
        self,
    ) -> Iterable[Path]:
        yield from self.writable_binds()

    def build_bwrap_arguments(
        self,
        *,
        command: Sequence[str],
        cwd: Path,
        network: bool,
    ) -> list[str]:
        """
        Build a complete Bubblewrap command.

        Every sandbox-policy decision comes from SandboxPolicy.
        """

        bwrap = shutil.which(
            "bwrap"
        )

        if bwrap is None:
            raise RuntimeError(
                "Bubblewrap is required for sandboxed execution but "
                "'bwrap' was not found in PATH."
            )

        policy = self.__policy

        args: list[str] = [
            bwrap,
            "--die-with-parent",
        ]

        if policy.new_terminal_session:
            args.append(
                "--new-session"
            )

        if policy.unshare_user_try:
            args.append(
                "--unshare-user-try"
            )

        if policy.unshare_pid:
            args.append(
                "--unshare-pid"
            )

        if policy.unshare_ipc:
            args.append(
                "--unshare-ipc"
            )

        if policy.unshare_uts:
            args.append(
                "--unshare-uts"
            )

        if policy.unshare_cgroup_try:
            args.append(
                "--unshare-cgroup-try"
            )

        if policy.disable_nested_user_namespaces:
            args.append(
                "--disable-userns"
            )

        if not network:
            args.append(
                "--unshare-net"
            )

        # Minimal synthetic process/device views.
        args.extend(
            (
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/dev/shm",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/var",
                "--tmpfs",
                "/var/tmp",
                "--tmpfs",
                "/run",
            )
        )

        for path in policy.masked_host_dirs:
            if path.exists():
                args.extend(
                    (
                        "--tmpfs",
                        str(path),
                    )
                )

        for path in policy.masked_host_files:
            if path.is_file():
                args.extend(
                    (
                        "--ro-bind",
                        "/dev/null",
                        str(path),
                    )
                )

        for path in policy.readonly_binds:
            if not path.exists():
                continue

            args.extend(
                (
                    "--ro-bind",
                    str(path),
                    str(path),
                )
            )

        for path in self.writable_binds():
            if not path.exists():
                continue

            args.extend(
                (
                    "--bind",
                    str(path),
                    str(path),
                )
            )

        for path in policy.extra_device_binds:
            if not path.exists():
                continue

            args.extend(
                (
                    "--dev-bind",
                    str(path),
                    str(path),
                )
            )

        for path in policy.private_files:
            if not path.is_file():
                continue

            args.extend(
                (
                    "--ro-bind",
                    "/dev/null",
                    str(path),
                )
            )

        args.extend(
            (
                "--chdir",
                str(cwd),
                "--",
                *command,
            )
        )

        return args

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
        network: bool = False,
        environment: Mapping[str, str] | None = None,
        input_text: str | None = None,
    ) -> SandboxResult:
        """
        Execute one command inside the sandbox.
        """

        if not command:
            raise ValueError(
                "Sandbox command cannot be empty."
            )

        if timeout <= 0:
            raise ValueError(
                "Sandbox timeout must be greater than zero."
            )

        if cwd is None:
            cwd_path = self.__source
        else:
            candidate = Path(cwd).expanduser()

            if candidate.is_absolute():
                cwd_path = candidate.resolve()
            else:
                cwd_path = (
                    self.__source / candidate
                ).resolve()

        if not cwd_path.is_dir():
            raise NotADirectoryError(
                f"Sandbox working directory does not exist: {cwd_path}"
            )

        if not any(
            _is_within(root, cwd_path)
            for root in self.writable_binds()
        ):
            raise ValueError(
                f"Sandbox working directory is outside writable roots: {cwd_path}"
            )

        network = self.allows_network(
            network
        )

        env = self.build_environment(
            overrides=environment,
        )

        bwrap_command = self.build_bwrap_arguments(
            command=command,
            cwd=cwd_path,
            network=network,
        )

        process = subprocess.Popen(
            bwrap_command,
            env=env,
            stdin=(
                subprocess.PIPE
                if input_text is not None
                else subprocess.DEVNULL
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        try:
            try:
                output, _ = process.communicate(
                    input=input_text,
                    timeout=timeout,
                )

                return SandboxResult(
                    returncode=process.returncode,
                    output=output,
                    timed_out=False,
                )

            except subprocess.TimeoutExpired:
                self.terminate_process(
                    process,
                    force=True,
                )

                output, _ = process.communicate()

                return SandboxResult(
                    returncode=process.returncode,
                    output=output,
                    timed_out=True,
                )

        finally:
            if process.poll() is None:
                self.terminate_process(
                    process,
                    force=True,
                )

    def popen(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        network: bool = False,
        environment: Mapping[str, str] | None = None,
    ) -> subprocess.Popen[bytes]:
        """Start a long-running sandboxed process owned by its caller."""
        if not command:
            raise ValueError("Sandbox command cannot be empty.")

        if cwd is None:
            cwd_path = self.__source
        else:
            candidate = Path(cwd).expanduser()
            cwd_path = (
                candidate.resolve()
                if candidate.is_absolute()
                else (self.__source / candidate).resolve()
            )
        if not cwd_path.is_dir():
            raise NotADirectoryError(
                f"Sandbox working directory does not exist: {cwd_path}"
            )
        if not any(
            _is_within(root, cwd_path)
            for root in self.writable_binds()
        ):
            raise ValueError(
                f"Sandbox working directory is outside writable roots: {cwd_path}"
            )

        env = self.build_environment(overrides=environment)
        bwrap_command = self.build_bwrap_arguments(
            command=command,
            cwd=cwd_path,
            network=self.allows_network(network),
        )
        return subprocess.Popen(
            bwrap_command,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

    @staticmethod
    def terminate_process(
        process: subprocess.Popen[str] | subprocess.Popen[bytes],
        *,
        force: bool = False,
        grace_seconds: float = 1.0,
    ) -> None:
        """
        Terminate the complete sandbox process group.
        """

        if process.poll() is not None:
            return

        try:
            os.killpg(
                process.pid,
                (
                    signal.SIGKILL
                    if force
                    else signal.SIGTERM
                ),
            )
        except ProcessLookupError:
            return

        if force:
            try:
                process.wait(
                    timeout=0.2
                )
            except subprocess.TimeoutExpired:
                pass

            return

        try:
            process.wait(
                timeout=max(
                    0.0,
                    grace_seconds,
                )
            )
            return

        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(
                process.pid,
                signal.SIGKILL,
            )
        except ProcessLookupError:
            return

        try:
            process.wait(
                timeout=1.0
            )
        except subprocess.TimeoutExpired:
            pass


def _is_within(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
