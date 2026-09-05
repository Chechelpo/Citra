from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from citra.sandbox.sandbox_mode import SandboxMode

if TYPE_CHECKING:
    from citra.config import SandboxPolicy
    from citra.context import WorkspaceContext


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SandboxResult:
    """Captured result of one foreground sandbox command."""

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
        workspace: WorkspaceContext | Path,
        policy: SandboxPolicy,
        *,
        base_environment: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the instance."""
        self.__workspace = workspace if not isinstance(workspace, Path) else None
        self.__source = (
            workspace.resolve()
            if isinstance(workspace, Path)
            else workspace.workspace
        )
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

        self._ensure_filesystem_environment()
        logger.info(
            "Workspace sandbox initialized",
            extra={
                "origin": __name__,
                "mode": policy.mode.name,
                "workspace": str(self.__source),
            },
        )

    @property
    def source(self) -> Path:
        """Return the writable copied-project root."""
        return self.__source

    @property
    def policy(self) -> SandboxPolicy:
        """Return the finalized policy consumed by this sandbox."""
        return self.__policy

    @property
    def mode(self) -> SandboxMode:
        """Return the workflow-selected sandbox mode."""
        return self.__policy.mode

    def allows_network(
        self,
        requested: bool,
    ) -> bool:
        """Apply the monotonic operator network restriction."""
        if not isinstance(requested, bool):
            raise TypeError(
                "requested network access must be boolean"
            )

        return (
            requested
            and not self.__policy.global_disallow_network
        )

    def resolve_command(self, command: str) -> Path | None:
        """Resolve a discovered or staged command in the isolated runtime."""
        if self.__workspace is None:
            logger.debug(
                "Sandbox command resolution has no workspace registry",
                extra={"origin": __name__, "command": command},
            )
            return None
        resolved = self.__workspace.resolve_command(command)
        logger.debug(
            "Resolved command through sandbox runtime",
            extra={
                "origin": __name__,
                "command": command,
                "found": resolved is not None,
            },
        )
        return resolved

    def readonly_binds(self) -> tuple[Path, ...]:
        """Return sandbox targets exposed read-only."""
        return tuple(target for _source, target in self.__policy.readonly_mounts)

    def readonly_mounts(self) -> tuple[tuple[Path, Path], ...]:
        """Return explicit host-source to sandbox-target read-only mounts."""
        return self.__policy.readonly_mounts

    def writable_binds(self) -> tuple[Path, ...]:
        """Return same-path writable roots available to sandboxed processes."""
        return tuple(
            dict.fromkeys(
                (
                    self.__source,
                    *self.__policy.writable_binds,
                )
            )
        )

    def masked_dirs(self) -> tuple[Path, ...]:
        """Return host directory targets replaced with empty tmpfs mounts."""
        return tuple(
            self.__policy.masked_host_dirs
        )

    def masked_files(self) -> tuple[Path, ...]:
        """Return host file targets replaced with ``/dev/null``."""
        return tuple(
            self.__policy.masked_host_files
        )

    def device_binds(self) -> tuple[Path, ...]:
        """Return explicitly allowed device bind paths."""
        return tuple(
            self.__policy.extra_device_binds
        )

    def private_files(self) -> tuple[Path, ...]:
        """Return private files masked after other policy mounts."""
        return tuple(
            self.__policy.private_files
        )

    def build_environment(
        self,
        base: Mapping[str, str] | None = None,
        *,
        overrides: Mapping[str, str] | None = None,
        path_prepend: Sequence[str | Path] = (),
    ) -> dict[str, str]:
        """Build a sanitized process environment with an immutable runtime PATH."""

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

        # Runtime-owned variables must survive sandbox execution.
        # These point to the Citra installation/configuration, not the workspace.
        runtime_environment = {
            "CITRA_ROOT",
            "CITRA_INSTALL_ROOT",
            "CITRA_CONFIG_PATH",
        }

        for name in self.__policy.drop_environment_variables:
            if (
                name not in explicitly_set
                and name not in runtime_environment
            ):
                environment.pop(
                    name,
                    None,
                )

        prefixes = self.__policy.drop_environment_prefixes

        if prefixes:
            for name in tuple(environment):
                if (
                    name in explicitly_set
                    or name in runtime_environment
                ):
                    continue

                if any(
                    name.startswith(prefix)
                    for prefix in prefixes
                ):
                    environment.pop(
                        name,
                        None,
                    )

        environment["CITRA_PROJECT_ROOT"] = str(
            self.__source
        )

        if "PATH" in self.__base_environment:
            canonical_path = self.__base_environment["PATH"]

            validated = self._validated_path_prepend(
                path_prepend
            )

            environment["PATH"] = os.pathsep.join(
                (
                    *validated,
                    canonical_path,
                )
            )

        environment.pop(
            "CITRA_SOURCE",
            None,
        )

        environment.pop(
            "CITRA_WORKSPACE",
            None,
        )

        return environment

    def iter_readonly_binds(
        self,
    ) -> Iterable[Path]:
        """Iterate read-only sandbox targets."""
        yield from self.readonly_binds()

    def iter_writable_binds(
        self,
    ) -> Iterable[Path]:
        """Iterate writable sandbox targets."""
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

        masked_dirs = tuple(
            path
            for path in policy.masked_host_dirs
            if path.exists()
        )
        masked_files = tuple(
            path
            for path in policy.masked_host_files
            if path.is_file()
        )
        readonly_mounts = _minimal_readonly_mounts(
            (
                mount
                for mount in policy.readonly_mounts
                if mount[0].exists()
            ),
            authoritative_targets=(
                path
                for path in policy.extra_ro_binds
                if path.is_dir()
            ),
        )
        writable_binds = tuple(
            path
            for path in self.writable_binds()
            if path.exists()
        )
        device_binds = tuple(
            path
            for path in policy.extra_device_binds
            if path.exists()
        )
        private_files = tuple(
            path
            for path in policy.private_files
            if path.is_file()
        )

        # Bubblewrap starts from an empty root. It can create the final mount
        # point, but only when every parent already exists in the namespace.
        # Runtime discovery deliberately exposes individual executables (for
        # example /usr/bin/cargo), so recreate their directory hierarchy
        # without broadening the sandbox by binding the host directories.
        args.extend(
            _mount_parent_arguments(masked_dirs)
        )

        for path in _parents_first(masked_dirs):
            args.extend(
                (
                    "--tmpfs",
                    str(path),
                )
            )

        args.extend(
            _mount_parent_arguments(masked_files)
        )

        for path in masked_files:
            args.extend(
                (
                    "--ro-bind",
                    "/dev/null",
                    str(path),
                )
            )

        args.extend(
            _mount_parent_arguments(
                (
                    *(target for _source, target in readonly_mounts),
                    *writable_binds,
                    *device_binds,
                    *private_files,
                )
            )
        )

        for source, target in readonly_mounts:
            args.extend(
                (
                    "--ro-bind",
                    str(source),
                    str(target),
                )
            )

        for path in _parents_first(writable_binds):
            args.extend(
                (
                    "--bind",
                    str(path),
                    str(path),
                )
            )

        for path in _parents_first(device_binds):
            args.extend(
                (
                    "--dev-bind",
                    str(path),
                    str(path),
                )
            )

        for path in private_files:
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

        resolved_command = self._resolve_command(command)
        bwrap_command = self.build_bwrap_arguments(
            command=resolved_command,
            cwd=cwd_path,
            network=network,
        )

        logger.debug(
            "Starting Bubblewrap command with %d argument(s).",
            len(bwrap_command),
            extra={"origin": __name__, "command": resolved_command[0]},
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
        path_prepend: Sequence[str | Path] = (),
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

        env = self.build_environment(
            overrides=environment,
            path_prepend=path_prepend,
        )
        resolved_command = self._resolve_command(command)
        bwrap_command = self.build_bwrap_arguments(
            command=resolved_command,
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
    
    def filesystem_environment(self) -> dict[str, str]:
        """
        Environment contract required by the Citra filesystem worker.

        ScopedFilesystem consumes these variables to construct its virtual
        filesystem aliases.
        """

        if self.__workspace is not None:
            environment = self.__workspace.environment()
            names = (
                "HOME",
                "CITRA_TMP",
                "CITRA_CACHE",
                "CITRA_ENV",
                "CITRA_RUNTIME",
                "XDG_CONFIG_HOME",
                "XDG_DATA_HOME",
                "XDG_RUNTIME_DIR",
            )
            return {name: environment[name] for name in names}
        runtime_root = self.__source / ".citra-runtime"
        return {
            "HOME": str(runtime_root / "home"),
            "CITRA_TMP": str(runtime_root / "tmp"),
            "CITRA_CACHE": str(runtime_root / "cache"),
            "CITRA_ENV": str(runtime_root / "env"),
            "CITRA_RUNTIME": str(runtime_root / "runtime"),
            "XDG_CONFIG_HOME": str(runtime_root / "config"),
            "XDG_DATA_HOME": str(runtime_root / "data"),
            "XDG_RUNTIME_DIR": str(runtime_root / "run"),
        }

    def _ensure_filesystem_environment(
        self,
    ) -> None:
        """Create fallback worker directories for path-only sandbox fixtures."""
        if self.__workspace is not None:
            return
        runtime_root = self.__source / ".citra-runtime"

        for directory in (
            runtime_root / "home",
            runtime_root / "tmp",
            runtime_root / "cache",
            runtime_root / "config",
            runtime_root / "data",
            runtime_root / "run",
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    def _resolve_command(self, command: Sequence[str]) -> tuple[str, ...]:
        """Resolve a bare executable through the canonical runtime launcher."""
        normalized = tuple(str(argument) for argument in command)
        executable = normalized[0]
        if "/" in executable or self.__workspace is None:
            return normalized
        resolved = self.resolve_command(executable)
        if resolved is None:
            logger.warning(
                "Command is not present in the isolated runtime",
                extra={"origin": __name__, "command": executable},
            )
            return normalized
        return (str(resolved), *normalized[1:])

    def _validated_path_prepend(
        self,
        entries: Sequence[str | Path],
    ) -> tuple[str, ...]:
        """Allow PATH extensions only from runtime or writable data roots."""
        validated: list[str] = []
        allowed = self.writable_binds()
        for raw in entries:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = self.__source / candidate
            candidate = candidate.absolute()
            if candidate == Path("/runtime") or _is_within(Path("/runtime"), candidate):
                validated.append(str(candidate))
                continue
            if not any(_is_within(root, candidate) for root in allowed):
                raise ValueError(
                    f"PATH entry is outside the isolated runtime: {candidate}"
                )
            validated.append(str(candidate))
        return tuple(dict.fromkeys(validated))

def _minimal_readonly_binds(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Remove read-only binds already covered by a directory bind."""
    result: list[Path] = []

    for path in _parents_first(paths):
        if any(
            parent.is_dir() and _is_within(parent, path)
            for parent in result
        ):
            continue
        result.append(path)

    return tuple(result)


def _minimal_readonly_mounts(
    mounts: Iterable[tuple[Path, Path]],
    *,
    authoritative_targets: Iterable[Path] = (),
) -> tuple[tuple[Path, Path], ...]:
    """Remove covered mounts and order read-only parents before consumers."""
    unique = dict.fromkeys(
        (source.absolute(), target.absolute())
        for source, target in mounts
    )
    authoritative = tuple(
        path.expanduser().absolute()
        for path in authoritative_targets
    )
    ordered = sorted(
        unique,
        key=lambda item: (
            0 if item[1] == Path("/runtime") else 1,
            len(item[1].parts),
            str(item[1]),
            str(item[0]),
        ),
    )
    result: list[tuple[Path, Path]] = []

    for source, target in ordered:
        authoritative_parent = next(
            (
                parent
                for parent in authoritative
                if source != target and _is_within(parent, target)
            ),
            None,
        )
        if authoritative_parent is not None:
            logger.debug(
                "Omitting runtime mount covered by an explicit read-only bind",
                extra={
                    "origin": __name__,
                    "source": str(source),
                    "target": str(target),
                    "covering_target": str(authoritative_parent),
                },
            )
            continue
        covering_mount = next(
            (
                (parent_source, parent_target)
                for parent_source, parent_target in result
                if _mount_covers_child(
                    parent_source,
                    parent_target,
                    source,
                    target,
                )
            ),
            None,
        )
        if covering_mount is not None:
            logger.debug(
                "Omitting redundant nested read-only mount",
                extra={
                    "origin": __name__,
                    "source": str(source),
                    "target": str(target),
                    "covering_target": str(covering_mount[1]),
                },
            )
            continue
        result.append((source, target))

    return tuple(result)


def _mount_covers_child(
    parent_source: Path,
    parent_target: Path,
    child_source: Path,
    child_target: Path,
) -> bool:
    """Return whether a parent mount already exposes the same child subtree."""
    if not parent_source.is_dir():
        return False

    try:
        relative_target = child_target.relative_to(parent_target)
    except ValueError:
        return False

    if not relative_target.parts:
        return child_source == parent_source

    return child_source == parent_source / relative_target


def _parents_first(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Return unique mount paths with parents ordered before descendants."""

    return tuple(
        sorted(
            dict.fromkeys(paths),
            key=lambda path: (
                len(path.parts),
                str(path),
            ),
        )
    )


def _mount_parent_arguments(paths: Iterable[Path]) -> tuple[str, ...]:
    """Build Bubblewrap ``--dir`` arguments for absolute mount parents."""

    parents: set[Path] = set()

    for path in paths:
        if not path.is_absolute():
            raise ValueError(
                f"Sandbox bind paths must be absolute: {path}"
            )

        current = path.parent
        while current != current.parent:
            parents.add(current)
            current = current.parent

    arguments: list[str] = []
    for parent in _parents_first(parents):
        arguments.extend(("--dir", str(parent)))

    return tuple(arguments)


def _is_within(root: Path, path: Path) -> bool:
    """Handle is within."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
