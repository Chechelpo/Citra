from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    from ..context.turn_workspace import WorkspaceContext


# ---------------------------------------------------------------------------
# Sandbox policy
# ---------------------------------------------------------------------------
#
# The default policy deliberately favors compatibility with normal developer
# commands over a minimal root filesystem:
#
#   1. The host root is visible read-only, preserving distro-specific layouts,
#      dynamic linkers, /usr/local, /opt toolchains, CA stores, etc.
#   2. Sensitive / stateful host trees are then masked.
#   3. Only Citra-owned lifecycle directories are rebound writable.
#   4. The original source workspace is rebound read-only both at its original
#      absolute path and at <agent workspace>/@source.
#
# Bubblewrap applies filesystem operations in command-line order, so later
# mounts intentionally override earlier ones.

# Read-only host paths that form the compatibility baseline. Keeping "/" here
# is the least distro-fragile option. If you later want a strict allow-list
# root, replace this with explicit paths such as /usr, /bin, /lib*, /etc, /opt
# and test it on every target distro/toolchain.
BASE_READONLY_BINDS: tuple[str, ...] = (
    "/",
)

# Host directories hidden after BASE_READONLY_BINDS are mounted. These are
# masked with empty tmpfs mounts. Do not add /etc or /opt casually: many normal
# developer tools depend on distro configuration, CA stores, loaders, or
# language runtimes there.
MASKED_HOST_DIRS: tuple[str, ...] = (
    "/home",
    "/root",
    "/run",
    "/tmp",
    "/var/tmp",
    "/mnt",
    "/media",
    "/srv",
    "/boot",
)

# Optional host files to hide. Files must be masked differently from
# directories, so keep them in a separate list. /dev/null is mounted over each
# existing file.
MASKED_HOST_FILES: tuple[str, ...] = ()

# Citra lifecycle-root directories which sandboxed commands may mutate. Importantly,
# "state" is NOT here: baseline/index/git bookkeeping remains control-plane
# state and is never exposed writable to arbitrary commands.
SANDBOX_WRITABLE_DIRS: tuple[str, ...] = (
    "workspace",
    "cache",
    "config",
    "data",
    "home",
    "runtime",
    "tmp",
)

# Extra same-path host mounts for local customization. Use these instead of
# reopening the whole host home. Typical examples might be one specific SDK or
# one NVM version directory.
EXTRA_READONLY_BINDS: tuple[str, ...] = ()
EXTRA_WRITABLE_BINDS: tuple[str, ...] = ()
EXTRA_DEVICE_BINDS: tuple[str, ...] = ()

# Citra's complete installation is reopened read-only as one authoritative
# mount after /home is masked. Mounting src, .venv, and support directories
# independently proved fragile. The operator configuration is masked again
# after the installation mount.
CITRA_PRIVATE_CONFIG_FILE = "config.toml"

# If PATH contains executable directories beneath a masked host tree (for
# example ~/.local/bin or ~/.cargo/bin), expose exactly those PATH directories
# read-only. This preserves many user-installed command launchers without
# reopening the rest of $HOME. Some runtimes need sibling directories too
# (notably NVM/npm or rustup); add their runtime root to EXTRA_READONLY_BINDS.
AUTO_BIND_MASKED_PATH_ENTRIES: bool = True

# Some environments point TLS libraries directly at a CA file/directory. If
# that target lives under a masked directory, reopen only that path read-only.
AUTO_BIND_ENV_PATHS: tuple[str, ...] = (
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "PLAYWRIGHT_BROWSERS_PATH",
    "NODE_EXTRA_CA_CERTS",
    "VIRTUAL_ENV",
    "JAVA_HOME",
    "GOROOT",
    "NVM_BIN",
    "NVM_DIR",
    "PYTHONPATH",
    "PYENV_ROOT",
    "RUSTUP_HOME",
)

# /run is masked to hide host D-Bus, Docker/Podman sockets, SSH/GPG agents,
# systemd private sockets, etc. On many Linux systems /etc/resolv.conf is a
# symlink into /run, so when networking is enabled we reopen only the resolved
# target file to keep DNS working without reopening /run wholesale.
AUTO_BIND_RESOLV_CONF_TARGET: bool = True

# Namespace hardening. --new-session is intentionally enabled: bubblewrap's
# own security guidance requires it when TIOCSTI is not blocked with seccomp.
UNSHARE_USER_TRY: bool = True
UNSHARE_PID: bool = True
UNSHARE_IPC: bool = True
UNSHARE_UTS: bool = True
UNSHARE_CGROUP_TRY: bool = False
NEW_TERMINAL_SESSION: bool = True

# Cgroup namespace isolation is optional because older bubblewrap builds may
# not expose --unshare-cgroup-try and it adds little filesystem protection.
# Enable UNSHARE_CGROUP_TRY after confirming your deployment version supports it.
#
# Turning this on can break tools which intentionally create nested user
# namespaces (some browsers/container tools). Leave disabled unless Citra's
# command set no longer needs those workflows.
DISABLE_NESTED_USER_NAMESPACES: bool = False

# Environment variables that point at host IPC/control-plane state. They are
# removed unless explicitly supplied in WorkspaceSandbox.run(environment=...).
DROP_ENVIRONMENT_VARIABLES: frozenset[str] = frozenset(
    {
        "DBUS_SESSION_BUS_ADDRESS",
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "GPG_AGENT_INFO",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    }
)

# Optional secret-prefix filtering. Empty by default for command compatibility.
# If WorkspaceContext.environment() currently inherits host credentials, this
# is a useful place to add prefixes such as "AWS_", "GITHUB_", "OPENAI_", etc.
DROP_ENVIRONMENT_PREFIXES: tuple[str, ...] = ()


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    output: str
    timed_out: bool


@dataclass(frozen=True)
class _FdMount:
    descriptor: int
    target: Path

@dataclass(frozen=True)
class SandboxEnvironmentInfo:
    """
    Describes information that might be relevant to the agent/tool
    """

    extra_readonly_binds: tuple[Path, ...]

class WorkspaceSandbox:
    def __init__(
        self,
        workspace: WorkspaceContext,
        *,
        config: object | None = None,
    ) -> None:
        self.__workspace = workspace
        self.__config = config

    def _setting(self, name: str, default: object) -> object:
        if self.__config is None:
            return default
        return getattr(self.__config, name, default)

    def _string_setting(
        self,
        name: str,
        default: Sequence[str],
    ) -> tuple[str, ...]:
        value = self._setting(name, default)
        if not isinstance(value, (list, tuple, frozenset)) or not all(
            isinstance(item, str)
            for item in value
        ):
            raise TypeError(f"Sandbox setting '{name}' must contain strings.")
        return tuple(value)

    def _bool_setting(self, name: str, default: bool) -> bool:
        value = self._setting(name, default)
        if not isinstance(value, bool):
            raise TypeError(f"Sandbox setting '{name}' must be boolean.")
        return value

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout: int,
        network: bool,
        environment: Mapping[str, str] | None = None,
        input_text: str | None = None,
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

        if timeout <= 0:
            raise ValueError(
                "Sandbox timeout must be greater than zero."
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

        turn_dirs = self._prepare_lifecycle_directories()

        raw_env = workspace.environment(
            environment
        )
        env = self._sandbox_environment(
            raw_env,
            explicit_environment=environment,
            turn_dirs=turn_dirs,
        )

        resolver_bind = (
            self._open_resolver_bind()
            if network
            and self._bool_setting(
                "auto_bind_resolv_conf_target",
                AUTO_BIND_RESOLV_CONF_TARGET,
            )
            else None
        )
        readonly_mounts: tuple[_FdMount, ...] = ()
        writable_mounts: tuple[_FdMount, ...] = ()
        source_mounts: tuple[_FdMount, ...] = ()

        try:
            readonly_mounts = self._open_readonly_mounts(
                command,
                env,
            )
            self._validate_command_mount_coverage(
                command,
                readonly_mounts,
            )
            writable_mounts = self._open_writable_mounts(
                turn_dirs
            )
            source_mounts = self._open_source_mounts()
        except Exception:
            self._close_mounts(
                readonly_mounts,
                writable_mounts,
                source_mounts,
            )
            if resolver_bind is not None:
                os.close(resolver_bind[0])
            raise
        descriptors = [
            *(mount.descriptor for mount in readonly_mounts),
            *(mount.descriptor for mount in writable_mounts),
            *(mount.descriptor for mount in source_mounts),
        ]
        if resolver_bind is not None:
            descriptors.append(resolver_bind[0])

        try:
            bwrap_command = self._build_bwrap_command(
                bwrap=bwrap,
                command=command,
                cwd_path=cwd_path,
                network=network,
                env=env,
                turn_dirs=turn_dirs,
                resolver_bind=resolver_bind,
                readonly_mounts=readonly_mounts,
                writable_mounts=writable_mounts,
                source_mounts=source_mounts,
            )

            proc = subprocess.Popen(
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
                pass_fds=tuple(descriptors),
            )
        finally:
            for descriptor in descriptors:
                os.close(descriptor)

        try:
            if input_text is None:
                output, _ = proc.communicate(timeout=timeout)
            else:
                output, _ = proc.communicate(input=input_text, timeout=timeout)

            return SandboxResult(
                returncode=proc.returncode,
                output=output,
                timed_out=False,
            )

        except subprocess.TimeoutExpired:
            # bwrap itself is the process-group leader created by Popen.
            # --die-with-parent plus bwrap's PID-namespace reaper are retained,
            # while killing the outer process group also terminates helpers
            # which remained in that group.
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
    def environment_info(
        self,
    ) -> SandboxEnvironmentInfo:
        extra_readonly_binds = tuple(
            self._expand_host_path(
                path
            )
            for path in self._string_setting(
                "extra_readonly_binds",
                EXTRA_READONLY_BINDS,
            )
        )

        return SandboxEnvironmentInfo(
            extra_readonly_binds=extra_readonly_binds,
        )
    def popen(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        network: bool = False,
        environment: Mapping[str, str] | None = None,
    ) -> subprocess.Popen[bytes]:
        """Start a lifecycle-owned process under the same mount policy."""
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise RuntimeError(
                "Bubblewrap is required for sandboxed execution but "
                "'bwrap' was not found in PATH."
            )
        if not command:
            raise ValueError("Sandbox command cannot be empty.")
        workspace = self.__workspace
        cwd_path = workspace.workspace if cwd is None else workspace.resolve_path(cwd)
        if not cwd_path.is_dir():
            raise NotADirectoryError(
                f"Sandbox working directory does not exist: {cwd_path}"
            )
        turn_dirs = self._prepare_lifecycle_directories()
        raw_env = workspace.environment(environment)
        env = self._sandbox_environment(
            raw_env,
            explicit_environment=environment,
            turn_dirs=turn_dirs,
        )
        resolver_bind = (
            self._open_resolver_bind()
            if network
            and self._bool_setting(
                "auto_bind_resolv_conf_target",
                AUTO_BIND_RESOLV_CONF_TARGET,
            )
            else None
        )
        readonly_mounts: tuple[_FdMount, ...] = ()
        writable_mounts: tuple[_FdMount, ...] = ()
        source_mounts: tuple[_FdMount, ...] = ()

        try:
            readonly_mounts = self._open_readonly_mounts(command, env)
            self._validate_command_mount_coverage(
                command,
                readonly_mounts,
            )
            writable_mounts = self._open_writable_mounts(turn_dirs)
            source_mounts = self._open_source_mounts()
        except Exception:
            self._close_mounts(
                readonly_mounts,
                writable_mounts,
                source_mounts,
            )
            if resolver_bind is not None:
                os.close(resolver_bind[0])
            raise
        descriptors = [
            *(mount.descriptor for mount in readonly_mounts),
            *(mount.descriptor for mount in writable_mounts),
            *(mount.descriptor for mount in source_mounts),
        ]
        if resolver_bind is not None:
            descriptors.append(resolver_bind[0])

        try:
            bwrap_command = self._build_bwrap_command(
                bwrap=bwrap,
                command=command,
                cwd_path=cwd_path,
                network=network,
                env=env,
                turn_dirs=turn_dirs,
                resolver_bind=resolver_bind,
                readonly_mounts=readonly_mounts,
                writable_mounts=writable_mounts,
                source_mounts=source_mounts,
            )
            return subprocess.Popen(
                bwrap_command,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                pass_fds=tuple(descriptors),
            )
        finally:
            for descriptor in descriptors:
                os.close(descriptor)

    @staticmethod
    def terminate_process(process: subprocess.Popen[object]) -> None:
        """Terminate a sandbox process and all helpers in its process group."""
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _prepare_lifecycle_directories(
        self,
    ) -> dict[str, Path]:
        workspace = self.__workspace
        root = workspace.root

        turn_dirs: dict[str, Path] = {}

        for name in SANDBOX_WRITABLE_DIRS:
            path = root / name
            path.mkdir(
                parents=True,
                exist_ok=True,
            )
            turn_dirs[name] = path

        # XDG_RUNTIME_DIR is expected to be private to the user. These paths
        # are Citra-owned and disposable, so tightening their modes is safe.
        runtime = turn_dirs.get(
            "runtime"
        )
        if runtime is not None:
            runtime.chmod(
                0o700
            )

        home = turn_dirs.get(
            "home"
        )
        if home is not None:
            home.chmod(
                0o700
            )

        # XDG_STATE_HOME must not point at Citra's trusted root/state control
        # plane. Give sandboxed programs a separate writable state directory.
        xdg_state = turn_dirs["data"] / "xdg-state"
        xdg_state.mkdir(
            parents=True,
            exist_ok=True,
        )
        turn_dirs["xdg-state"] = xdg_state

        return turn_dirs

    def _sandbox_environment(
        self,
        env: Mapping[str, str],
        *,
        explicit_environment: Mapping[str, str] | None,
        turn_dirs: Mapping[str, Path],
    ) -> dict[str, str]:
        result = dict(
            env
        )

        explicitly_set = set(
            explicit_environment or {}
        )

        dropped_variables = self._string_setting(
            "drop_environment_variables",
            tuple(DROP_ENVIRONMENT_VARIABLES),
        )
        dropped_prefixes = self._string_setting(
            "drop_environment_prefixes",
            DROP_ENVIRONMENT_PREFIXES,
        )

        for name in dropped_variables:
            if name not in explicitly_set:
                result.pop(
                    name,
                    None,
                )

        if dropped_prefixes:
            for name in list(
                result
            ):
                if name in explicitly_set:
                    continue

                if any(
                    name.startswith(prefix)
                    for prefix in dropped_prefixes
                ):
                    result.pop(
                        name,
                        None,
                    )

        # Force mutable per-command state into Citra-owned directories rather
        # than the masked host home/tmp trees. Git, npm, Python tempfile, and
        # most Unix developer tools respect these conventional locations.
        result["HOME"] = str(
            turn_dirs["home"]
        )
        result["XDG_CONFIG_HOME"] = str(
            turn_dirs["config"]
        )
        result["XDG_CACHE_HOME"] = str(
            turn_dirs["cache"]
        )
        result["XDG_DATA_HOME"] = str(
            turn_dirs["data"]
        )
        result["XDG_STATE_HOME"] = str(
            turn_dirs["xdg-state"]
        )
        result["XDG_RUNTIME_DIR"] = str(
            turn_dirs["runtime"]
        )
        result["TMPDIR"] = str(
            turn_dirs["tmp"]
        )
        result["TMP"] = str(
            turn_dirs["tmp"]
        )
        result["TEMP"] = str(
            turn_dirs["tmp"]
        )

        return result

    def _build_bwrap_command(
        self,
        *,
        bwrap: str,
        command: Sequence[str],
        cwd_path: Path,
        network: bool,
        env: Mapping[str, str],
        turn_dirs: Mapping[str, Path],
        resolver_bind: tuple[int, Path] | None = None,
        readonly_mounts: Sequence[_FdMount] | None = None,
        writable_mounts: Sequence[_FdMount] | None = None,
        source_mounts: Sequence[_FdMount] | None = None,
    ) -> list[str]:
        workspace = self.__workspace

        args: list[str] = [
            bwrap,
            "--die-with-parent",
        ]
        created_mount_directories: set[Path] = set()

        if NEW_TERMINAL_SESSION:
            args.append(
                "--new-session"
            )

        if UNSHARE_USER_TRY:
            args.append(
                "--unshare-user-try"
            )

        if UNSHARE_PID:
            args.append(
                "--unshare-pid"
            )

        if UNSHARE_IPC:
            args.append(
                "--unshare-ipc"
            )

        if UNSHARE_UTS:
            args.append(
                "--unshare-uts"
            )

        if UNSHARE_CGROUP_TRY:
            args.append(
                "--unshare-cgroup-try"
            )

        if DISABLE_NESTED_USER_NAMESPACES:
            args.append(
                "--disable-userns"
            )

        if not network:
            args.append(
                "--unshare-net"
            )

        # Compatibility baseline first.
        for path in BASE_READONLY_BINDS:
            args.extend(
                (
                    "--ro-bind",
                    path,
                    path,
                )
            )

        # Then hide host state. Later explicit mounts can reopen only the
        # narrow paths Citra actually needs. A mount destination beneath the
        # read-only root must already exist: bwrap cannot create a missing
        # destination such as /media after ``--ro-bind / /``. Missing paths
        # contain no host state to hide, so omit them instead of making the
        # entire sandbox fail during setup.
        for path in MASKED_HOST_DIRS:
            if not Path(path).is_dir():
                continue

            args.extend(
                (
                    "--tmpfs",
                    path,
                )
            )

        for path in MASKED_HOST_FILES:
            if not Path(path).is_file():
                continue

            args.extend(
                (
                    "--ro-bind",
                    "/dev/null",
                    path,
                )
            )

        # Always replace host process/device views with sandbox-owned ones.
        args.extend(
            (
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/dev/shm",
            )
        )

        # /tmp and /var/tmp remain writable but contain no host files. Most
        # programs will use TMPDIR (the persistent Citra turn tmp directory),
        # while hard-coded /tmp users still work in an isolated tmpfs.
        # MASKED_HOST_DIRS already creates these tmpfs mounts.

        # Explicitly hide the entire Citra turn root as seen through the
        # read-only host baseline. We then reopen only approved data-plane
        # directories. This keeps root/state and workspace.git invisible.
        self._append_masked_parent_directories(
            args,
            target=workspace.root,
            created=created_mount_directories,
        )
        args.extend(
            (
                "--tmpfs",
                str(workspace.root),
            )
        )

        # DNS compatibility when /run is masked.
        if resolver_bind is not None:
            resolver_fd, resolver_target = resolver_bind
            self._append_resolver_bind(
                args,
                resolver_fd=resolver_fd,
                target=resolver_target,
            )

        # Reopen selected host tool/runtime paths read-only.
        # Fixed Citra worker modules and the active Python environment may live
        # beneath a masked home directory. Production calls pass descriptors
        # opened before namespace setup, so masking /home cannot hide the bind
        # sources. The path fallback keeps direct command-construction tests
        # useful without requiring descriptor management.
        if readonly_mounts is None:
            readonly_candidates = [
                *self._compatibility_readonly_binds(env),
                *(
                    self._expand_host_path(path)
                    for path in self._string_setting(
                        "extra_readonly_binds",
                        EXTRA_READONLY_BINDS,
                    )
                ),
                *self._citra_runtime_readonly_binds(),
                *self._command_runtime_readonly_binds(command),
            ]
            readonly_paths = self._minimal_existing_bind_paths(
                readonly_candidates
            )
        else:
            readonly_paths = ()

        for path in readonly_paths:
            self._append_masked_parent_directories(
                args,
                target=path,
                created=created_mount_directories,
            )
            args.extend(
                (
                    "--ro-bind",
                    str(path),
                    str(path),
                )
            )

        for mount in readonly_mounts or ():
            self._append_masked_parent_directories(
                args,
                target=mount.target,
                created=created_mount_directories,
            )
            args.extend(
                (
                    "--ro-bind-fd",
                    str(mount.descriptor),
                    str(mount.target),
                )
            )

        # The whole Citra installation is available read-only. Hide the
        # operator's API/provider configuration again after that mount.
        for path in self._citra_private_config_files():
            if not path.is_file():
                continue
            args.extend(
                (
                    "--ro-bind",
                    "/dev/null",
                    str(path),
                )
            )

        # Reopen only Citra-owned turn directories writable.
        if writable_mounts is None:
            for name in SANDBOX_WRITABLE_DIRS:
                path = turn_dirs[name]
                args.extend(
                    (
                        "--bind",
                        str(path),
                        str(path),
                    )
                )
        else:
            for mount in writable_mounts:
                args.extend(
                    (
                        "--bind-fd",
                        str(mount.descriptor),
                        str(mount.target),
                    )
                )

        # XDG_STATE_HOME is a child of data, so the data bind above already
        # exposes it. No additional mount is necessary.

        if writable_mounts is None:
            for path in EXTRA_WRITABLE_BINDS:
                expanded = self._expand_host_path(
                    path
                )

                if not expanded.exists():
                    continue

                self._append_masked_parent_directories(
                    args,
                    target=expanded,
                    created=created_mount_directories,
                )
                args.extend(
                    (
                        "--bind",
                        str(expanded),
                        str(expanded),
                    )
                )

        # Optional hardware/device access is opt-in. /dev itself is synthetic.
        for path in EXTRA_DEVICE_BINDS:
            expanded = self._expand_host_path(
                path
            )

            if not expanded.exists():
                continue

            self._append_masked_parent_directories(
                args,
                target=expanded,
                created=created_mount_directories,
            )
            args.extend(
                (
                    "--dev-bind",
                    str(expanded),
                    str(expanded),
                )
            )

        # The original project is authoritative and immutable. Put these last
        # so no earlier compatibility/writable override can make it writable.
        source_alias = workspace.workspace / "@source"
        if source_mounts is None:
            source = workspace.source_workspace
            self._append_masked_parent_directories(
                args,
                target=source,
                created=created_mount_directories,
            )
            args.extend(
                (
                    "--ro-bind",
                    str(source),
                    str(source),
                    "--ro-bind",
                    str(source),
                    str(source_alias),
                )
            )
        else:
            for mount in source_mounts:
                if not self._is_inside(
                    mount.target,
                    workspace.workspace,
                ):
                    self._append_masked_parent_directories(
                        args,
                        target=mount.target,
                        created=created_mount_directories,
                    )
                args.extend(
                    (
                        "--ro-bind-fd",
                        str(mount.descriptor),
                        str(mount.target),
                    )
                )

        args.extend(
            (
                "--chdir",
                str(cwd_path),
                "--",
            )
        )

        args.extend(
            command
        )

        return args

    @staticmethod
    def _append_masked_parent_directories(
        args: list[str],
        *,
        target: Path,
        created: set[Path],
    ) -> None:
        """Recreate a mount target's parents beneath an active mask.

        A tmpfs mounted on ``/home`` or ``/mnt`` intentionally removes the
        host directory tree from the sandbox. Bubblewrap can create the final
        bind destination, but it cannot traverse missing intermediate parents.
        Recreate only the empty directory skeleton required by an explicitly
        approved later bind.
        """
        absolute = target.absolute()
        applicable_masks = [
            Path(raw_mask).absolute()
            for raw_mask in MASKED_HOST_DIRS
            if Path(raw_mask).is_dir()
            and WorkspaceSandbox._is_inside(
                absolute,
                Path(raw_mask).absolute(),
            )
        ]

        if not applicable_masks:
            return

        mask = max(
            applicable_masks,
            key=lambda path: len(path.parts),
        )
        parents: list[Path] = []
        parent = absolute.parent

        while parent != mask:
            if not WorkspaceSandbox._is_inside(parent, mask):
                return
            parents.append(parent)
            parent = parent.parent

        for directory in reversed(parents):
            if directory in created:
                continue
            args.extend(
                (
                    "--dir",
                    str(directory),
                )
            )
            created.add(directory)

    @staticmethod
    def _minimal_existing_bind_paths(
        candidates: Sequence[Path],
    ) -> tuple[Path, ...]:
        """Return existing bind roots without redundant nested mounts."""
        unique: list[Path] = []

        for candidate in candidates:
            path = candidate.absolute()
            if not path.exists() or path in unique:
                continue
            unique.append(path)

        return tuple(
            path
            for path in unique
            if not any(
                other != path
                and other.is_dir()
                and WorkspaceSandbox._is_inside(path, other)
                for other in unique
            )
        )

    def _open_readonly_mounts(
        self,
        command: Sequence[str],
        env: Mapping[str, str],
    ) -> tuple[_FdMount, ...]:
        candidates = [
            *self._compatibility_readonly_binds(env),
            *(
                self._expand_host_path(path)
                for path in self._string_setting(
                    "extra_readonly_binds",
                    EXTRA_READONLY_BINDS,
                )
            ),
            *self._citra_runtime_readonly_binds(),
            *self._command_runtime_readonly_binds(command),
        ]
        paths = self._minimal_existing_bind_paths(candidates)
        return self._open_mounts(
            (path, path)
            for path in paths
        )

    def _open_writable_mounts(
        self,
        turn_dirs: Mapping[str, Path],
    ) -> tuple[_FdMount, ...]:
        paths = [
            *(turn_dirs[name] for name in SANDBOX_WRITABLE_DIRS),
            *(
                self._expand_host_path(path)
                for path in EXTRA_WRITABLE_BINDS
            ),
        ]
        return self._open_mounts(
            (path, path)
            for path in paths
            if path.exists()
        )

    def _open_source_mounts(self) -> tuple[_FdMount, ...]:
        source = self.__workspace.source_workspace
        return self._open_mounts(
            (
                (source, source),
                (source, self.__workspace.workspace / "@source"),
            )
        )

    @staticmethod
    def _open_mounts(
        mounts: Iterable[tuple[Path, Path]],
    ) -> tuple[_FdMount, ...]:
        opened: list[_FdMount] = []

        try:
            for source, target in mounts:
                descriptor = os.open(
                    source,
                    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
                )
                opened.append(
                    _FdMount(
                        descriptor=descriptor,
                        target=target,
                    )
                )
        except Exception:
            for mount in opened:
                os.close(mount.descriptor)
            raise

        return tuple(opened)

    @staticmethod
    def _close_mounts(
        *groups: Sequence[_FdMount],
    ) -> None:
        for group in groups:
            for mount in group:
                os.close(mount.descriptor)

    @staticmethod
    def _citra_runtime_readonly_binds() -> tuple[Path, ...]:
        install_root = WorkspaceSandbox._citra_install_root()
        candidates = [
            install_root,
            Path(sys.prefix).resolve(),
        ]
        return WorkspaceSandbox._minimal_existing_bind_paths(candidates)

    @staticmethod
    def _citra_install_root() -> Path:
        raw = os.environ.get("CITRA_INSTALL_ROOT")

        if not raw:
            raise RuntimeError(
                "CITRA_INSTALL_ROOT is not defined. "
                "Citra must be launched through start.sh."
            )

        root = Path(raw).expanduser().resolve()

        if not root.is_dir():
            raise NotADirectoryError(
                f"Citra installation directory does not exist: {root}"
            )

        return root

    @staticmethod
    def _citra_private_config_files() -> tuple[Path, ...]:
        configured = os.environ.get("CITRA_CONFIG_PATH")
        if configured:
            return (Path(configured).expanduser().resolve(),)

        state_root = os.environ.get("CITRA_ROOT")
        if state_root:
            return (
                Path(state_root).expanduser().resolve()
                / CITRA_PRIVATE_CONFIG_FILE,
            )

        raise RuntimeError(
            "CITRA_CONFIG_PATH and CITRA_ROOT are not defined. "
            "Citra must be launched through start.sh."
        )

    @staticmethod
    def _command_runtime_readonly_binds(
        command: Sequence[str],
    ) -> tuple[Path, ...]:
        """Expose a masked executable and its conventional runtime root."""
        executable = Path(command[0])
        if not executable.is_absolute() or not executable.exists():
            return ()
        resolved = executable.resolve()
        candidates = [executable.parent, resolved.parent]
        if executable.parent.name == "bin":
            candidates.append(executable.parent.parent)
        if resolved.parent.name == "bin":
            candidates.append(resolved.parent.parent)
        return tuple(dict.fromkeys(path.resolve() for path in candidates))

    @staticmethod
    def _validate_command_mount_coverage(
        command: Sequence[str],
        readonly_mounts: Sequence[_FdMount],
    ) -> None:
        """Fail before bwrap if masking would hide the worker executable.

        The command path and its resolved symlink target can live in different
        trees.  A Citra-local ``.venv/bin/python`` commonly resolves to a
        Python installation elsewhere under ``/home``.  Both paths must be
        covered when their host tree is masked.
        """
        executable = Path(command[0])

        if not executable.is_absolute():
            return

        if not executable.exists():
            raise FileNotFoundError(
                f"Sandbox command does not exist: {executable}"
            )

        paths = tuple(
            dict.fromkeys(
                (
                    executable.absolute(),
                    executable.resolve(),
                )
            )
        )

        for path in paths:
            if not WorkspaceSandbox._is_under_masked_host_dir(path):
                continue

            if any(
                WorkspaceSandbox._is_inside(path, mount.target)
                for mount in readonly_mounts
            ):
                continue

            raise RuntimeError(
                "Sandbox setup would hide the command executable: "
                f"{path}. Add its runtime root to the read-only binds."
            )

    def _compatibility_readonly_binds(
        self,
        env: Mapping[str, str],
    ) -> list[Path]:
        workspace = self.__workspace
        candidates: list[Path] = []

        if AUTO_BIND_MASKED_PATH_ENTRIES:
            path_value = env.get(
                "PATH",
                "",
            )

            for raw_path in path_value.split(
                os.pathsep
            ):
                if not raw_path:
                    continue

                path = Path(
                    raw_path
                ).expanduser()

                if not path.is_absolute():
                    continue

                if self._is_inside(
                    path,
                    workspace.root,
                ):
                    continue

                if self._is_under_masked_host_dir(
                    path
                ):
                    candidates.append(
                        path
                    )

        for name in AUTO_BIND_ENV_PATHS:
            raw_value = env.get(
                name
            )

            if not raw_value:
                continue

            # SSL_CERT_DIR can contain a colon-separated directory list on
            # OpenSSL systems. NODE_PATH is intentionally not included in the
            # default variable set, but handling os.pathsep here makes custom
            # additions safe.
            for value in raw_value.split(
                os.pathsep
            ):
                if not value:
                    continue

                path = Path(
                    value
                ).expanduser()

                if not path.is_absolute():
                    continue

                if self._is_inside(
                    path,
                    workspace.root,
                ):
                    continue

                if self._is_under_masked_host_dir(
                    path
                ):
                    candidates.append(
                        path
                    )

        # Preserve order while deduplicating. Do not resolve symlinks here:
        # consumers may rely on the original path spelling, and bwrap sources
        # are looked up against the host root.
        seen: set[str] = set()
        result: list[Path] = []

        for path in candidates:
            key = os.path.normpath(
                str(path)
            )

            if key in seen:
                continue

            seen.add(
                key
            )
            result.append(
                Path(key)
            )

        return result

    @staticmethod
    def _expand_host_path(
        value: str,
    ) -> Path:
        return Path(
            os.path.expandvars(
                os.path.expanduser(
                    value
                )
            )
        )

    @staticmethod
    def _is_inside(
        path: Path,
        parent: Path,
    ) -> bool:
        try:
            path.absolute().relative_to(
                parent.absolute()
            )
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_under_masked_host_dir(
        path: Path,
    ) -> bool:
        absolute = path.absolute()

        for raw_mask in MASKED_HOST_DIRS:
            mask = Path(
                raw_mask
            ).absolute()

            try:
                absolute.relative_to(
                    mask
                )
                return True
            except ValueError:
                continue

        return False

    @staticmethod
    def _resolv_conf_runtime_target() -> Path | None:
        resolv_conf = Path(
            "/etc/resolv.conf"
        )

        try:
            target = resolv_conf.resolve(
                strict=True
            )
        except (
            FileNotFoundError,
            OSError,
            RuntimeError,
        ):
            return None

        # /etc/resolv.conf itself is already visible through the read-only
        # host root. We only need an extra bind when masking /run would hide
        # the symlink target.
        try:
            target.relative_to(
                "/run"
            )
        except ValueError:
            return None

        if not target.is_file():
            return None

        return target

    def _open_resolver_bind(
        self,
    ) -> tuple[int, Path] | None:
        """Open resolver data before ``/run`` is hidden by Bubblewrap.

        On systemd-based hosts ``/etc/resolv.conf`` commonly points into
        ``/run``. Reusing that path as the bind source after mounting an empty
        ``/run`` fails because the source has already disappeared. Passing an
        already-open descriptor to ``--ro-bind-fd`` avoids that ordering bug
        and does not expose any host runtime directory.
        """
        target = self._resolv_conf_runtime_target()

        if target is None:
            return None

        try:
            descriptor = os.open(
                "/etc/resolv.conf",
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError:
            return None

        return descriptor, target

    @staticmethod
    def _append_resolver_bind(
        args: list[str],
        *,
        resolver_fd: int,
        target: Path,
    ) -> None:
        run_root = Path("/run")
        parents: list[Path] = []

        parent = target.parent

        while parent != run_root:
            try:
                parent.relative_to(run_root)
            except ValueError:
                return

            parents.append(parent)
            parent = parent.parent

        for directory in reversed(parents):
            args.extend(
                (
                    "--dir",
                    str(directory),
                )
            )

        args.extend(
            (
                "--ro-bind-fd",
                str(resolver_fd),
                str(target),
            )
        )
