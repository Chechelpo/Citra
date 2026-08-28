from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from typing import Any, TYPE_CHECKING, Iterable, Mapping, Sequence

from citra.sandbox.runtime_discovery import *

if TYPE_CHECKING:
    from ..context.turn_workspace import WorkspaceContext


# ---------------------------------------------------------------------------
# Sandbox policy
# ---------------------------------------------------------------------------
#
# The default policy starts from Bubblewrap's empty root and exposes only the
# explicit compatibility assets selected by runtime provisioning, plus the
# process-lifetime Agent Runtime data-plane roots.  A broad compatibility bind
# can still be configured by an operator, but is never the default.
#
# Bubblewrap applies filesystem operations in command-line order, so later
# mounts intentionally override earlier ones.

class SandboxMode(IntEnum):
    """
    Level of sandboxing required by a mode.

    Each level includes the guarantees of the previous level.
    """

    FULL_ACCESS = 0
    ONLY_SOURCE = 1
    PARTIAL_SANDBOX = 2
    FULL_SANDBOX = 3

    @property
    def uses_direct_source(self) -> bool:
        """Whether the authoritative source is the active project root."""
        return self <= SandboxMode.ONLY_SOURCE

# Optional operator compatibility baseline.  Normal runtime assets arrive via
# WorkspaceContext.runtime_readonly_binds instead.
BASE_READONLY_BINDS: tuple[str, ...] = ()

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
    "env",
    "cache",
    "home",
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
CITRA_CONFIG_DIRECTORY = "config"
CITRA_LEGACY_PRIVATE_CONFIG_FILE = "config.toml"

# If PATH contains executable directories beneath a masked host tree (for
# example ~/.local/bin or ~/.cargo/bin), expose exactly those PATH directories
# read-only. This preserves many user-installed command launchers without
# reopening the rest of $HOME. Some runtimes need sibling directories too
# (notably NVM/npm or rustup); add their runtime root to EXTRA_READONLY_BINDS.
AUTO_BIND_MASKED_PATH_ENTRIES: bool = False

# Some environments point TLS libraries directly at a CA file/directory. If
# that target lives under a masked directory, reopen only that path read-only.
AUTO_BIND_ENV_PATHS: tuple[str, ...] = ()

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

ISOLATION_ENVIRONMENT_VARIABLES: tuple[str, ...] = (
    "HOME",
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_RUNTIME_DIR",
    "VIRTUAL_ENV",
    "CITRA_WORKSPACE",
    "CITRA_SOURCE",
    "CITRA_LIBRARY",
    "CITRA_AGENT_ROOT",
    "CITRA_RUNTIME",
    "CITRA_ENV",
    "CITRA_TMP",
    "CITRA_CACHE",
)


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

    mode: SandboxMode
    extra_readonly_binds: tuple[Path, ...]

class WorkspaceSandbox:
    def __init__(
        self,
        workspace: WorkspaceContext,
        *,
        config: object | None = None,
        mode_config: object | None = None,
        runtime_discovery: Sequence[type[RuntimeDiscovery]] | None = None,
    ) -> None:
        self.__workspace = workspace
        self.__config = config
        self.__mode_config = mode_config
        self.__runtime_discovery = tuple(
            RUNTIME_DISCOVERY
            if runtime_discovery is None
            else runtime_discovery
        )
        self.__runtime_discovery_result = self._run_runtime_discovery()

    def _run_runtime_discovery(self) -> RuntimeDiscoveryResult:
        """Run the configured runtime-discovery classes exactly once."""
        readonly_binds: list[Path] = []

        for discovery in self.__runtime_discovery:
            if not isinstance(discovery, type) or not issubclass(
                discovery,
                RuntimeDiscovery,
            ):
                raise TypeError(
                    "Runtime discovery entries must be RuntimeDiscovery classes."
                )

            result = discovery.discover(self.__workspace)
            if not isinstance(result, RuntimeDiscoveryResult):
                raise TypeError(
                    f"{discovery.__name__}.discover() must return "
                    "RuntimeDiscoveryResult."
                )

            for path in result.readonly_binds:
                if not isinstance(path, Path):
                    raise TypeError(
                        f"{discovery.__name__}.discover() returned a non-Path "
                        "readonly bind."
                    )
                readonly_binds.append(path)

        return RuntimeDiscoveryResult(
            readonly_binds=self._minimal_existing_bind_paths(readonly_binds),
        )

    def _effective_extra_readonly_binds(self) -> tuple[Path, ...]:
        """Merge mode/operator read-only binds with runtime discovery."""
        configured = tuple(
            self._expand_host_path(path)
            for path in self._string_setting(
                "extra_readonly_binds",
                EXTRA_READONLY_BINDS,
            )
        )
        return tuple(
            dict.fromkeys(
                (
                    *configured,
                    *self.__runtime_discovery_result.readonly_binds,
                )
            )
        )

    @property
    def runtime_discovery_result(self) -> RuntimeDiscoveryResult:
        """Return the immutable runtime requirements captured at creation."""
        return self.__runtime_discovery_result

    @property
    def mode(self) -> SandboxMode:
        configured = getattr(
            self.__mode_config,
            "mode",
            None,
        )
        if configured is None:
            return (
                SandboxMode.ONLY_SOURCE
                if bool(getattr(self.__workspace, "direct_source", False))
                else SandboxMode.FULL_SANDBOX
            )
        if not isinstance(configured, SandboxMode):
            raise TypeError("Mode sandbox configuration has an invalid mode")
        return configured

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
        configured = tuple(value)
        mode_name = {
            "extra_readonly_binds": "additional_ro_binds",
            "extra_writable_binds": "additional_w_binds",
        }.get(name)
        if mode_name is None or self.__mode_config is None:
            return configured

        mode_paths = getattr(self.__mode_config, mode_name, ())
        if not isinstance(mode_paths, tuple) or not all(
            isinstance(path, Path)
            for path in mode_paths
        ):
            raise TypeError(
                f"Mode sandbox setting '{mode_name}' must contain Path values."
            )
        return tuple(
            dict.fromkeys(
                (
                    *(str(path) for path in mode_paths),
                    *configured,
                )
            )
        )

    def _bool_setting(self, name: str, default: bool) -> bool:
        value = self._setting(name, default)
        if not isinstance(value, bool):
            raise TypeError(f"Sandbox setting '{name}' must be boolean.")
        return value

    def allows_network(self, requested: bool) -> bool:
        """Apply mode and operator global network restrictions."""
        if not isinstance(requested, bool):
            raise TypeError("requested network access must be boolean")
        mode_disallows = bool(
            getattr(
                self.__mode_config,
                "global_network_disallow",
                False,
            )
        )
        operator_disallows = self._bool_setting(
            "global_network_disallow",
            False,
        )
        return requested and not (mode_disallows or operator_disallows)

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
        self.__workspace.ensure_active()
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

        network = self.allows_network(network)

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
            if not workspace.processes.register(proc):
                raise RuntimeError("Agent Runtime began closing during process start.")
        finally:
            for descriptor in descriptors:
                os.close(descriptor)

        try:
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
                self.terminate_process(proc, force=True)
                output, _ = proc.communicate()
                return SandboxResult(
                    returncode=proc.returncode,
                    output=output,
                    timed_out=True,
                )
        finally:
            workspace.processes.unregister(proc)
    def environment_info(
        self,
    ) -> SandboxEnvironmentInfo:
        return SandboxEnvironmentInfo(
            mode=self.mode,
            extra_readonly_binds=self._effective_extra_readonly_binds(),
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
        self.__workspace.ensure_active()
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise RuntimeError(
                "Bubblewrap is required for sandboxed execution but "
                "'bwrap' was not found in PATH."
            )
        if not command:
            raise ValueError("Sandbox command cannot be empty.")
        network = self.allows_network(network)
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
            process = subprocess.Popen(
                bwrap_command,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                pass_fds=tuple(descriptors),
            )
            if not workspace.processes.register(process):
                raise RuntimeError("Agent Runtime began closing during process start.")
            return process
        finally:
            for descriptor in descriptors:
                os.close(descriptor)

    def terminate_process(
        self,
        process: subprocess.Popen[Any],
        *,
        force: bool = False,
        grace_seconds: float = 1.0,
    ) -> None:
        """Terminate a sandbox process group with a bounded graceful phase."""
        try:
            if process.poll() is not None:
                return
            try:
                os.killpg(
                    process.pid,
                    signal.SIGKILL if force else signal.SIGTERM,
                )
            except ProcessLookupError:
                return
            if force:
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    # Keep the process registered so aggregate runtime cleanup
                    # gets one final bounded chance to reap it.
                    return
                return
            try:
                process.wait(timeout=max(0.0, grace_seconds))
                return
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                # The OS will reap the daemon-owned process asynchronously. Do
                # not let shutdown wait without a bound.
                return
        finally:
            if process.poll() is not None:
                self.__workspace.processes.unregister(process)

    def _prepare_lifecycle_directories(
        self,
    ) -> dict[str, Path]:
        workspace = self.__workspace
        turn_dirs = {
            "workspace": workspace.workspace,
            "env": workspace.env,
            "cache": workspace.cache,
            "home": workspace.home,
            "tmp": workspace.tmp,
            "config": workspace.config,
            "data": workspace.data,
            "xdg-state": workspace.home / ".local" / "state",
            "runtime-state": workspace.runtime_state,
        }
        for path in turn_dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        workspace.home.chmod(0o700)
        workspace.runtime_state.chmod(0o700)
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

        # WorkspaceContext is the sole canonical environment builder. Reassert
        # its exact forced values after filtering, including cache/xdg rather
        # than constructing a subtly different sandbox-only mapping.
        del turn_dirs
        for name in ISOLATION_ENVIRONMENT_VARIABLES:
            if name in env:
                result[name] = env[name]

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

        for setting, default, option in (
            ("new_terminal_session", NEW_TERMINAL_SESSION, "--new-session"),
            ("unshare_user_try", UNSHARE_USER_TRY, "--unshare-user-try"),
            ("unshare_pid", UNSHARE_PID, "--unshare-pid"),
            ("unshare_ipc", UNSHARE_IPC, "--unshare-ipc"),
            ("unshare_uts", UNSHARE_UTS, "--unshare-uts"),
            ("unshare_cgroup_try", UNSHARE_CGROUP_TRY, "--unshare-cgroup-try"),
            (
                "disable_nested_user_namespaces",
                DISABLE_NESTED_USER_NAMESPACES,
                "--disable-userns",
            ),
        ):
            if self._bool_setting(setting, default):
                args.append(option)
        if not network:
            args.append("--unshare-net")

        direct_readonly: tuple[tuple[Path, Path], ...] = ()
        if readonly_mounts is None:
            candidates: list[tuple[Path, Path]] = []
            same_path = [
                *(
                    self._expand_host_path(path)
                    for path in self._string_setting(
                        "base_readonly_binds",
                        BASE_READONLY_BINDS,
                    )
                ),
                *self._compatibility_readonly_binds(env),
                *self._effective_extra_readonly_binds(),
            ]
            if self._bool_setting("auto_bind_citra_runtime", False):
                same_path.extend(self._citra_runtime_readonly_binds())
            candidates.extend((path, path) for path in same_path if path.exists())
            if workspace.runtime.exists():
                candidates.append((workspace.runtime, workspace.runtime))
            candidates.extend(workspace.runtime_readonly_binds)
            direct_readonly = self._minimal_mount_pairs(candidates)

        root_direct = tuple(
            pair for pair in direct_readonly if pair[1] == Path("/")
        )
        other_direct = tuple(
            pair for pair in direct_readonly if pair[1] != Path("/")
        )
        root_fd = tuple(
            mount for mount in (readonly_mounts or ())
            if mount.target == Path("/")
        )
        other_fd = tuple(
            mount for mount in (readonly_mounts or ())
            if mount.target != Path("/")
        )

        # A configured broad compatibility root is applied first so masking
        # can narrow it.  The default has neither form.
        for source, target in root_direct:
            args.extend(("--ro-bind", str(source), str(target)))
        for mount in root_fd:
            args.extend(
                ("--ro-bind-fd", str(mount.descriptor), str(mount.target))
            )
        broad_root = bool(root_direct or root_fd)
        if broad_root:
            for path in self._string_setting("masked_host_dirs", MASKED_HOST_DIRS):
                if Path(path).is_dir():
                    args.extend(("--tmpfs", path))
            for path in self._string_setting("masked_host_files", MASKED_HOST_FILES):
                if Path(path).is_file():
                    args.extend(("--ro-bind", "/dev/null", path))

        # Synthetic process/device views and isolated conventional temp roots.
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
        created_mount_directories.update(
            {Path("/tmp"), Path("/var"), Path("/var/tmp"), Path("/run")}
        )

        # Hide the controller-owned process root even when an operator opts
        # into a broad host compatibility bind. Approved child roots are
        # reopened explicitly below; metadata is never reopened.
        self._append_masked_parent_directories(
            args,
            target=workspace.root,
            created=created_mount_directories,
            broad_root=broad_root,
        )
        args.extend(("--tmpfs", str(workspace.root)))
        created_mount_directories.add(workspace.root)

        direct_writable: tuple[Path, ...] = ()
        if writable_mounts is None:
            direct_writable = tuple(
                dict.fromkeys(
                    [
                        *(turn_dirs[name] for name in SANDBOX_WRITABLE_DIRS),
                        *(
                            self._expand_host_path(path)
                            for path in self._string_setting(
                                "extra_writable_binds",
                                EXTRA_WRITABLE_BINDS,
                            )
                            if self._expand_host_path(path).exists()
                        ),
                    ]
                )
            )

        device_paths = tuple(
            path
            for path in (
                self._expand_host_path(value)
                for value in self._string_setting(
                    "extra_device_binds",
                    EXTRA_DEVICE_BINDS,
                )
            )
            if path.exists()
        )

        direct_source = bool(getattr(workspace, "direct_source", False))
        source_alias = (
            workspace.source_workspace
            if direct_source
            else workspace.workspace / "@source"
        )
        direct_sources: tuple[tuple[Path, Path], ...] = ()
        if source_mounts is None and not direct_source:
            direct_sources = (
                (workspace.source_workspace, workspace.source_workspace),
                (workspace.source_workspace, source_alias),
            )

        protected_mount_targets = [
            *(target for _, target in other_direct),
            *(mount.target for mount in other_fd),
            *direct_writable,
            *(mount.target for mount in (writable_mounts or ())),
            *device_paths,
        ]
        for target in protected_mount_targets:
            if self._overlaps_metadata(target):
                raise RuntimeError(
                    "Sandbox mount would expose controller metadata: "
                    f"{target}"
                )

        # Create only empty destination-parent skeletons.  The host root and
        # controller metadata never become visible merely because an absolute
        # target shares their spelling.
        targets = [
            *(target for _, target in other_direct),
            *(mount.target for mount in other_fd),
            *direct_writable,
            *(mount.target for mount in (writable_mounts or ())),
            *device_paths,
            *(
                target
                for _, target in direct_sources
                if not self._is_inside(target, workspace.workspace)
            ),
            *(
                mount.target
                for mount in (source_mounts or ())
                if not self._is_inside(mount.target, workspace.workspace)
            ),
        ]
        if resolver_bind is not None:
            targets.append(resolver_bind[1])
        for target in targets:
            self._append_masked_parent_directories(
                args,
                target=target,
                created=created_mount_directories,
                broad_root=broad_root,
            )

        if resolver_bind is not None:
            resolver_fd, resolver_target = resolver_bind
            args.extend(
                ("--ro-bind-fd", str(resolver_fd), str(resolver_target))
            )

        for source, target in other_direct:
            args.extend(("--ro-bind", str(source), str(target)))
        for mount in other_fd:
            args.extend(
                ("--ro-bind-fd", str(mount.descriptor), str(mount.target))
            )

        for path in direct_writable:
            args.extend(("--bind", str(path), str(path)))
        for mount in writable_mounts or ():
            args.extend(("--bind-fd", str(mount.descriptor), str(mount.target)))

        for path in device_paths:
            args.extend(("--dev-bind", str(path), str(path)))

        # In isolated-copy mode, source is authoritative and immutable. These
        # mounts are last so no compatibility mount can make either view
        # writable. Direct-source mode intentionally omits them; its workspace
        # writable bind is the authoritative source itself.
        for source, target in direct_sources:
            args.extend(("--ro-bind", str(source), str(target)))
        for mount in source_mounts or ():
            args.extend(
                ("--ro-bind-fd", str(mount.descriptor), str(mount.target))
            )

        # Secret/library masks are the final filesystem operation so neither an
        # installation-root bind nor an authoritative source bind can reopen
        # controller-owned state.
        private_directories = self._citra_private_directories()
        for path in private_directories:
            for target in self._private_path_targets(path, source_alias):
                self._append_masked_parent_directories(
                    args,
                    target=target,
                    created=created_mount_directories,
                    broad_root=broad_root,
                )
                args.extend(("--tmpfs", str(target)))

        private_files = (
            *self._citra_private_config_files(),
            *(
                self._expand_host_path(path).resolve()
                for path in self._string_setting("private_files", ())
            ),
        )
        for path in private_files:
            if not path.is_file():
                continue
            if any(self._is_inside(path, directory) for directory in private_directories):
                continue
            for target in self._private_path_targets(path, source_alias):
                self._append_masked_parent_directories(
                    args,
                    target=target,
                    created=created_mount_directories,
                    broad_root=broad_root,
                )
                args.extend(("--ro-bind", "/dev/null", str(target)))

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

    def _append_masked_parent_directories(
        self,
        args: list[str],
        *,
        target: Path,
        created: set[Path],
        broad_root: bool = False,
    ) -> None:
        """Create empty parents needed by one explicit mount destination."""
        absolute = target.absolute()
        boundary = Path("/")
        if broad_root:
            applicable_masks = [
                Path(raw_mask).absolute()
                for raw_mask in self._string_setting(
                    "masked_host_dirs",
                    MASKED_HOST_DIRS,
                )
                if Path(raw_mask).is_dir()
                and self._is_inside(absolute, Path(raw_mask).absolute())
            ]
            if not applicable_masks:
                return
            boundary = max(applicable_masks, key=lambda path: len(path.parts))

        parents: list[Path] = []
        parent = absolute.parent

        while parent != boundary:
            if not self._is_inside(parent, boundary):
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

    @staticmethod
    def _minimal_mount_pairs(
        candidates: Iterable[tuple[Path, Path]],
    ) -> tuple[tuple[Path, Path], ...]:
        """Deduplicate mounts when an earlier directory pair covers a child."""
        result: list[tuple[Path, Path]] = []
        seen: set[tuple[str, str]] = set()
        for raw_source, raw_target in candidates:
            source = raw_source.absolute()
            target = raw_target.absolute()
            key = (os.path.normpath(str(source)), os.path.normpath(str(target)))
            if key in seen:
                continue
            seen.add(key)

            covered = False
            for parent_source, parent_target in result:
                if not parent_source.is_dir():
                    continue
                try:
                    relative = target.relative_to(parent_target)
                except ValueError:
                    continue
                expected_source = parent_source / relative
                if os.path.realpath(expected_source) == os.path.realpath(source):
                    covered = True
                    break
            if not covered:
                result.append((source, target))
        return tuple(result)

    def _open_readonly_mounts(
        self,
        command: Sequence[str],
        env: Mapping[str, str],
    ) -> tuple[_FdMount, ...]:
        del command  # command closure is supplied by declarative provisioning
        same_path_candidates = [
            *(
                self._expand_host_path(path)
                for path in self._string_setting(
                    "base_readonly_binds",
                    BASE_READONLY_BINDS,
                )
            ),
            *self._compatibility_readonly_binds(env),
            *self._effective_extra_readonly_binds(),
        ]
        if self._bool_setting("auto_bind_citra_runtime", False):
            same_path_candidates.extend(self._citra_runtime_readonly_binds())

        workspace = self.__workspace
        mounts: list[tuple[Path, Path]] = [
            (path, path)
            for path in same_path_candidates
            if path.exists()
        ]
        runtime = getattr(workspace, "runtime", None)
        if isinstance(runtime, Path) and runtime.exists():
            mounts.append((runtime, runtime))
        mounts.extend(
            (Path(source), Path(target))
            for source, target in getattr(
                workspace,
                "runtime_readonly_binds",
                (),
            )
            if Path(source).exists()
        )
        return self._open_mounts(self._minimal_mount_pairs(mounts))

    def _open_writable_mounts(
        self,
        turn_dirs: Mapping[str, Path],
    ) -> tuple[_FdMount, ...]:
        paths = [
            *(turn_dirs[name] for name in SANDBOX_WRITABLE_DIRS),
            *(
                self._expand_host_path(path)
                for path in self._string_setting(
                    "extra_writable_binds",
                    EXTRA_WRITABLE_BINDS,
                )
            ),
        ]
        return self._open_mounts(
            (path, path)
            for path in paths
            if path.exists()
        )

    def _open_source_mounts(self) -> tuple[_FdMount, ...]:
        if bool(getattr(self.__workspace, "direct_source", False)):
            return ()
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
            # Direct library/test use does not necessarily pass through the
            # launcher.  Infer the installed package root without broadening
            # the sandbox beyond this one explicit path.
            return Path(__file__).resolve().parents[3]

        root = Path(raw).expanduser().resolve()

        if not root.is_dir():
            raise NotADirectoryError(
                f"Citra installation directory does not exist: {root}"
            )

        return root

    def _citra_private_directories(self) -> tuple[Path, ...]:
        workspace = self.__workspace
        candidates = [workspace.library]
        configured = os.environ.get("CITRA_CONFIG_PATH")
        if configured:
            configured_path = Path(configured).expanduser().resolve()
            if configured_path.is_dir():
                candidates.append(configured_path)
        state_root = os.environ.get("CITRA_ROOT")
        if state_root:
            root = Path(state_root).expanduser().resolve()
            if root.is_dir() and root != workspace.source_workspace:
                candidates.append(root)

        result: list[Path] = []
        for path in sorted(set(candidates), key=lambda item: len(item.parts)):
            if any(self._is_inside(path, parent) for parent in result):
                continue
            result.append(path)
        return tuple(result)

    @staticmethod
    def _citra_private_config_files() -> tuple[Path, ...]:
        """Return every operator config file that must be hidden in sandboxes."""
        configured = os.environ.get("CITRA_CONFIG_PATH")
        if configured:
            configured_path = Path(configured).expanduser().resolve()
            if configured_path.is_dir():
                return tuple(
                    sorted(
                        path.resolve()
                        for path in configured_path.glob("*.toml")
                        if path.is_file()
                    )
                )
            return (configured_path,)

        state_root = os.environ.get("CITRA_ROOT")
        if state_root:
            root = Path(state_root).expanduser().resolve()
            config_dir = root / CITRA_CONFIG_DIRECTORY
            if config_dir.is_dir():
                return tuple(
                    sorted(
                        path.resolve()
                        for path in config_dir.glob("*.toml")
                        if path.is_file()
                    )
                )
            return (root / CITRA_LEGACY_PRIVATE_CONFIG_FILE,)

        return ()

    def _private_path_targets(
        self,
        path: Path,
        source_alias: Path,
    ) -> tuple[Path, ...]:
        targets = [path]
        try:
            relative = path.relative_to(self.__workspace.source_workspace)
        except ValueError:
            pass
        else:
            targets.append(source_alias / relative)
        return tuple(dict.fromkeys(targets))

    def _overlaps_metadata(self, target: Path) -> bool:
        absolute = target.absolute()
        if absolute == Path("/"):
            # The opt-in broad root is narrowed by the unconditional runtime
            # root tmpfs above.
            return False
        metadata = self.__workspace.metadata.absolute()
        return self._is_inside(metadata, absolute) or self._is_inside(
            absolute,
            metadata,
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

    def _validate_command_mount_coverage(
        self,
        command: Sequence[str],
        readonly_mounts: Sequence[_FdMount],
    ) -> None:
        """Fail before bwrap if the explicit mounts omit an executable."""
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
            if any(
                self._is_inside(path, mount.target)
                for mount in readonly_mounts
            ):
                continue

            writable_roots = getattr(self.__workspace, "writable_roots", ())
            if any(self._is_inside(path, root) for root in writable_roots):
                continue

            raise RuntimeError(
                "Sandbox setup does not declare the command executable: "
                f"{path}. Add it to a tool/runtime definition."
            )

    def _compatibility_readonly_binds(
        self,
        env: Mapping[str, str],
    ) -> list[Path]:
        workspace = self.__workspace
        candidates: list[Path] = []

        if self._bool_setting(
            "auto_bind_masked_path_entries",
            AUTO_BIND_MASKED_PATH_ENTRIES,
        ):
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

        for name in self._string_setting(
            "auto_bind_env_paths",
            AUTO_BIND_ENV_PATHS,
        ):
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
        resolv_conf = Path("/etc/resolv.conf")
        try:
            if not resolv_conf.resolve(strict=True).is_file():
                return None
        except (FileNotFoundError, OSError, RuntimeError):
            return None
        # Bind the already-open file at the conventional path. This works even
        # when the host path is a symlink into a hidden /run hierarchy.
        return resolv_conf

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
        args.extend(
            (
                "--ro-bind-fd",
                str(resolver_fd),
                str(target),
            )
        )
