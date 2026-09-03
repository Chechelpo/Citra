from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from citra.config._constants import SANDBOX_CONFIG_FILE
from citra.config._file_config import TomlConfig
from citra.config.runtime_discovery import (
    RO_BINDS,
    RuntimeDiscoveryResult,
)
from citra.sandbox.sandbox_mode import SandboxMode


class WorkflowSandboxConfig(Protocol):
    @property
    def mode(self) -> SandboxMode: ...

    @property
    def additional_ro_binds(self) -> tuple[Path, ...]: ...

    @property
    def additional_w_binds(self) -> tuple[Path, ...]: ...

    @property
    def global_network_disallow(self) -> bool: ...


@dataclass
class SandboxPolicy(TomlConfig):
    """
    Complete mutable policy for one sandbox lifecycle.

    Workflows, runtime discovery, and operator configuration may
    contribute to this object before WorkspaceSandbox is constructed.

    WorkspaceSandbox consumes only the finalized SandboxPolicy.
    """

    mode: SandboxMode = SandboxMode.FULL_SANDBOX

    extra_ro_binds: list[Path] = field(
        default_factory=list,
    )

    extra_w_binds: list[Path] = field(
        default_factory=list,
    )

    runtime_results: list[RuntimeDiscoveryResult] = field(
        default_factory=lambda: list(RO_BINDS),
    )

    base_readonly_binds: list[Path] = field(
        default_factory=lambda: [
            Path("/usr/bin")
        ]
    )

    masked_host_dirs: list[Path] = field(
        default_factory=lambda: [
            Path("/home"),
            Path("/root"),
            Path("/run"),
            Path("/tmp"),
            Path("/var/tmp"),
            Path("/mnt"),
            Path("/media"),
            Path("/srv"),
            Path("/boot"),
        ],
    )

    masked_host_files: list[Path] = field(
        default_factory=list,
    )

    extra_device_binds: list[Path] = field(
        default_factory=list,
    )

    private_files: list[Path] = field(
        default_factory=list,
    )

    citra_config_exclude: list[str] = field(
        default_factory=lambda: [
            "citra.config.toml",
        ],
    )

    auto_bind_env_paths: list[str] = field(
        default_factory=list,
    )

    drop_environment_variables: list[str] = field(
        default_factory=lambda: [
            "DBUS_SESSION_BUS_ADDRESS",
            "SSH_AUTH_SOCK",
            "SSH_AGENT_PID",
            "GPG_AGENT_INFO",
            "GIT_DIR",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        ],
    )

    drop_environment_prefixes: list[str] = field(
        default_factory=list,
    )

    global_disallow_network: bool = False

    auto_bind_citra_runtime: bool = False
    auto_bind_citra_config: bool = True
    auto_bind_masked_path_entries: bool = False
    auto_bind_resolv_conf_target: bool = True

    unshare_user_try: bool = True
    unshare_pid: bool = True
    unshare_ipc: bool = True
    unshare_uts: bool = True
    unshare_cgroup_try: bool = False

    new_terminal_session: bool = True
    disable_nested_user_namespaces: bool = False

    FILE_NAME = SANDBOX_CONFIG_FILE

    @classmethod
    def create(
        cls,
        raw: dict[str, Any],
    ) -> SandboxPolicy:
        sandbox = raw.get(
            "sandbox",
            raw,
        )

        if not isinstance(sandbox, dict):
            raise ValueError(
                "'sandbox' must be a TOML table."
            )

        defaults = cls()

        return cls(
            extra_ro_binds=_path_list(
                sandbox,
                "extra_readonly_binds",
                default=defaults.extra_ro_binds,
            ),

            extra_w_binds=_path_list(
                sandbox,
                "extra_writable_binds",
                default=defaults.extra_w_binds,
            ),

            runtime_results=list(
                defaults.runtime_results
            ),

            base_readonly_binds=_path_list(
                sandbox,
                "base_readonly_binds",
                default=defaults.base_readonly_binds,
            ),

            masked_host_dirs=_path_list(
                sandbox,
                "masked_host_dirs",
                default=defaults.masked_host_dirs,
            ),

            masked_host_files=_path_list(
                sandbox,
                "masked_host_files",
                default=defaults.masked_host_files,
            ),

            extra_device_binds=_path_list(
                sandbox,
                "extra_device_binds",
                default=defaults.extra_device_binds,
            ),

            private_files=_path_list(
                sandbox,
                "private_files",
                default=defaults.private_files,
            ),

            citra_config_exclude=_string_list(
                sandbox,
                "citra_config_exclude",
                default=defaults.citra_config_exclude,
            ),

            auto_bind_env_paths=_string_list(
                sandbox,
                "auto_bind_env_paths",
                default=defaults.auto_bind_env_paths,
            ),

            drop_environment_variables=_string_list(
                sandbox,
                "drop_environment_variables",
                default=defaults.drop_environment_variables,
            ),

            drop_environment_prefixes=_string_list(
                sandbox,
                "drop_environment_prefixes",
                default=defaults.drop_environment_prefixes,
            ),

            global_disallow_network=_bool(
                sandbox,
                "global_network_disallow",
                default=defaults.global_disallow_network,
            ),

            auto_bind_citra_runtime=_bool(
                sandbox,
                "auto_bind_citra_runtime",
                default=defaults.auto_bind_citra_runtime,
            ),

            auto_bind_citra_config=_bool(
                sandbox,
                "auto_bind_citra_config",
                default=defaults.auto_bind_citra_config,
            ),

            auto_bind_masked_path_entries=_bool(
                sandbox,
                "auto_bind_masked_path_entries",
                default=defaults.auto_bind_masked_path_entries,
            ),

            auto_bind_resolv_conf_target=_bool(
                sandbox,
                "auto_bind_resolv_conf_target",
                default=defaults.auto_bind_resolv_conf_target,
            ),

            unshare_user_try=_bool(
                sandbox,
                "unshare_user_try",
                default=defaults.unshare_user_try,
            ),

            unshare_pid=_bool(
                sandbox,
                "unshare_pid",
                default=defaults.unshare_pid,
            ),

            unshare_ipc=_bool(
                sandbox,
                "unshare_ipc",
                default=defaults.unshare_ipc,
            ),

            unshare_uts=_bool(
                sandbox,
                "unshare_uts",
                default=defaults.unshare_uts,
            ),

            unshare_cgroup_try=_bool(
                sandbox,
                "unshare_cgroup_try",
                default=defaults.unshare_cgroup_try,
            ),

            new_terminal_session=_bool(
                sandbox,
                "new_terminal_session",
                default=defaults.new_terminal_session,
            ),

            disable_nested_user_namespaces=_bool(
                sandbox,
                "disable_nested_user_namespaces",
                default=defaults.disable_nested_user_namespaces,
            ),
        )

    def add_readonly_bind(
        self,
        path: str | Path,
    ) -> None:
        path = _path(path)

        if path not in self.extra_ro_binds:
            self.extra_ro_binds.append(path)

    def add_writable_bind(
        self,
        path: str | Path,
    ) -> None:
        path = _path(path)

        if path not in self.extra_w_binds:
            self.extra_w_binds.append(path)

    def add_runtime_result(
        self,
        result: RuntimeDiscoveryResult,
    ) -> None:
        if not isinstance(
            result,
            RuntimeDiscoveryResult,
        ):
            raise TypeError(
                "Runtime result must be RuntimeDiscoveryResult."
            )

        self.runtime_results.append(
            result
        )

    def apply_workflow_config(
        self,
        config: WorkflowSandboxConfig,
    ) -> None:
        """Merge one workflow contribution into this operator policy.

        Bind lists are additive.  Network denial is monotonic: either the
        selected workflow or the operator configuration may deny it.
        """
        if not isinstance(config.mode, SandboxMode):
            raise TypeError("Sandbox mode contribution has no valid mode.")

        mode_readonly = tuple(
            _path(path)
            for path in config.additional_ro_binds
        )
        mode_writable = tuple(
            _path(path)
            for path in config.additional_w_binds
        )

        self.mode = config.mode
        self.extra_ro_binds = list(
            dict.fromkeys((*mode_readonly, *self.extra_ro_binds))
        )
        self.extra_w_binds = list(
            dict.fromkeys((*mode_writable, *self.extra_w_binds))
        )
        self.global_disallow_network = bool(
            self.global_disallow_network
            or config.global_network_disallow
        )

    def apply_mode_config(self, config: WorkflowSandboxConfig) -> None:
        """Compatibility alias for :meth:`apply_workflow_config`."""
        self.apply_workflow_config(config)

    def clone(self) -> SandboxPolicy:
        """Return an independent mutable policy for one sandbox lifecycle."""
        return SandboxPolicy(
            mode=self.mode,
            extra_ro_binds=list(self.extra_ro_binds),
            extra_w_binds=list(self.extra_w_binds),
            runtime_results=list(self.runtime_results),
            base_readonly_binds=list(self.base_readonly_binds),
            masked_host_dirs=list(self.masked_host_dirs),
            masked_host_files=list(self.masked_host_files),
            extra_device_binds=list(self.extra_device_binds),
            private_files=list(self.private_files),
            citra_config_exclude=list(self.citra_config_exclude),
            auto_bind_env_paths=list(self.auto_bind_env_paths),
            drop_environment_variables=list(self.drop_environment_variables),
            drop_environment_prefixes=list(self.drop_environment_prefixes),
            global_disallow_network=self.global_disallow_network,
            auto_bind_citra_runtime=self.auto_bind_citra_runtime,
            auto_bind_citra_config=self.auto_bind_citra_config,
            auto_bind_masked_path_entries=self.auto_bind_masked_path_entries,
            auto_bind_resolv_conf_target=self.auto_bind_resolv_conf_target,
            unshare_user_try=self.unshare_user_try,
            unshare_pid=self.unshare_pid,
            unshare_ipc=self.unshare_ipc,
            unshare_uts=self.unshare_uts,
            unshare_cgroup_try=self.unshare_cgroup_try,
            new_terminal_session=self.new_terminal_session,
            disable_nested_user_namespaces=(
                self.disable_nested_user_namespaces
            ),
        )

    @property
    def runtime_readonly_binds(
        self,
    ) -> tuple[Path, ...]:
        paths: list[Path] = []

        for result in self.runtime_results:
            for path in result.readonly_binds:
                path = _path(path)

                if path not in paths:
                    paths.append(path)

        return tuple(paths)

    @property
    def readonly_binds(
        self,
    ) -> tuple[Path, ...]:
        """
        Return every policy-defined read-only bind.

        WorkspaceSandbox does not need to know whether a bind came from
        configuration, a mode, or runtime discovery.
        """
        return tuple(
            dict.fromkeys(
                (
                    *self.base_readonly_binds,
                    *self.extra_ro_binds,
                    *self.runtime_readonly_binds,
                )
            )
        )

    @property
    def writable_binds(
        self,
    ) -> tuple[Path, ...]:
        return tuple(
            dict.fromkeys(
                self.extra_w_binds
            )
        )


def _path(
    value: str | Path,
) -> Path:
    return Path(
        value
    ).expanduser()


def _bool(
    table: dict[str, Any],
    name: str,
    *,
    default: bool,
) -> bool:
    value = table.get(
        name,
        default,
    )

    if not isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"'sandbox.{name}' must be a boolean."
        )

    return value


def _string_list(
    table: dict[str, Any],
    name: str,
    *,
    default: list[str],
) -> list[str]:
    value = table.get(
        name,
        default,
    )

    if not isinstance(
        value,
        (list, tuple),
    ):
        raise ValueError(
            f"'sandbox.{name}' must be an array of strings."
        )

    if not all(
        isinstance(item, str)
        for item in value
    ):
        raise ValueError(
            f"'sandbox.{name}' must contain only strings."
        )

    return list(value)


def _path_list(
    table: dict[str, Any],
    name: str,
    *,
    default: list[Path],
) -> list[Path]:
    values = _string_list(
        table,
        name,
        default=[
            str(path)
            for path in default
        ],
    )

    return [
        _path(value)
        for value in values
    ]
