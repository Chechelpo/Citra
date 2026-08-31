from __future__ import annotations

import tomllib

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from citra.config._constants import SANDBOX_CONFIG_FILE
from citra.config._file_config import TomlConfig

BASE_READONLY_BINDS: tuple[str, ...] = ()


@dataclass(frozen=True)
class SandboxPolicy(TomlConfig):
    """Operator-controlled sandbox policy."""

    extra_ro_binds: tuple[Path, ...] = field(default=())
    extra_w_binds: tuple[Path, ...] = field(default=())

    global_disallow_network: bool = field(default=False)

    base_readonly_binds: tuple[Path, ...] = field(default=())

    masked_host_dirs: tuple[Path, ...] = field(
        default=(
            Path("/home"),
            Path("/root"),
            Path("/run"),
            Path("/tmp"),
            Path("/var/tmp"),
            Path("/mnt"),
            Path("/media"),
            Path("/srv"),
            Path("/boot"),
        ),
    )

    masked_host_files: tuple[Path, ...] = field(default=())

    extra_device_binds: tuple[Path, ...] = field(default=())

    private_files: tuple[Path, ...] = field(default=())

    auto_bind_citra_runtime: bool = field(default=False)
    auto_bind_citra_config: bool = field(default=True)

    citra_config_exclude: tuple[str, ...] = field(
        default=("citra.configtoml",),
    )

    auto_bind_masked_path_entries: bool = field(default=False)

    auto_bind_env_paths: tuple[str, ...] = field(default=())

    auto_bind_resolv_conf_target: bool = field(default=True)

    unshare_user_try: bool = field(default=True)
    unshare_pid: bool = field(default=True)
    unshare_ipc: bool = field(default=True)
    unshare_uts: bool = field(default=True)
    unshare_cgroup_try: bool = field(default=False)

    new_terminal_session: bool = field(default=True)

    disable_nested_user_namespaces: bool = field(default=False)

    drop_environment_variables: tuple[str, ...] = field(
        default=(
            "DBUS_SESSION_BUS_ADDRESS",
            "SSH_AUTH_SOCK",
            "SSH_AGENT_PID",
            "GPG_AGENT_INFO",
            "GIT_DIR",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        ),
    )

    drop_environment_prefixes: tuple[str, ...] = field(default=())


    @classmethod
    def load(
        cls,
        config_dir: Path,
    ) -> SandboxPolicy:
        path = config_dir / SANDBOX_CONFIG_FILE

        if not path.is_file():
            raise FileNotFoundError(
                f"Sandbox config file not found: {path}"
            )

        with path.open("rb") as file:
            raw = tomllib.load(file)

        if not isinstance(raw, dict):
            raise ValueError(
                f"{SANDBOX_CONFIG_FILE} must contain a TOML table."
            )

        return cls.create(raw)


    @classmethod
    def create(
        cls,
        raw: dict[str, Any],
    ) -> SandboxPolicy:
        defaults = cls()

        sandbox_raw = raw.get(
            "sandbox",
            raw,
        )

        if not isinstance(sandbox_raw, dict):
            raise ValueError(
                "'sandbox' must be a TOML table."
            )

        return cls(
            extra_ro_binds=_path_tuple(
                sandbox_raw,
                "extra_readonly_binds",
                section="sandbox",
                default=defaults.extra_ro_binds,
            ),
            extra_w_binds=_path_tuple(
                sandbox_raw,
                "extra_writable_binds",
                section="sandbox",
                default=defaults.extra_w_binds,
            ),
            global_disallow_network=_bool(
                sandbox_raw,
                "global_network_disallow",
                section="sandbox",
                default=defaults.global_disallow_network,
            ),
            base_readonly_binds=_path_tuple(
                sandbox_raw,
                "base_readonly_binds",
                section="sandbox",
                default=defaults.base_readonly_binds,
            ),
            masked_host_dirs=_path_tuple(
                sandbox_raw,
                "masked_host_dirs",
                section="sandbox",
                default=defaults.masked_host_dirs,
            ),
            masked_host_files=_path_tuple(
                sandbox_raw,
                "masked_host_files",
                section="sandbox",
                default=defaults.masked_host_files,
            ),
            extra_device_binds=_path_tuple(
                sandbox_raw,
                "extra_device_binds",
                section="sandbox",
                default=defaults.extra_device_binds,
            ),
            private_files=_path_tuple(
                sandbox_raw,
                "private_files",
                section="sandbox",
                default=defaults.private_files,
            ),
            auto_bind_citra_runtime=_bool(
                sandbox_raw,
                "auto_bind_citra_runtime",
                section="sandbox",
                default=defaults.auto_bind_citra_runtime,
            ),
            auto_bind_citra_config=_bool(
                sandbox_raw,
                "auto_bind_citra_config",
                section="sandbox",
                default=defaults.auto_bind_citra_config,
            ),
            citra_config_exclude=_string_tuple(
                sandbox_raw,
                "citra_config_exclude",
                section="sandbox",
                default=defaults.citra_config_exclude,
            ),
            auto_bind_masked_path_entries=_bool(
                sandbox_raw,
                "auto_bind_masked_path_entries",
                section="sandbox",
                default=defaults.auto_bind_masked_path_entries,
            ),
            auto_bind_env_paths=_string_tuple(
                sandbox_raw,
                "auto_bind_env_paths",
                section="sandbox",
                default=defaults.auto_bind_env_paths,
            ),
            auto_bind_resolv_conf_target=_bool(
                sandbox_raw,
                "auto_bind_resolv_conf_target",
                section="sandbox",
                default=defaults.auto_bind_resolv_conf_target,
            ),
            unshare_user_try=_bool(
                sandbox_raw,
                "unshare_user_try",
                section="sandbox",
                default=defaults.unshare_user_try,
            ),
            unshare_pid=_bool(
                sandbox_raw,
                "unshare_pid",
                section="sandbox",
                default=defaults.unshare_pid,
            ),
            unshare_ipc=_bool(
                sandbox_raw,
                "unshare_ipc",
                section="sandbox",
                default=defaults.unshare_ipc,
            ),
            unshare_uts=_bool(
                sandbox_raw,
                "unshare_uts",
                section="sandbox",
                default=defaults.unshare_uts,
            ),
            unshare_cgroup_try=_bool(
                sandbox_raw,
                "unshare_cgroup_try",
                section="sandbox",
                default=defaults.unshare_cgroup_try,
            ),
            new_terminal_session=_bool(
                sandbox_raw,
                "new_terminal_session",
                section="sandbox",
                default=defaults.new_terminal_session,
            ),
            disable_nested_user_namespaces=_bool(
                sandbox_raw,
                "disable_nested_user_namespaces",
                section="sandbox",
                default=defaults.disable_nested_user_namespaces,
            ),
            drop_environment_variables=_string_tuple(
                sandbox_raw,
                "drop_environment_variables",
                section="sandbox",
                default=defaults.drop_environment_variables,
            ),
            drop_environment_prefixes=_string_tuple(
                sandbox_raw,
                "drop_environment_prefixes",
                section="sandbox",
                default=defaults.drop_environment_prefixes,
            ),
        )


def _bool(
    table: dict[str, Any],
    name: str,
    *,
    section: str,
    default: bool,
) -> bool:
    value = table.get(name, default)

    if not isinstance(value, bool):
        raise ValueError(
            f"'{section}.{name}' must be a boolean."
        )

    return value


def _string_tuple(
    table: dict[str, Any],
    name: str,
    *,
    section: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = table.get(name, default)

    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"'{section}.{name}' must be an array of strings."
        )

    if not all(
        isinstance(item, str)
        for item in value
    ):
        raise ValueError(
            f"'{section}.{name}' must contain only strings."
        )

    return tuple(value)


def _path_tuple(
    table: dict[str, Any],
    name: str,
    *,
    section: str,
    default: tuple[Path, ...],
) -> tuple[Path, ...]:
    default_strings = tuple(
        str(path)
        for path in default
    )

    values = _string_tuple(
        table,
        name,
        section=section,
        default=default_strings,
    )

    return tuple(
        Path(value).expanduser()
        for value in values
    )
