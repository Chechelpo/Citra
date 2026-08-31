from __future__ import annotations

import tomllib

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from citra.config._constants import TOOLS_CONFIG_FILE
from citra.config._file_config import TomlConfig

@dataclass(frozen=True)
class WebSearchConfig:
    host_url: str = field()

    @classmethod
    def create(
        cls,
        raw: dict[str, Any],
    ) -> WebSearchConfig:
        return cls(
            host_url=_required_str(
                raw,
                "host_url",
                section="web-search",
            ),
        )


@dataclass(frozen=True)
class BashConfig:
    always_allow_network: bool = False
    permission_timeout: int = 30

    @classmethod
    def create(
        cls,
        raw: dict[str, Any],
    ) -> BashConfig:
        return cls(
            always_allow_network=_bool(
                raw,
                "always_allow_network",
                section="bash",
                default=False,
            ),
            permission_timeout=_positive_int(
                raw,
                "permission_timeout",
                section="bash",
                default=30,
            ),
        )


@dataclass(frozen=True)
class SubprocessConfig:
    always_allow_network: bool = False
    permission_timeout: int = 30
    max_output_length: int = 100_000

    @classmethod
    def create(
        cls,
        raw: dict[str, Any],
    ) -> SubprocessConfig:
        return cls(
            always_allow_network=_bool(
                raw,
                "always_allow_network",
                section="subprocess",
                default=False,
            ),
            permission_timeout=_positive_int(
                raw,
                "permission_timeout",
                section="subprocess",
                default=30,
            ),
            max_output_length=_positive_int(
                raw,
                "max_output_length",
                section="subprocess",
                default=100_000,
            ),
        )


@dataclass(frozen=True)
class BrowserConfig:
    always_allow_network: bool = False
    permission_timeout: int = 30
    request_timeout: float = 30.0
    browsers_path: str = "~/.cache/ms-playwright"
    enabled_unsafe_actions: tuple[str, ...] = ()
    always_allow_unsafe_actions: bool = False

    @classmethod
    def create(
        cls,
        raw: dict[str, Any],
    ) -> BrowserConfig:
        defaults = cls()

        return cls(
            always_allow_network=_bool(
                raw,
                "always_allow_network",
                section="browser",
                default=defaults.always_allow_network,
            ),
            permission_timeout=_positive_int(
                raw,
                "permission_timeout",
                section="browser",
                default=defaults.permission_timeout,
            ),
            request_timeout=_positive_float(
                raw,
                "request_timeout",
                section="browser",
                default=defaults.request_timeout,
            ),
            browsers_path=_str(
                raw,
                "browsers_path",
                section="browser",
                default=defaults.browsers_path,
            ),
            enabled_unsafe_actions=_string_tuple(
                raw,
                "enabled_unsafe_actions",
                section="browser",
                default=defaults.enabled_unsafe_actions,
            ),
            always_allow_unsafe_actions=_bool(
                raw,
                "always_allow_unsafe_actions",
                section="browser",
                default=defaults.always_allow_unsafe_actions,
            ),
        )


@dataclass(frozen=True)
class ToolConfigs(TomlConfig):
    web_search: WebSearchConfig
    bash: BashConfig
    subprocess: SubprocessConfig
    browser: BrowserConfig

    @classmethod
    def load(
        cls,
        config_dir: Path,
    ) -> ToolConfigs:
        path = config_dir / TOOLS_CONFIG_FILE

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing tool configuration file: {path}"
            )

        with path.open("rb") as file:
            raw = tomllib.load(file)

        if not isinstance(raw, dict):
            raise ValueError(
                f"{TOOLS_CONFIG_FILE} must contain a TOML table."
            )

        return cls.create(raw)

    @classmethod
    def create(
        cls,
        raw: dict[str, Any],
    ) -> ToolConfigs:
        return cls(
            web_search=WebSearchConfig.create(
                _table(raw, "web-search"),
            ),
            bash=BashConfig.create(
                _table(raw, "bash"),
            ),
            subprocess=SubprocessConfig.create(
                _table(raw, "subprocess"),
            ),
            browser=BrowserConfig.create(
                _table(raw, "browser"),
            ),
        )


def _table(
    raw: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    value = raw.get(
        name,
        {},
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"'{name}' must be a TOML table."
        )

    return value


def _bool(
    table: dict[str, Any],
    name: str,
    *,
    section: str,
    default: bool,
) -> bool:
    value = table.get(
        name,
        default,
    )

    if not isinstance(value, bool):
        raise ValueError(
            f"'{section}.{name}' must be a boolean."
        )

    return value


def _positive_int(
    table: dict[str, Any],
    name: str,
    *,
    section: str,
    default: int,
) -> int:
    value = table.get(
        name,
        default,
    )

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(
            f"'{section}.{name}' must be a positive integer."
        )

    return value


def _positive_float(
    table: dict[str, Any],
    name: str,
    *,
    section: str,
    default: float,
) -> float:
    value = table.get(
        name,
        default,
    )

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(
            f"'{section}.{name}' must be a positive number."
        )

    return float(value)


def _required_str(
    table: dict[str, Any],
    name: str,
    *,
    section: str,
) -> str:
    if name not in table:
        raise ValueError(
            f"Missing required config value: {section}.{name}"
        )

    value = table[name]

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"'{section}.{name}' must be a non-empty string."
        )

    return value


def _str(
    table: dict[str, Any],
    name: str,
    *,
    section: str,
    default: str,
) -> str:
    value = table.get(
        name,
        default,
    )

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"'{section}.{name}' must be a non-empty string."
        )

    return value


def _string_tuple(
    table: dict[str, Any],
    name: str,
    *,
    section: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = table.get(
        name,
        default,
    )

    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"'{section}.{name}' must be an array of strings."
        )

    if not all(
        isinstance(item, str) and item
        for item in value
    ):
        raise ValueError(
            f"'{section}.{name}' must contain only non-empty strings."
        )

    return tuple(value)