from __future__ import annotations

from pathlib import Path
from typing import Any


def table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a TOML table.")
    return value


def bool_value(raw: dict[str, Any], name: str, *, section: str, default: bool) -> bool:
    value = raw.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"'{section}.{name}' must be a boolean.")
    return value


def int_value(raw: dict[str, Any], name: str, *, section: str, default: int) -> int:
    value = raw.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"'{section}.{name}' must be a positive integer.")
    return value


def string_value(raw: dict[str, Any], name: str, *, section: str, default: str | None = None) -> str:
    value = raw.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{section}.{name}' must be a non-empty string.")
    return value


def string_tuple(raw: dict[str, Any], name: str, *, section: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    value = raw.get(name, default)
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"'{section}.{name}' must contain non-empty strings.")
    return tuple(value)


def path_tuple(raw: dict[str, Any], name: str, *, section: str, default: tuple[Path, ...] = ()) -> tuple[Path, ...]:
    return tuple(Path(value).expanduser() for value in string_tuple(raw, name, section=section, default=tuple(map(str, default))))
