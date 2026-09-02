from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any, TypeVar

T = TypeVar('T', bound='TomlConfig')


class TomlConfig:
    FILE_NAME: str

    @classmethod
    def load(cls: type[T], config_dir: Path) -> T:
        path = Path(config_dir).expanduser().resolve() / cls.FILE_NAME
        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open('rb') as file:
            raw = tomllib.load(file)

        if not isinstance(raw, dict):
            raise ValueError(f'{path.name} must contain a TOML table.')

        return cls.create(raw)

    @classmethod
    def create(cls: type[T], raw: dict[str, Any]) -> T:
        raise NotImplementedError
