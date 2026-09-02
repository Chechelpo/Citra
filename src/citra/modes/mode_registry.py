from __future__ import annotations

import tomllib
from collections.abc import Iterable
from pathlib import Path

from .mode import Mode
from .chat import ChatMode


MODE_CONFIG_FILE = "mode.toml"
BUILTIN_DEFAULT_MODE = ChatMode._NAME


class ModeRegistry:
    """Registry and pre-runtime selector for available operating modes."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        modes: Iterable[Mode] | None = None,
        default_mode: str | None = None,
    ) -> None:
        if modes is None:
            from citra.modes.simple_task import SimpleTask
            modes = (
                ChatMode(),
                SimpleTask()
            )
        installed = tuple(modes)
        self._modes: dict[str, Mode] = {}
        for mode in installed:
            mode.validate()
            if mode.name in self._modes:
                raise ValueError(f"Duplicate mode name: {mode.name!r}")
            self._modes[mode.name] = mode

        if not self._modes:
            raise ValueError("At least one mode must be registered")

        configured_default = (
            default_mode
            if default_mode is not None
            else self._load_default(config_path)
        )
        if configured_default is None:
            configured_default = BUILTIN_DEFAULT_MODE
        if configured_default not in self._modes:
            raise ValueError(
                f"Unknown default mode {configured_default!r}. "
                f"Available modes: {', '.join(self._modes)}"
            )

        self._default_mode = self._modes[configured_default]
        self._active_mode = self._default_mode

    @staticmethod
    def _mode_config_path(
        config_path: str | Path | None,
    ) -> Path | None:
        if config_path is None:
            return None
        path = Path(config_path).expanduser().resolve()
        return (
            path / MODE_CONFIG_FILE
            if path.is_dir()
            else path.parent / MODE_CONFIG_FILE
        )

    @classmethod
    def _load_default(
        cls,
        config_path: str | Path | None,
    ) -> str | None:
        path = cls._mode_config_path(config_path)
        if path is None or not path.is_file():
            return None
        with path.open("rb") as file:
            raw = tomllib.load(file)
        unexpected = set(raw) - {"default"}
        if unexpected:
            raise ValueError(
                f"{MODE_CONFIG_FILE} contains unsupported key(s): "
                + ", ".join(sorted(unexpected))
            )
        default = raw.get("default")
        if not isinstance(default, str) or not default.strip():
            raise ValueError(
                f"'{MODE_CONFIG_FILE}.default' must be a non-empty string"
            )
        return default.strip()

    @property
    def modes(self) -> tuple[Mode, ...]:
        return tuple(self._modes.values())

    @property
    def default_mode(self) -> Mode:
        return self._default_mode

    @property
    def active_mode(self) -> Mode:
        return self._active_mode

    def get(self, name: str) -> Mode:
        try:
            return self._modes[name]
        except KeyError as error:
            raise KeyError(
                f"Unknown mode {name!r}. Available modes: "
                + ", ".join(self._modes)
            ) from error

    def select(self, selection: str | None = None) -> Mode:
        value = (selection or "").strip()
        if not value:
            self._active_mode = self._default_mode
            return self._active_mode

        if value.isdecimal():
            index = int(value)
            if 1 <= index <= len(self._modes):
                self._active_mode = self.modes[index - 1]
                return self._active_mode
            raise ValueError(
                f"Mode number must be between 1 and {len(self._modes)}"
            )

        self._active_mode = self.get(value)
        return self._active_mode
