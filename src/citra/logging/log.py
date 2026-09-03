from __future__ import annotations

import json
import os
import threading
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any


LEVELS = {
    "trace": 10,
    "debug": 20,
    "info": 30,
    "warning": 40,
    "error": 50,
}


class Logger:
    def __init__(self, source: str):
        self.source = source
        self._lock = threading.Lock()

        self.log_dir = self._resolve_log_dir()
        self.level = self._load_level()

    def _resolve_log_dir(self) -> Path:
        config_path = os.environ.get("CITRA_ROOT")

        if not config_path:
            raise RuntimeError("CITRA_CONFIG_PATH is not set")

        path = Path(config_path)
        print(path)
        if path.suffix == ".toml":
            log_dir = path.parent / "logs"
        else:
            log_dir = path / "logs"

        log_dir.mkdir(parents=True, exist_ok=True)

        return log_dir

    def _load_level(self) -> int:
        config_file = self.log_dir / "config.toml"

        if not config_file.exists():
            return LEVELS["trace"]

        try:
            with config_file.open("rb") as f:
                config = tomllib.load(f)

            level = str(config.get("level", "trace")).lower()

            return LEVELS.get(level, LEVELS["trace"])

        except Exception:
            # Logging should never break the application.
            return LEVELS["trace"]

    def _write(self, level: str, message: str, **fields: Any) -> None:
        if LEVELS[level] < self.level:
            return

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "level": level,
            "source": self.source,
            "message": message,
        }

        if fields:
            entry["fields"] = fields

        with self._lock:
            with (self.log_dir / "latest.txt").open(
                "a",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def trace(self, message: str, **fields: Any) -> None:
        self._write("trace", message, **fields)

    def debug(self, message: str, **fields: Any) -> None:
        self._write("debug", message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self._write("info", message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._write("warning", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._write("error", message, **fields)