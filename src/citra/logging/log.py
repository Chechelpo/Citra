from __future__ import annotations

import json
import logging
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
    "warn": 40,
    "warning": 40,
    "error": 50,
}


class Logger:
    """Write compact structured diagnostics without breaking application code."""

    def __init__(self, source: str):
        """Create a source-labelled logger with optional file persistence."""
        self.source = source
        self._lock = threading.Lock()

        self.log_dir = self._resolve_log_dir()
        self.level = self._load_level()

    def _resolve_log_dir(self) -> Path | None:
        """Resolve the configured log directory, or use stdlib logging only."""
        config_path = os.environ.get("CITRA_ROOT") or os.environ.get(
            "CITRA_CONFIG_PATH"
        )

        if not config_path:
            return None

        path = Path(config_path)
        if path.suffix == ".toml":
            log_dir = path.parent / "logs"
        else:
            log_dir = path / "logs"

        log_dir.mkdir(parents=True, exist_ok=True)

        return log_dir

    def _load_level(self) -> int:
        """Load the configured threshold while keeping logging non-fatal."""
        if self.log_dir is None:
            return LEVELS["trace"]
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
        """Emit one structured record at the requested severity."""
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

        if self.log_dir is None:
            std_level = {
                "trace": logging.DEBUG,
                "debug": logging.DEBUG,
                "info": logging.INFO,
                "warn": logging.WARNING,
                "warning": logging.WARNING,
                "error": logging.ERROR,
            }[level]
            logging.getLogger(f"citra.{self.source}").log(
                std_level,
                message,
                extra={"origin": self.source, **fields},
            )
            return
        try:
            with self._lock:
                with (self.log_dir / "latest.txt").open(
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logging.getLogger(f"citra.{self.source}").exception(
                "Could not persist structured log record",
                extra={"origin": self.source},
            )

    def trace(self, message: str, **fields: Any) -> None:
        """Emit a trace-level record."""
        self._write("trace", message, **fields)

    def debug(self, message: str, **fields: Any) -> None:
        """Emit a debug-level record."""
        self._write("debug", message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        """Emit an info-level record."""
        self._write("info", message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        """Emit a warning-level record."""
        self._write("warning", message, **fields)

    def warn(self, message: str, **fields: Any) -> None:
        """Emit a warn-level record using the requested level spelling."""
        self._write("warn", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        """Emit an error-level record."""
        self._write("error", message, **fields)
