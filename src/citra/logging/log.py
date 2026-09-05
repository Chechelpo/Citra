from __future__ import annotations

import json
import logging
import os
import tomllib
from pathlib import Path
from threading import current_thread
from typing import Any


LEVELS = {
    "trace": 10,
    "debug": 20,
    "info": 30,
    "warn": 40,
    "warning": 40,
    "error": 50,
}

LOG_DIRECTORY_NAME = "logs"
LATEST_LOG_NAME = "latest.log"


def _resolve_log_config_directory(citra_root: str | Path | None = None) -> Path:
    """Return the installation directory containing log configuration."""
    root = citra_root if citra_root is not None else os.environ.get("CITRA_ROOT")
    if root is None or not str(root).strip():
        raise RuntimeError("CITRA_ROOT is not defined.")
    return Path(root).expanduser() / LOG_DIRECTORY_NAME


class Logger:
    """Emit structured diagnostics through the process logging pipeline."""

    def __init__(self, source: str):
        """Create a source-labelled logger."""
        self.source = source

        self.verbose = (
            os.environ.get(
                "CITRA_LOG_VERBOSE",
                "false",
            ).lower()
            in ("1", "true", "yes", "on")
        )

        self.log_config_dir = self._resolve_log_config_dir()
        self.level = self._load_level()

    def _resolve_log_config_dir(self) -> Path:
        """Resolve the installation directory containing only log config."""
        return _resolve_log_config_directory()

    def _load_level(self) -> int:
        """Load configured log level."""
        config_file = self.log_config_dir / "config.toml"

        if not config_file.exists():
            return LEVELS["trace"]

        try:
            with config_file.open("rb") as f:
                config = tomllib.load(f)

            level = str(
                config.get(
                    "level",
                    "trace",
                )
            ).lower()

            return LEVELS.get(
                level,
                LEVELS["trace"],
            )

        except Exception:
            return LEVELS["trace"]

    def _serialize(self, value: Any) -> str:
        """Serialize log metadata with readable multiline values."""
        try:
            if isinstance(value, dict):
                parts: list[str] = []

                for key, item in value.items():
                    if isinstance(item, str) and "\n" in item:
                        parts.append(
                            f"\n--- {key} ---\n{item}\n--- end {key} ---"
                        )
                    else:
                        parts.append(
                            f"{key}={json.dumps(item, ensure_ascii=False, default=str)}"
                        )

                return " ".join(parts)

            if isinstance(value, str) and "\n" in value:
                return f"\n{value}"

            return json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )

        except Exception:
            return repr(value)

    def _write(
        self,
        level: str,
        event: str | dict[str, Any],
        *,
        logger_name: str | None = None,
        file: str | None = None,
        line: int | None = None,
        **fields: Any,
    ) -> None:
        """Emit one structured log record."""

        if LEVELS[level] < self.level:
            return

        thread = current_thread()

        logger_name = (
            logger_name
            or f"citra.{self.source}"
        )

        metadata: dict[str, Any] = {}

        if isinstance(event, dict):
            metadata.update(event)

            text = str(
                metadata.pop(
                    "message",
                    "structured event",
                )
            )

        else:
            text = event

        metadata.update(fields)

        if self.verbose:
            metadata.update(
                {
                    "pid": os.getpid(),
                    "thread": thread.name,
                    "thread_id": thread.ident,
                    "logger": logger_name,
                }
            )

            if file is not None:
                metadata["file"] = file

            if line is not None:
                metadata["line"] = line

        record = text

        if metadata:
            record += " " + self._serialize(metadata)

        std_level = {
            "trace": logging.DEBUG,
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warn": logging.WARNING,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }[level]
        logging.getLogger(logger_name).log(
            std_level,
            record,
            extra={"origin": self.source},
        )

    def trace(
        self,
        event: str | dict[str, Any],
        *,
        logger_name: str | None = None,
        file: str | None = None,
        line: int | None = None,
        **fields: Any,
    ) -> None:
        self._write(
            "trace",
            event,
            logger_name=logger_name,
            file=file,
            line=line,
            **fields,
        )

    def debug(
        self,
        event: str | dict[str, Any],
        *,
        logger_name: str | None = None,
        file: str | None = None,
        line: int | None = None,
        **fields: Any,
    ) -> None:
        self._write(
            "debug",
            event,
            logger_name=logger_name,
            file=file,
            line=line,
            **fields,
        )

    def info(
        self,
        event: str | dict[str, Any],
        *,
        logger_name: str | None = None,
        file: str | None = None,
        line: int | None = None,
        **fields: Any,
    ) -> None:
        self._write(
            "info",
            event,
            logger_name=logger_name,
            file=file,
            line=line,
            **fields,
        )

    def warning(
        self,
        event: str | dict[str, Any],
        *,
        logger_name: str | None = None,
        file: str | None = None,
        line: int | None = None,
        **fields: Any,
    ) -> None:
        self._write(
            "warning",
            event,
            logger_name=logger_name,
            file=file,
            line=line,
            **fields,
        )

    def warn(
        self,
        event: str | dict[str, Any],
        *,
        logger_name: str | None = None,
        file: str | None = None,
        line: int | None = None,
        **fields: Any,
    ) -> None:
        self._write(
            "warn",
            event,
            logger_name=logger_name,
            file=file,
            line=line,
            **fields,
        )

    def error(
        self,
        event: str | dict[str, Any],
        *,
        logger_name: str | None = None,
        file: str | None = None,
        line: int | None = None,
        **fields: Any,
    ) -> None:
        self._write(
            "error",
            event,
            logger_name=logger_name,
            file=file,
            line=line,
            **fields,
        )
