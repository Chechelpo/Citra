"""Private last-process diagnostics for the Citra controller."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import platform
import time
from types import TracebackType
from typing import IO, Iterator

from citra.logging import ERROR_LOG_NAME, LATEST_LOG_NAME, LOG_DIRECTORY_NAME


# Compatibility alias for callers that imported the previous constant name.
LAST_PROCESS_LOG_NAME = LATEST_LOG_NAME


class _CitraLogFilter(logging.Filter):
    """Keep dependency debug chatter out of the project diagnostic log."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Keep Citra records and attach a module-and-line origin."""
        if record.name != "citra" and not record.name.startswith("citra."):
            return False
        record_fields = vars(record)
        declared_origin = record_fields.setdefault(
            "_citra_declared_origin",
            record_fields.get("origin", record.name),
        )
        record.origin = f"{declared_origin}:{record.lineno}"
        return True


class _UtcFormatter(logging.Formatter):
    """Represent UtcFormatter."""
    converter = staticmethod(time.gmtime)


class _LazyWarningErrorHandler(logging.Handler):
    """Write Citra warnings to a private file only after the first error."""

    terminator = "\n"

    def __init__(self, path: Path) -> None:
        """Configure a delayed warning/error destination at ``path``."""
        super().__init__(level=logging.WARNING)
        self._path = path
        self._pending_warnings: list[str] = []
        self._stream: IO[str] | None = None

    def emit(self, record: logging.LogRecord) -> None:
        """Buffer warnings and activate the destination for errors or criticals."""
        try:
            rendered = self.format(record)
            if self._stream is None and record.levelno < logging.ERROR:
                self._pending_warnings.append(rendered)
                return
            if self._stream is None:
                self._stream = _open_private_log(self._path)
                for warning in self._pending_warnings:
                    self._write(warning)
                self._pending_warnings.clear()
            self._write(rendered)
        except Exception:
            self.handleError(record)

    def _write(self, rendered: str) -> None:
        """Append and immediately flush one preformatted record."""
        if self._stream is None:
            return
        self._stream.write(rendered + self.terminator)
        self._stream.flush()

    def close(self) -> None:
        """Release an activated error stream and discard buffered warnings."""
        self.acquire()
        try:
            self._pending_warnings.clear()
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        finally:
            self.release()
            super().close()


def _open_private_log(path: Path) -> IO[str]:
    """Create or truncate a line-buffered owner-only UTF-8 log file."""
    descriptor = os.open(
        path,
        os.O_APPEND | os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
        0o600,
    )
    os.chmod(path, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)


def _log_formatter() -> _UtcFormatter:
    """Build the shared UTC formatter used by all process log files."""
    return _UtcFormatter(
        "%(asctime)sZ %(levelname)s %(name)s [%(threadName)s] "
        "[%(origin)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


@contextmanager
def process_log(log_directory: str | Path) -> Iterator[Path]:
    """Capture Citra logs in one process runtime's ``logs/latest.log``.

    The main file is truncated at process start, uses owner-only permissions,
    and is flushed after every record so a crash still leaves diagnostics. A
    sibling ``errors.log`` buffers warnings in memory and is created only when
    the first error occurs. Existing application handlers are preserved and
    restored. Configuration remains controller-owned under ``CITRA_ROOT/logs``;
    this function writes only to the supplied lifecycle directory.
    """

    log_directory = Path(log_directory).expanduser().resolve()
    log_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_directory.chmod(0o700)

    log_path = log_directory / LAST_PROCESS_LOG_NAME
    stream = _open_private_log(log_path)

    error_log_path = log_directory / ERROR_LOG_NAME
    error_log_path.unlink(missing_ok=True)

    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    handler.addFilter(_CitraLogFilter())
    handler.setFormatter(_log_formatter())

    error_handler = _LazyWarningErrorHandler(error_log_path)
    error_handler.addFilter(_CitraLogFilter())
    error_handler.setFormatter(_log_formatter())

    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.addHandler(handler)
    root_logger.addHandler(error_handler)
    if previous_level == logging.NOTSET or previous_level > logging.DEBUG:
        root_logger.setLevel(logging.DEBUG)

    process_logger = logging.getLogger("citra.process")
    process_logger.info(
        "Citra process started | pid=%d | runtime=%s | python=%s | "
        "platform=%s | time=%s",
        os.getpid(),
        log_directory.parent,
        platform.python_version(),
        platform.platform(),
        datetime.now(timezone.utc).isoformat(),
    )

    error: BaseException | None = None
    traceback: TracebackType | None = None
    try:
        yield log_path
    except BaseException as caught:
        error = caught
        traceback = caught.__traceback__
        process_logger.critical(
            "Citra process terminated unexpectedly.",
            exc_info=(type(caught), caught, traceback),
        )
        raise
    finally:
        if error is None:
            process_logger.info("Citra process stopped normally.")
        handler.flush()
        root_logger.removeHandler(handler)
        root_logger.removeHandler(error_handler)
        root_logger.setLevel(previous_level)
        handler.close()
        error_handler.close()
        stream.close()


__all__ = [
    "ERROR_LOG_NAME",
    "LATEST_LOG_NAME",
    "LAST_PROCESS_LOG_NAME",
    "LOG_DIRECTORY_NAME",
    "process_log",
]
