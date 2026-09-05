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
from typing import Iterator

from citra.logging import LATEST_LOG_NAME, LOG_DIRECTORY_NAME


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


@contextmanager
def process_log(log_directory: str | Path) -> Iterator[Path]:
    """Capture Citra logs in one process runtime's ``logs/latest.log``.

    The file is truncated at process start, uses owner-only permissions, and
    is flushed after every record so a crash still leaves useful diagnostics.
    Existing application logging handlers are preserved and restored. Log
    configuration remains controller-owned under ``CITRA_ROOT/logs``; this
    function writes only to the supplied lifecycle directory.
    """

    log_directory = Path(log_directory).expanduser().resolve()
    log_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_directory.chmod(0o700)

    log_path = log_directory / LAST_PROCESS_LOG_NAME
    descriptor = os.open(
        log_path,
        os.O_APPEND | os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
        0o600,
    )
    os.chmod(log_path, 0o600)
    stream = os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)

    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    handler.addFilter(_CitraLogFilter())
    handler.setFormatter(
        _UtcFormatter(
            "%(asctime)sZ %(levelname)s %(name)s [%(threadName)s] "
            "[%(origin)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.addHandler(handler)
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
        root_logger.setLevel(previous_level)
        handler.close()
        stream.close()


__all__ = [
    "LATEST_LOG_NAME",
    "LAST_PROCESS_LOG_NAME",
    "LOG_DIRECTORY_NAME",
    "process_log",
]
