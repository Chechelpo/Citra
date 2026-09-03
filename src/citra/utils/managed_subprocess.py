"""Lifecycle-scoped sandboxed subprocess management."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from threading import Lock, Thread
import time
from typing import Any, IO

from citra.sandbox.sandbox import WorkspaceSandbox


@dataclass
class _ProcessRecord:
    """Represent ProcessRecord."""
    id: int
    command: str
    cwd: Path
    network: bool
    process: subprocess.Popen[bytes]
    started_at: float = field(default_factory=time.monotonic)
    output: bytearray = field(default_factory=bytearray)
    lock: Lock = field(default_factory=Lock)


class ManagedSubprocesses:
    """Own long-running commands until stopped or Citra exits."""

    MAX_BUFFER_BYTES = 1_000_000

    def __init__(self, sandbox: WorkspaceSandbox) -> None:
        """Initialize the instance."""
        self._sandbox = sandbox
        self._records: dict[int, _ProcessRecord] = {}
        self._lock = Lock()
        self._next_id = 1
        self._closed = False

    def start(self, command: str, *, cwd: Path, network: bool) -> int:
        """Handle start."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Managed subprocess service is closing.")
        process = self._sandbox.popen(
            ["bash", "--noprofile", "--norc", "-c", command],
            cwd=cwd,
            network=network,
        )
        with self._lock:
            if self._closed:
                self._sandbox.terminate_process(process, force=True)
                raise RuntimeError("Managed subprocess service is closing.")
            process_id = self._next_id
            self._next_id += 1
            record = _ProcessRecord(process_id, command, cwd, network, process)
            self._records[process_id] = record
        self._read_stream(record, process.stdout, "")
        self._read_stream(record, process.stderr, "[stderr] ")
        return process_id

    def poll(self, process_id: int, *, clear: bool = True) -> dict[str, object]:
        """Handle poll."""
        record = self._get(process_id)
        with record.lock:
            output = bytes(record.output).decode("utf-8", errors="replace")
            if clear:
                record.output.clear()
        return {
            "id": process_id,
            "running": record.process.poll() is None,
            "returncode": record.process.poll(),
            "output": output,
        }

    def write(self, process_id: int, text: str) -> None:
        """Handle write."""
        record = self._get(process_id)
        if record.process.poll() is not None or record.process.stdin is None:
            raise RuntimeError(f"Subprocess {process_id} is not running.")
        record.process.stdin.write(text.encode("utf-8"))
        record.process.stdin.flush()

    def stop(self, process_id: int) -> dict[str, object]:
        """Handle stop."""
        record = self._get(process_id)
        self._sandbox.terminate_process(record.process)
        return self.poll(process_id, clear=True)

    def list(self) -> tuple[dict[str, object], ...]:
        """Handle list."""
        with self._lock:
            records = tuple(self._records.values())
        return tuple(
            {
                "id": record.id,
                "command": record.command,
                "cwd": str(record.cwd),
                "network": record.network,
                "running": record.process.poll() is None,
                "returncode": record.process.poll(),
            }
            for record in records
        )

    def close(self, *, force: bool = False) -> None:
        """Handle close."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            records = tuple(self._records.values())
        for record in records:
            self._sandbox.terminate_process(record.process, force=force)

    def _get(self, process_id: int) -> _ProcessRecord:
        """Handle get."""
        with self._lock:
            record = self._records.get(process_id)
        if record is None:
            raise KeyError(f"Unknown subprocess id: {process_id}")
        return record

    def _read_stream(
        self,
        record: _ProcessRecord,
        stream: IO[Any] | None,
        prefix: str,
    ) -> None:
        """Handle read stream."""
        if stream is None:
            return

        def reader() -> None:
            """Handle reader."""
            while True:
                chunk = stream.readline()
                if not chunk:
                    return
                rendered = prefix.encode() + chunk
                with record.lock:
                    record.output.extend(rendered)
                    overflow = len(record.output) - self.MAX_BUFFER_BYTES
                    if overflow > 0:
                        del record.output[:overflow]

        Thread(target=reader, daemon=True).start()
