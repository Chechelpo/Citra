"""JSON-RPC ownership for the lifecycle-scoped browser worker."""

from __future__ import annotations

import json
from pathlib import Path
import selectors
import subprocess
import sys
from threading import Lock, Thread
from typing import Any

from .sandbox import WorkspaceSandbox


class BrowserManager:
    def __init__(
        self,
        sandbox: WorkspaceSandbox,
        workspace: Path,
        *,
        request_timeout: float,
    ) -> None:
        self._sandbox = sandbox
        self._workspace = workspace
        self._request_timeout = request_timeout
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = Lock()
        self._stderr = bytearray()

    def request(self, action: str, **arguments: Any) -> dict[str, Any]:
        with self._lock:
            process = self._ensure_process()
            assert process.stdin is not None
            assert process.stdout is not None
            payload = json.dumps({"action": action, **arguments}) + "\n"
            process.stdin.write(payload.encode("utf-8"))
            process.stdin.flush()

            selector = selectors.DefaultSelector()
            try:
                selector.register(process.stdout, selectors.EVENT_READ)
                if not selector.select(self._request_timeout):
                    raise TimeoutError(
                        f"Browser worker timed out during '{action}'."
                    )
                line = process.stdout.readline()
            finally:
                selector.close()

            if not line:
                stderr = bytes(self._stderr).decode("utf-8", errors="replace")
                raise RuntimeError(
                    "Browser worker exited unexpectedly. " + stderr[-4000:]
                )
            response = json.loads(line)
            if not response.get("ok", False):
                raise RuntimeError(str(response.get("error", "browser error")))
            return response

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
        if process is None:
            return
        self._sandbox.terminate_process(process)

    def _ensure_process(self) -> subprocess.Popen[bytes]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._process = self._sandbox.popen(
            [sys.executable, "-m", "citra.utils.browser_worker"],
            cwd=self._workspace,
            network=True,
            environment={"PYTHONUNBUFFERED": "1"},
        )
        self._stderr.clear()
        if self._process.stderr is not None:
            stderr = self._process.stderr

            def drain() -> None:
                while True:
                    chunk = stderr.read(4096)
                    if not chunk:
                        return
                    self._stderr.extend(chunk)
                    overflow = len(self._stderr) - 100_000
                    if overflow > 0:
                        del self._stderr[:overflow]

            Thread(target=drain, daemon=True).start()
        return self._process
