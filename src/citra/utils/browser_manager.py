"""JSON-RPC ownership for the lifecycle-scoped browser worker."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import selectors
import subprocess
import os
from threading import Lock, Thread
from typing import Any

from citra.sandbox.sandbox import WorkspaceSandbox


logger = logging.getLogger(__name__)


class BrowserManager:
    """Represent BrowserManager."""
    def __init__(
        self,
        sandbox: WorkspaceSandbox,
        workspace: Path,
        *,
        request_timeout: float,
        browsers_path: str | Path | None = None,
    ) -> None:
        """Initialize the instance."""
        self._sandbox = sandbox
        self._workspace = workspace
        self._request_timeout = request_timeout
        self._browsers_path = self._resolve_browsers_path(
            browsers_path
        )
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = Lock()
        self._stderr = bytearray()
        self._closed = False

    def request(self, action: str, **arguments: Any) -> dict[str, Any]:
        """Handle request."""
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

    def close(self, *, force: bool = False) -> None:
        """Handle close."""
        with self._lock:
            self._closed = True
            process = self._process
            self._process = None
        if process is None:
            return
        logger.info(
            "Closing sandboxed browser worker",
            extra={"origin": __name__, "force": force},
        )
        self._sandbox.terminate_process(process, force=force)

    def _ensure_process(self) -> subprocess.Popen[bytes]:
        """Handle ensure process."""
        if self._closed:
            raise RuntimeError("The browser manager is closing.")
        if self._process is not None and self._process.poll() is None:
            return self._process

        python = (
            self._sandbox.resolve_command("python")
            or self._sandbox.resolve_command("python3")
        )
        if python is None:
            logger.error(
                "Browser worker has no isolated Python runtime",
                extra={"origin": __name__},
            )
            raise RuntimeError(
                "The isolated runtime does not provide Python for browser tools."
            )
        logger.debug(
            "Starting sandboxed browser worker",
            extra={"origin": __name__, "python": str(python)},
        )
        self._process = self._sandbox.popen(
            [str(python), "-m", "citra.utils.browser_worker"],
            cwd=self._workspace,
            network=True,
            environment={
                "PYTHONUNBUFFERED": "1",
                "PLAYWRIGHT_BROWSERS_PATH": str(
                    self._browsers_path
                ),
            },
        )

        self._stderr.clear()

        if self._process.stderr is not None:
            stderr = self._process.stderr

            def drain() -> None:
                """Handle drain."""
                while True:
                    chunk = stderr.read(4096)

                    if not chunk:
                        return

                    self._stderr.extend(chunk)
                    overflow = len(self._stderr) - 100_000

                    if overflow > 0:
                        del self._stderr[:overflow]

            Thread(
                target=drain,
                daemon=True,
            ).start()

        return self._process
    
    @staticmethod
    def _resolve_browsers_path(
        configured: str | Path | None,
    ) -> Path:
        """Handle resolve browsers path."""
        if configured is None:
            configured = os.environ.get(
                "PLAYWRIGHT_BROWSERS_PATH"
            )

        if configured is None:
            path = Path.home() / ".cache" / "ms-playwright"
        else:
            path = Path(configured).expanduser()

        path = path.absolute()

        return path
