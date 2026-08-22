"""Controller-side client for the fixed-function filesystem worker."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from .sandbox import WorkspaceSandbox


class SandboxedFilesystem:
    """Execute scoped reads and writes inside Bubblewrap, never in-process."""

    DEFAULT_TIMEOUT_SECONDS = 30

    def __init__(self, sandbox: WorkspaceSandbox) -> None:
        self._sandbox = sandbox

    def execute(
        self,
        operation: str,
        arguments: dict[str, Any],
        *,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> str:
        source_root = Path(__file__).resolve().parents[2]
        payload = json.dumps(
            {"operation": operation, "arguments": arguments},
            ensure_ascii=False,
        )
        result = self._sandbox.run(
            [sys.executable, "-m", "citra.workers.filesystem"],
            timeout=timeout,
            network=False,
            input_text=payload,
            environment={
                "PYTHONPATH": str(source_root),
                "PYTHONNOUSERSITE": "1",
            },
        )
        if result.timed_out:
            raise TimeoutError(
                f"Sandboxed filesystem operation timed out after {timeout}s."
            )
        if result.returncode != 0:
            detail = result.output.strip() or "worker exited without output"
            raise RuntimeError(
                f"Sandboxed filesystem worker exited with code "
                f"{result.returncode}: {detail}"
            )
        try:
            response = json.loads(result.output)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Sandboxed filesystem worker returned invalid JSON: "
                f"{result.output[:500]}"
            ) from error
        if not isinstance(response, dict):
            raise RuntimeError("Sandboxed filesystem worker returned an invalid response.")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "filesystem operation failed"))
        value = response.get("result")
        if not isinstance(value, str):
            raise RuntimeError("Sandboxed filesystem worker returned a non-text result.")
        return value

