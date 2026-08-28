"""Controller-side client for the fixed-function filesystem worker."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import TypeVar

from .filesystem_ops import FilesystemInput, FilesystemOutput

from .sandbox import WorkspaceSandbox


OutputT = TypeVar("OutputT", bound=FilesystemOutput)


class SandboxedFilesystem:
    """Execute typed scoped filesystem operations inside Bubblewrap."""

    DEFAULT_TIMEOUT_SECONDS = 30

    def __init__(self, sandbox: WorkspaceSandbox) -> None:
        self._sandbox = sandbox

    def execute(
        self,
        operation: FilesystemInput[OutputT],
        *,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> OutputT:
        source_root = Path(__file__).resolve().parents[2]
        payload = json.dumps(
            {
                "operation": operation.operation,
                "arguments": operation.to_arguments(),
            },
            ensure_ascii=False,
        )
        result = self._sandbox.run(
            [sys.executable, "-m", "citra.sandbox.filesystem"],
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
            raise RuntimeError(
                "Sandboxed filesystem worker returned an invalid response."
            )
        if not response.get("ok"):
            raise RuntimeError(
                str(response.get("error") or "filesystem operation failed")
            )
        return operation.parse_output(response.get("result"))
