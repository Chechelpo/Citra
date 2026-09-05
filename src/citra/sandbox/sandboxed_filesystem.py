"""Controller-side client for the fixed-function filesystem worker."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypeVar

from .filesystem_ops import FilesystemInput, FilesystemOutput

from .sandbox import WorkspaceSandbox, SandboxResult


OutputT = TypeVar("OutputT", bound=FilesystemOutput)
logger = logging.getLogger(__name__)


class SandboxedFilesystem:
    """Execute typed scoped filesystem operations inside Bubblewrap."""

    DEFAULT_TIMEOUT_SECONDS = 30

    def __init__(self, sandbox: WorkspaceSandbox) -> None:
        """Bind the worker client to the process-lifetime sandbox."""
        self._sandbox = sandbox
        self._worker_python = (
            sandbox.resolve_command("citra-filesystem-python")
            or sandbox.resolve_command("python")
            or sandbox.resolve_command("python3")
        )
        if self._worker_python is None:
            logger.error(
                "Filesystem worker has no isolated Python runtime",
                extra={"origin": __name__},
            )
            raise RuntimeError(
                "The isolated runtime does not provide Python for filesystem tools."
            )

    def execute(
        self,
        operation: FilesystemInput[OutputT],
        *,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> OutputT:
        """Execute and decode one typed fixed-function worker operation."""
        source_root = Path(__file__).resolve().parents[2]
        payload = json.dumps(
            {
                "operation": operation.operation,
                "arguments": operation.to_arguments(),
            },
            ensure_ascii=False,
        )
        logger.debug(
            "Starting sandboxed filesystem operation '%s'",
            operation.operation,
            extra={"origin": __name__, "operation": operation.operation},
        )
        result: SandboxResult = self._sandbox.run(
            [str(self._worker_python), "-m", "citra.sandbox.filesystem"],
            timeout=timeout,
            network=False,
            input_text=payload,
            environment={
                "PYTHONPATH": str(source_root),
                "PYTHONNOUSERSITE": "1",
                **self._sandbox.filesystem_environment(),
            }
        )
        if result.timed_out:
            logger.error(
                "Sandboxed filesystem operation '%s' timed out after %ss",
                operation.operation,
                timeout,
                extra={"origin": __name__, "operation": operation.operation},
            )
            raise TimeoutError(
                f"Sandboxed filesystem operation timed out after {timeout}s."
            )
        if result.returncode != 0:
            detail = result.output.strip() or "worker exited without output"
            logger.error(
                "Sandboxed filesystem operation '%s' failed with exit code %d",
                operation.operation,
                result.returncode,
                extra={
                    "origin": __name__,
                    "operation": operation.operation,
                    "returncode": result.returncode,
                    "detail": detail[:500],
                },
            )
            raise RuntimeError(
                f"Sandboxed filesystem worker exited with code "
                f"{result.returncode}: {detail}"
            )
        try:
            response = json.loads(result.output)
        except json.JSONDecodeError as error:
            logger.error(
                "Sandboxed filesystem operation '%s' returned invalid JSON",
                operation.operation,
                extra={"origin": __name__, "operation": operation.operation},
            )
            raise RuntimeError(
                "Sandboxed filesystem worker returned invalid JSON: "
                f"{result.output[:500]}"
            ) from error
        if not isinstance(response, dict):
            logger.error(
                "Sandboxed filesystem operation '%s' returned an invalid response",
                operation.operation,
                extra={"origin": __name__, "operation": operation.operation},
            )
            raise RuntimeError(
                "Sandboxed filesystem worker returned an invalid response."
            )
        if not response.get("ok"):
            logger.error(
                "Sandboxed filesystem operation '%s' reported an error: %s",
                operation.operation,
                str(response.get("error") or "filesystem operation failed"),
                extra={"origin": __name__, "operation": operation.operation},
            )
            raise RuntimeError(
                str(response.get("error") or "filesystem operation failed")
            )
        output = operation.parse_output(response.get("result"))
        logger.info(
            "Sandboxed filesystem operation '%s' completed",
            operation.operation,
            extra={"origin": __name__, "operation": operation.operation},
        )
        return output
