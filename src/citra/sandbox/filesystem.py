"""Sandbox-side entrypoint for Citra's typed filesystem operations.

The controller sends one JSON request on stdin. This worker performs the actual
filesystem operation after Bubblewrap has installed the mount policy, then
returns one structured JSON response on stdout. No model-supplied Python or
shell source is evaluated here.
"""

from __future__ import annotations

import json
import sys

from .filesystem_ops.registry import OPERATIONS
from .filesystem_ops.scope import ScopedFilesystem


MAX_REQUEST_BYTES = 16 * 1024 * 1024


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("Filesystem request is too large.")

        request = json.loads(raw.decode("utf-8"))
        if not isinstance(request, dict):
            raise ValueError("Filesystem request must be a JSON object.")

        operation_name = request.get("operation")
        if not isinstance(operation_name, str):
            raise ValueError("'operation' must be a string.")

        try:
            operation = OPERATIONS[operation_name]
        except KeyError as error:
            raise ValueError(
                f"Unsupported filesystem operation: {operation_name!r}"
            ) from error

        arguments = request.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("Filesystem arguments must be a JSON object.")

        order = operation.input_type.parse(arguments)
        output = operation.execute(order, ScopedFilesystem())
        response = {"ok": True, "result": output.to_payload()}
    except Exception as error:
        response = {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }

    sys.stdout.write(json.dumps(response, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
