"""LSP protocol helpers: URI/path conversion and JSON-RPC primitives.

Only the subset of JSON-RPC 2.0 and LSP needed by this subsystem is
implemented.  Everything is plain ``dict`` / ``list`` to stay stdlib-only.
"""

from __future__ import annotations

from pathlib import Path
import urllib.parse
from typing import Any

from .errors import LspProtocolError

# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------


def path_to_uri(path: str | Path) -> str:
    """Convert a filesystem path to a ``file://`` URI."""
    resolved = Path(path).resolve()
    # Path.resolve() collapses away any trailing slash; urllib quoting
    # below preserves directory semantics by operating on parts.
    parts = resolved.parts
    # On POSIX the first element is "/".
    drive = resolved.drive  # empty on POSIX
    if drive:
        # Windows-style path.  Build ``file:///drive:/rest``.
        path_str = resolved.as_posix()
        return "file:///" + urllib.parse.quote(path_str, safe="/:")
    # POSIX: ``file:///`` + absolute path with each component quoted.
    quoted = "/".join(
        urllib.parse.quote(part, safe="")
        for part in parts[1:]
    )
    return "file:///" + quoted


def uri_to_path(uri: str) -> Path:
    """Convert a ``file://`` URI back to a local :class:`~pathlib.Path`."""
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "file":
        raise LspProtocolError(f"Not a file URI: {uri}")
    # urlsplit / urlparse keeps the leading slash in ``path`` on POSIX.
    path = urllib.parse.unquote(parsed.path)
    # On Windows the path may start with ``/C:/...``; strip the leading
    # slash before a drive letter.
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


# ---------------------------------------------------------------------------
# JSON-RPC message construction
# ---------------------------------------------------------------------------


def make_request(
    id: int | str,
    method: str,
    params: dict[str, Any] | list[Any] | None = None,
) -> dict[str, Any]:
    """Handle make request."""
    message: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
    }
    if params is not None:
        message["params"] = params
    return message


def make_notification(
    method: str,
    params: dict[str, Any] | list[Any] | None = None,
) -> dict[str, Any]:
    """Handle make notification."""
    message: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        message["params"] = params
    return message


def make_response(
    id: int | str | None,
    result: Any,
) -> dict[str, Any]:
    """Handle make response."""
    return {
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    }


def make_error_response(
    id: int | str | None,
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    """Handle make error response."""
    error: dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if data is not None:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": id,
        "error": error,
    }


def is_response(message: dict[str, Any]) -> bool:
    """Return whether is response."""
    return "id" in message and ("result" in message or "error" in message)


def is_request(message: dict[str, Any]) -> bool:
    """Return whether is request."""
    return "id" in message and "method" in message


def is_notification(message: dict[str, Any]) -> bool:
    """Return whether is notification."""
    return "id" not in message and "method" in message
