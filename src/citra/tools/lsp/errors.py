"""Typed exception hierarchy for the LSP subsystem.

All exceptions raised by ``citra.tools.lsp`` derive from :class:`LspError`
so callers can catch the entire subsystem with a single ``except`` clause.
"""

from __future__ import annotations

from typing import Any


class LspError(Exception):
    """Base class for every error raised by the LSP subsystem."""


class LspUnavailable(LspError):
    """The configured language-server executable could not be found."""


class LspUnsupportedCapability(LspError):
    """The server does not advertise a capability required by the operation."""


class LspStartupError(LspError):
    """The server process failed to start or returned a non-zero exit early."""


class LspStartupTimeout(LspError):
    """The server did not respond to ``initialize`` within the timeout."""


class LspRequestError(LspError):
    """The server returned a JSON-RPC error response.

    Attributes:
        code:    JSON-RPC error code.
        message: Error message from the server.
        data:    Optional error data payload.
    """

    def __init__(
        self,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.data = data


class LspRequestTimeout(LspError):
    """A request did not complete within the allotted timeout."""


class LspProtocolError(LspError):
    """The server returned a malformed or invalid JSON-RPC message."""


class LspServerExited(LspError):
    """The underlying language-server process exited unexpectedly.

    Attributes:
        exit_code:   Process exit code (``None`` if not available).
        stderr_tail: Last lines of stderr (may be empty).
    """

    def __init__(
        self,
        exit_code: int | None = None,
        stderr_tail: str = "",
    ) -> None:
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail
        detail = f"Language server exited unexpectedly (code {exit_code})."
        if stderr_tail:
            detail += f"\n\nLast stderr:\n{stderr_tail}"
        super().__init__(detail)


class LspDocumentError(LspError):
    """An unknown document was referenced or a version mismatch occurred."""


class LspDiagnosticsTimeout(LspError):
    """Diagnostics could not be obtained within the configured timeout."""


class LspWorkspaceEditError(LspError):
    """A ``WorkspaceEdit`` could not be parsed or applied."""


class LspTransportError(LspError):
    """Low-level transport failure (framing, I/O)."""
