from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import FilesystemInput, FilesystemOutput, require_payload_dict
from .scope import ScopedFilesystem


# A generous hard cap that keeps the JSON-over-stdio worker responsive and
# fits comfortably within Citra's tool-result token budgets. Vision-capable
# models are typically happy with a few MB; anything larger is better
# processed out-of-band by the model provider.
DEFAULT_MAX_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ReadBinaryOutput(FilesystemOutput):
    """Worker response for a binary read.

    The payload carries a base64 string rather than raw bytes because the
    sandbox worker protocol is JSON over stdio. The tool layer is responsible
    for producing a model-facing data URL from this content.
    """

    content_b64: str
    size: int
    mime_type: str | None

    @classmethod
    def from_payload(cls, payload: Any) -> ReadBinaryOutput:
        """Create an instance from payload."""
        raw = require_payload_dict(payload)
        content_b64 = raw.get("content_b64")
        size = raw.get("size")
        mime_type = raw.get("mime_type")
        if not isinstance(content_b64, str):
            raise ValueError(
                "ReadBinary output 'content_b64' must be a string."
            )
        if not isinstance(size, int) or size < 0:
            raise ValueError(
                "ReadBinary output 'size' must be a non-negative integer."
            )
        if mime_type is not None and not isinstance(mime_type, str):
            raise ValueError(
                "ReadBinary output 'mime_type' must be a string or null."
            )
        return cls(
            content_b64=content_b64,
            size=size,
            mime_type=mime_type,
        )

    def to_payload(self) -> dict[str, Any]:
        """Convert the value to payload."""
        return {
            "content_b64": self.content_b64,
            "size": self.size,
            "mime_type": self.mime_type,
        }

    def render(self) -> str:
        # ReadBinary carries a model-facing binary payload, not text. The
        # legacy string view is intentionally short: callers should consume
        # the structured fields, not this rendering.
        """Handle render."""
        return (
            f"binary content: {self.size} bytes, "
            f"mime_type={self.mime_type or 'application/octet-stream'}"
        )


@dataclass(frozen=True, slots=True)
class ReadBinaryInput(FilesystemInput[ReadBinaryOutput]):
    """Represent ReadBinaryInput."""
    operation = "read_binary"
    output_type = ReadBinaryOutput

    path: str
    max_bytes: int = DEFAULT_MAX_BYTES

    def __post_init__(self) -> None:
        """Validate and initialize the instance after construction."""
        if not isinstance(self.max_bytes, int) or self.max_bytes <= 0:
            raise ValueError("'max_bytes' must be a positive integer.")

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> ReadBinaryInput:
        """Handle parse."""
        path = arguments.get("path")
        if not isinstance(path, str):
            raise ValueError("'path' must be a string.")
        raw_max = arguments.get("max_bytes", DEFAULT_MAX_BYTES)
        if not isinstance(raw_max, int) or raw_max <= 0:
            raise ValueError("'max_bytes' must be a positive integer.")
        return cls(
            path=path,
            max_bytes=raw_max,
        )

    def to_arguments(self) -> dict[str, Any]:
        """Convert the value to arguments."""
        return {
            "path": self.path,
            "max_bytes": self.max_bytes,
        }


def _guess_mime_type(path: Path) -> str | None:
    """Handle guess mime type."""
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type


def execute(order: ReadBinaryInput, fs: ScopedFilesystem) -> ReadBinaryOutput:
    """Execute the execute operation."""
    resolved = fs.require_allowed_path(fs.resolve_path(order.path))
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {fs.display_path(resolved)}")

    size = resolved.stat().st_size
    if size > order.max_bytes:
        raise ValueError(
            f"File too large to read as binary: {size} bytes "
            f"(limit: {order.max_bytes} bytes)."
        )

    with resolved.open("rb") as stream:
        raw_bytes = stream.read()

    return ReadBinaryOutput(
        content_b64=base64.b64encode(raw_bytes).decode("ascii"),
        size=len(raw_bytes),
        mime_type=_guess_mime_type(resolved),
    )
