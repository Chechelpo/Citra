from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import FilesystemInput, FilesystemOutput, require_payload_dict, require_string
from .scope import ScopedFilesystem


@dataclass(frozen=True, slots=True)
class WriteOutput(FilesystemOutput):
    """Represent WriteOutput."""
    status: str = "ok"

    @classmethod
    def from_payload(cls, payload: Any) -> "WriteOutput":
        """Create an instance from payload."""
        raw = require_payload_dict(payload)
        status = raw.get("status")
        if not isinstance(status, str):
            raise ValueError("Write output 'status' must be a string.")
        return cls(status=status)

    def to_payload(self) -> dict[str, Any]:
        """Convert the value to payload."""
        return {"status": self.status}

    def render(self) -> str:
        """Handle render."""
        return self.status


@dataclass(frozen=True, slots=True)
class WriteInput(FilesystemInput[WriteOutput]):
    """Represent WriteInput."""
    operation = "write"
    output_type = WriteOutput

    path: str
    content: str

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> "WriteInput":
        """Handle parse."""
        return cls(
            path=require_string(arguments, "path"),
            content=require_string(arguments, "content"),
        )

    def to_arguments(self) -> dict[str, Any]:
        """Convert the value to arguments."""
        return {"path": self.path, "content": self.content}


def execute(order: WriteInput, fs: ScopedFilesystem) -> WriteOutput:
    """Execute the execute operation."""
    fs.write_text_atomic(order.path, order.content)
    return WriteOutput()
