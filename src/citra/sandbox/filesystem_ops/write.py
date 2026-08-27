from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import FilesystemInput, FilesystemOutput, require_payload_dict, require_string
from .scope import ScopedFilesystem


@dataclass(frozen=True, slots=True)
class WriteOutput(FilesystemOutput):
    status: str = "ok"

    @classmethod
    def from_payload(cls, payload: Any) -> "WriteOutput":
        raw = require_payload_dict(payload)
        status = raw.get("status")
        if not isinstance(status, str):
            raise ValueError("Write output 'status' must be a string.")
        return cls(status=status)

    def to_payload(self) -> dict[str, Any]:
        return {"status": self.status}

    def render(self) -> str:
        return self.status


@dataclass(frozen=True, slots=True)
class WriteInput(FilesystemInput[WriteOutput]):
    operation = "write"
    output_type = WriteOutput

    path: str
    content: str

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> "WriteInput":
        return cls(
            path=require_string(arguments, "path"),
            content=require_string(arguments, "content"),
        )

    def to_arguments(self) -> dict[str, Any]:
        return {"path": self.path, "content": self.content}


def execute(order: WriteInput, fs: ScopedFilesystem) -> WriteOutput:
    fs.write_text_atomic(order.path, order.content)
    return WriteOutput()
