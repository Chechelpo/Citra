from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import FilesystemInput, FilesystemOutput, require_payload_dict, require_string
from .scope import ScopedFilesystem


@dataclass(frozen=True, slots=True)
class ReadRawOutput(FilesystemOutput):
    content: str

    @classmethod
    def from_payload(cls, payload: Any) -> "ReadRawOutput":
        raw = require_payload_dict(payload)
        content = raw.get("content")
        if not isinstance(content, str):
            raise ValueError("ReadRaw output 'content' must be a string.")
        return cls(content=content)

    def to_payload(self) -> dict[str, Any]:
        return {"content": self.content}

    def render(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True)
class ReadRawInput(FilesystemInput[ReadRawOutput]):
    operation = "read_raw"
    output_type = ReadRawOutput

    path: str

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> "ReadRawInput":
        return cls(path=require_string(arguments, "path"))

    def to_arguments(self) -> dict[str, Any]:
        return {"path": self.path}


def execute(order: ReadRawInput, fs: ScopedFilesystem) -> ReadRawOutput:
    path = fs.require_allowed_path(fs.resolve_path(order.path))
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {fs.display_path(path)}")
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        return ReadRawOutput(content=stream.read())
