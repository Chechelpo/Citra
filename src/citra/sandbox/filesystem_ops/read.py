from __future__ import annotations

from dataclasses import dataclass
import glob as globlib
from pathlib import Path
from typing import Any

from citra.logging import Logger

from .base import FilesystemInput, FilesystemOutput, require_payload_dict
from .scope import ScopedFilesystem


_logger = Logger("read.py")


MAX_READ_REQUESTS = 20


@dataclass(frozen=True, slots=True)
class ReadSlice:
    """A single file read request with optional line range selection."""

    path: str
    offset: int = 0
    limit: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.offset, int) or self.offset < 0:
            raise ValueError("'offset' must be a non-negative integer.")
        if self.limit is not None and (
            not isinstance(self.limit, int) or self.limit < 0
        ):
            raise ValueError("'limit' must be a non-negative integer.")

    @classmethod
    def parse(cls, value: Any, *, index: int | None = None) -> "ReadSlice":
        label = "request" if index is None else f"requests[{index}]"

        if not isinstance(value, dict):
            raise ValueError(f"'{label}' must be a JSON object.")

        if "path" not in value:
            raise ValueError(f"'{label}.path' is required.")

        path = value["path"]

        if not isinstance(path, str):
            raise ValueError(f"'{label}.path' must be a string.")

        return cls(
            path=path,
            offset=value.get("offset", 0),
            limit=value.get("limit"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": self.path,
            "offset": self.offset,
        }

        if self.limit is not None:
            result["limit"] = self.limit

        return result


@dataclass(frozen=True, slots=True)
class ReadEntry:
    """One resolved file and its read content."""

    path: str
    content: str
    selected: bool

    @classmethod
    def from_payload(cls, payload: Any) -> "ReadEntry":
        if not isinstance(payload, dict):
            raise ValueError("Read output entry must be a JSON object.")

        path = payload.get("path")
        content = payload.get("content")
        selected = payload.get("selected")

        if not isinstance(path, str) or not isinstance(content, str):
            raise ValueError("Read output entry has invalid text fields.")

        if not isinstance(selected, bool):
            raise ValueError("Read output entry 'selected' must be boolean.")

        return cls(
            path=path,
            content=content,
            selected=selected,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "selected": self.selected,
        }


@dataclass(frozen=True, slots=True)
class ReadOutput(FilesystemOutput):
    """Structured result of a filesystem read operation."""

    entries: tuple[ReadEntry, ...]
    single_literal: bool

    @classmethod
    def from_payload(cls, payload: Any) -> "ReadOutput":
        raw = require_payload_dict(payload)

        entries = raw.get("entries")
        single_literal = raw.get("single_literal")

        if not isinstance(entries, list):
            raise ValueError("Read output 'entries' must be an array.")

        if not isinstance(single_literal, bool):
            raise ValueError("Read output 'single_literal' must be boolean.")

        return cls(
            entries=tuple(
                ReadEntry.from_payload(entry)
                for entry in entries
            ),
            single_literal=single_literal,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "entries": [
                entry.to_payload()
                for entry in self.entries
            ],
            "single_literal": self.single_literal,
        }

    def render(self) -> str:
        outputs = [
            f"===== {entry.path} =====\n{entry.content}"
            for entry in self.entries
        ]

        if (
            self.single_literal
            and len(self.entries) == 1
            and self.entries[0].selected
        ):
            return self.entries[0].content

        return "\n\n".join(outputs)


@dataclass(frozen=True, slots=True)
class ReadInput(FilesystemInput[ReadOutput]):
    """Input schema for reading one or more scoped files."""

    operation = "read"
    output_type = ReadOutput

    requests: tuple[ReadSlice, ...]
    single_path: bool = False

    def __post_init__(self) -> None:
        if not self.requests:
            raise ValueError("'requests' must not be empty.")

        if len(self.requests) > MAX_READ_REQUESTS:
            raise ValueError(
                f"At most {MAX_READ_REQUESTS} read requests are allowed."
            )

        if self.single_path and len(self.requests) != 1:
            raise ValueError(
                "A single-path read must contain exactly one request."
            )