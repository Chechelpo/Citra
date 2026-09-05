from __future__ import annotations

from dataclasses import dataclass
import glob as globlib
from pathlib import Path
from typing import Any

from .base import FilesystemInput, FilesystemOutput, require_payload_dict
from .scope import ScopedFilesystem


MAX_READ_REQUESTS = 20


@dataclass(frozen=True, slots=True)
class ReadSlice:
    """A single file read request with optional line range selection."""

    path: str
    offset: int = 0
    limit: int | None = None

    def __post_init__(self) -> None:
        """Validate and initialize the instance after construction."""
        if not isinstance(self.offset, int) or self.offset < 0:
            raise ValueError("'offset' must be a non-negative integer.")
        if self.limit is not None and (
            not isinstance(self.limit, int) or self.limit < 0
        ):
            raise ValueError("'limit' must be a non-negative integer.")

    @classmethod
    def parse(cls, value: Any, *, index: int | None = None) -> "ReadSlice":
        """Handle parse."""
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
        """Convert the value to dict."""
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
        """Create an instance from payload."""
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
        """Convert the value to payload."""
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
        """Create an instance from payload."""
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
        """Convert the value to payload."""
        return {
            "entries": [
                entry.to_payload()
                for entry in self.entries
            ],
            "single_literal": self.single_literal,
        }

    def render(self) -> str:
        """Handle render."""
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
        """Validate batch cardinality and single-path consistency."""
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

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> "ReadInput":
        """Parse either one top-level path or a batch of read requests."""
        if not isinstance(arguments, dict):
            raise ValueError("Filesystem arguments must be a JSON object.")
        has_path = "path" in arguments
        has_requests = "requests" in arguments
        if has_path == has_requests:
            raise ValueError("Provide exactly one of 'path' or 'requests'.")
        if has_path:
            request = ReadSlice.parse(arguments)
            return cls(requests=(request,), single_path=True)
        raw_requests = arguments.get("requests")
        if not isinstance(raw_requests, list):
            raise ValueError("'requests' must be an array.")
        return cls(
            requests=tuple(
                ReadSlice.parse(request, index=index)
                for index, request in enumerate(raw_requests)
            ),
            single_path=False,
        )

    def to_arguments(self) -> dict[str, Any]:
        """Serialize the normalized read request for the worker protocol."""
        if self.single_path:
            return self.requests[0].to_dict()
        return {"requests": [request.to_dict() for request in self.requests]}


def execute(order: ReadInput, fs: ScopedFilesystem) -> ReadOutput:
    """Read scoped literal paths or glob matches without following directories."""
    entries: list[ReadEntry] = []
    single_literal = order.single_path and not globlib.has_magic(order.requests[0].path)
    for request in order.requests:
        resolved_pattern = fs.resolve_path(request.path)
        if globlib.has_magic(request.path):
            matches = tuple(
                Path(value)
                for value in sorted(
                    globlib.glob(str(resolved_pattern), recursive=True)
                )
            )
        else:
            matches = (resolved_pattern,)
        for path in matches:
            allowed = fs.require_allowed_path(path)
            if not allowed.is_file():
                if single_literal:
                    raise FileNotFoundError(
                        f"File not found: {fs.display_path(allowed)}"
                    )
                continue
            text = allowed.read_text(encoding="utf-8", errors="strict")
            selected = request.offset != 0 or request.limit is not None
            if selected:
                lines = text.splitlines(keepends=True)
                stop = (
                    None
                    if request.limit is None
                    else request.offset + request.limit
                )
                text = "".join(lines[request.offset:stop])
            entries.append(
                ReadEntry(
                    path=fs.display_path(allowed),
                    content=text,
                    selected=selected,
                )
            )
    return ReadOutput(entries=tuple(entries), single_literal=single_literal)
