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
        result: dict[str, Any] = {"path": self.path, "offset": self.offset}
        if self.limit is not None:
            result["limit"] = self.limit
        return result


@dataclass(frozen=True, slots=True)
class ReadEntry:
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
        return cls(path=path, content=content, selected=selected)

    def to_payload(self) -> dict[str, Any]:
        return {"path": self.path, "content": self.content, "selected": self.selected}


@dataclass(frozen=True, slots=True)
class ReadOutput(FilesystemOutput):
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
            entries=tuple(ReadEntry.from_payload(entry) for entry in entries),
            single_literal=single_literal,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_payload() for entry in self.entries],
            "single_literal": self.single_literal,
        }

    def render(self) -> str:
        outputs = [f"===== {entry.path} =====\n{entry.content}" for entry in self.entries]
        if (
            self.single_literal
            and len(self.entries) == 1
            and self.entries[0].selected
        ):
            return self.entries[0].content
        return "\n\n".join(outputs)


@dataclass(frozen=True, slots=True)
class ReadInput(FilesystemInput[ReadOutput]):
    operation = "read"
    output_type = ReadOutput

    requests: tuple[ReadSlice, ...]
    single_path: bool = False

    def __post_init__(self) -> None:
        if not self.requests:
            raise ValueError("'requests' must not be empty.")
        if len(self.requests) > MAX_READ_REQUESTS:
            raise ValueError(f"At most {MAX_READ_REQUESTS} read requests are allowed.")
        if self.single_path and len(self.requests) != 1:
            raise ValueError("A single-path read must contain exactly one request.")

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> "ReadInput":
        if not isinstance(arguments, dict):
            raise ValueError("Filesystem arguments must be a JSON object.")

        top_path = arguments.get("path")
        requests = arguments.get("requests")
        if top_path is not None and requests is not None:
            raise ValueError("Use either 'path' or 'requests', not both.")

        if top_path is not None:
            if not isinstance(top_path, str):
                raise ValueError("'path' must be a string.")
            return cls(
                requests=(
                    ReadSlice(
                        path=top_path,
                        offset=arguments.get("offset", 0),
                        limit=arguments.get("limit"),
                    ),
                ),
                single_path=True,
            )

        if requests is None:
            raise ValueError("'path' or 'requests' is required.")
        if arguments.get("offset") is not None:
            raise ValueError("Top-level 'offset' requires top-level 'path'.")
        if arguments.get("limit") is not None:
            raise ValueError("Top-level 'limit' requires top-level 'path'.")
        if not isinstance(requests, list):
            raise ValueError("'requests' must be an array.")

        return cls(
            requests=tuple(
                ReadSlice.parse(request, index=index)
                for index, request in enumerate(requests)
            ),
            single_path=False,
        )

    def to_arguments(self) -> dict[str, Any]:
        if self.single_path:
            request = self.requests[0]
            result: dict[str, Any] = {"path": request.path}
            if request.offset:
                result["offset"] = request.offset
            if request.limit is not None:
                result["limit"] = request.limit
            return result
        return {"requests": [request.to_dict() for request in self.requests]}


def _split_glob(value: str) -> tuple[str, str]:
    path = Path(value)
    static: list[str] = []
    pattern: list[str] = []
    found = False
    for part in path.parts:
        if not found and globlib.has_magic(part):
            found = True
        (pattern if found else static).append(part)
    if not pattern:
        return value, ""
    return (str(Path(*static)) if static else ".", str(Path(*pattern)))


def _parent_listing(fs: ScopedFilesystem, path: Path) -> str:
    parent = path.parent
    if not parent.is_dir():
        return f"Parent directory does not exist: {fs.display_path(parent)}"
    entries = sorted(
        parent.iterdir(),
        key=lambda entry: (not entry.is_dir(), entry.name.casefold()),
    )
    if not entries:
        return f"Parent directory is empty: {fs.display_path(parent)}"
    lines = [f"  {entry.name}{'/' if entry.is_dir() else ''}" for entry in entries]
    return (
        f"Parent directory {fs.display_path(parent)} contains:\n"
        + "\n".join(lines)
    )


def _expand_read_path(fs: ScopedFilesystem, value: str) -> list[Path]:
    if not globlib.has_magic(value):
        resolved = fs.require_allowed_path(fs.resolve_path(value))
        if not resolved.exists():
            raise FileNotFoundError(
                f"File not found: {fs.display_path(resolved)}\n"
                f"{_parent_listing(fs, resolved)}"
            )
        return [resolved]

    base, pattern = _split_glob(value)
    root = fs.require_allowed_path(fs.resolve_path(base))
    if not root.exists():
        raise FileNotFoundError(
            f"Search directory does not exist: {fs.display_path(root)}"
        )
    if not root.is_dir():
        raise NotADirectoryError(
            f"Glob search root is not a directory: {fs.display_path(root)}"
        )

    matches: set[Path] = set()
    for raw in globlib.glob(str(root / pattern), recursive=True):
        try:
            resolved = fs.require_allowed_path(raw)
        except ValueError:
            continue
        if resolved.is_file():
            matches.add(resolved)
    return sorted(matches, key=lambda path: fs.display_path(path).casefold())


def _read_one_file(
    fs: ScopedFilesystem,
    path: Path,
    *,
    offset: int,
    limit: int | None,
) -> str:
    shown = fs.display_path(path)
    if not path.is_file():
        raise IsADirectoryError(f"Path is not a file: {shown}")

    readable = fs.convert_readable(path)
    with readable.open("r", encoding="utf-8", errors="replace") as stream:
        lines = stream.readlines()

    chosen = lines[offset:] if limit is None else lines[offset : offset + limit]
    return "".join(
        f"{offset + index + 1:4}| {line}"
        for index, line in enumerate(chosen)
    )


def execute(order: ReadInput, fs: ScopedFilesystem) -> ReadOutput:
    entries: list[ReadEntry] = []
    seen: set[Path] = set()

    single_literal = order.single_path and not globlib.has_magic(order.requests[0].path)

    for request in order.requests:
        requested = request.path
        try:
            matches = _expand_read_path(fs, requested)
        except Exception as error:
            entries.append(
                ReadEntry(
                    path=requested,
                    content=f"error: {error}",
                    selected=False,
                )
            )
            continue

        if not matches:
            entries.append(
                ReadEntry(
                    path=requested,
                    content=f"error: No files matched pattern: {requested}",
                    selected=False,
                )
            )
            continue

        for path in matches:
            if path in seen:
                continue
            seen.add(path)
            shown = fs.display_path(path)
            try:
                content = _read_one_file(
                    fs,
                    path,
                    offset=request.offset,
                    limit=request.limit,
                )
            except Exception as error:
                content = f"error: {error}"
            entries.append(ReadEntry(path=shown, content=content, selected=True))

    return ReadOutput(entries=tuple(entries), single_literal=single_literal)
