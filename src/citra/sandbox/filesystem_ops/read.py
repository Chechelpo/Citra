from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import FilesystemInput, FilesystemOutput, require_payload_dict
from .scope import ScopedFilesystem


MAX_READ_PATHS = 20


@dataclass(frozen=True, slots=True)
class ReadEntry:
    """One resolved file and its read content."""

    path: str
    content: str

    @classmethod
    def from_payload(cls, payload: Any) -> "ReadEntry":
        """Create an instance from payload."""
        if not isinstance(payload, dict):
            raise ValueError("Read output entry must be a JSON object.")

        path = payload.get("path")
        content = payload.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            raise ValueError("Read output entry has invalid text fields.")

        return cls(
            path=path,
            content=content,
        )

    def to_payload(self) -> dict[str, Any]:
        """Convert the value to payload."""
        return {
            "path": self.path,
            "content": self.content,
        }


@dataclass(frozen=True, slots=True)
class ReadOutput(FilesystemOutput):
    """Structured result of a filesystem read operation."""

    entries: tuple[ReadEntry, ...]
    single_path: bool

    @classmethod
    def from_payload(cls, payload: Any) -> "ReadOutput":
        """Create an instance from payload."""
        raw = require_payload_dict(payload)

        entries = raw.get("entries")
        single_path = raw.get("single_path")

        if not isinstance(entries, list):
            raise ValueError("Read output 'entries' must be an array.")

        if not isinstance(single_path, bool):
            raise ValueError("Read output 'single_path' must be boolean.")

        return cls(
            entries=tuple(
                ReadEntry.from_payload(entry)
                for entry in entries
            ),
            single_path=single_path,
        )

    def to_payload(self) -> dict[str, Any]:
        """Convert the value to payload."""
        return {
            "entries": [
                entry.to_payload()
                for entry in self.entries
            ],
            "single_path": self.single_path,
        }

    def render(self) -> str:
        """Handle render."""
        outputs = [
            f"===== {entry.path} =====\n{entry.content}"
            for entry in self.entries
        ]

        if (
            self.single_path
            and len(self.entries) == 1
        ):
            return self.entries[0].content

        return "\n\n".join(outputs)


@dataclass(frozen=True, slots=True)
class ReadInput(FilesystemInput[ReadOutput]):
    """Input schema for reading one or more literal scoped file paths."""

    operation = "read"
    output_type = ReadOutput

    paths: tuple[str, ...]
    single_path: bool = False

    def __post_init__(self) -> None:
        """Validate literal paths, batch cardinality, and shape consistency."""
        if not self.paths:
            raise ValueError("'paths' must not be empty.")

        if len(self.paths) > MAX_READ_PATHS:
            raise ValueError(
                f"At most {MAX_READ_PATHS} paths may be read at once."
            )

        if self.single_path and len(self.paths) != 1:
            raise ValueError(
                "A single-path read must contain exactly one path."
            )

        for index, path in enumerate(self.paths):
            label = "path" if self.single_path else f"paths[{index}]"
            if not isinstance(path, str) or not path:
                raise ValueError(f"'{label}' must be a non-empty string.")
            if any(character in path for character in ("*", "?", "[")):
                raise ValueError(
                    f"'{label}' must be a literal file path, not a glob pattern."
                )

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> "ReadInput":
        """Parse exactly one literal path or a batch of literal paths."""
        if not isinstance(arguments, dict):
            raise ValueError("Filesystem arguments must be a JSON object.")

        unknown = set(arguments) - {"path", "paths"}
        if unknown:
            raise ValueError(
                "Unsupported read arguments: " + ", ".join(sorted(unknown))
            )

        has_path = "path" in arguments
        has_paths = "paths" in arguments
        if has_path == has_paths:
            raise ValueError("Provide exactly one of 'path' or 'paths'.")

        if has_path:
            return cls(paths=(arguments["path"],), single_path=True)

        raw_paths = arguments["paths"]
        if not isinstance(raw_paths, list):
            raise ValueError("'paths' must be an array.")
        return cls(paths=tuple(raw_paths), single_path=False)

    def to_arguments(self) -> dict[str, Any]:
        """Serialize the normalized read request for the worker protocol."""
        if self.single_path:
            return {"path": self.paths[0]}
        return {"paths": list(self.paths)}


def execute(order: ReadInput, fs: ScopedFilesystem) -> ReadOutput:
    """Read scoped literal paths without following directories."""
    entries: list[ReadEntry] = []
    for raw_path in order.paths:
        path = fs.require_allowed_path(fs.resolve_path(raw_path))
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {fs.display_path(path)}")
        entries.append(
            ReadEntry(
                path=fs.display_path(path),
                content=path.read_text(encoding="utf-8", errors="strict"),
            )
        )
    return ReadOutput(entries=tuple(entries), single_path=order.single_path)
