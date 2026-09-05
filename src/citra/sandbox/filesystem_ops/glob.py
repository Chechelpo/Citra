from __future__ import annotations

from dataclasses import dataclass
import glob as globlib
from pathlib import Path
from typing import Any

from .base import (
    FilesystemInput,
    FilesystemOutput,
    optional_string,
    require_payload_dict,
)
from .scope import ScopedFilesystem


@dataclass(frozen=True, slots=True)
class GlobOutput(FilesystemOutput):
    """Structured output returned by a glob filesystem operation."""

    paths: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Any) -> "GlobOutput":
        """Deserialize glob output from a tool payload."""
        raw = require_payload_dict(payload)

        paths = raw.get("paths")

        if not isinstance(paths, list) or not all(
            isinstance(path, str) for path in paths
        ):
            raise ValueError("Glob output 'paths' must be an array of strings.")

        return cls(paths=tuple(paths))

    def to_payload(self) -> dict[str, Any]:
        """Serialize glob output into a tool payload."""
        return {"paths": list(self.paths)}

    def render(self) -> str:
        """Render matching paths for model consumption."""
        return "\n".join(self.paths) or "none"


@dataclass(frozen=True, slots=True)
class GlobInput(FilesystemInput[GlobOutput]):
    """Input parameters for a recursive glob search."""

    operation = "glob"
    output_type = GlobOutput

    pat: str
    path: str = "."

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> "GlobInput":
        """Parse glob tool arguments."""
        pat = arguments.get("pat", arguments.get("pattern"))

        if not isinstance(pat, str):
            raise ValueError("'pat' or 'pattern' must be a string.")

        return cls(
            pat=pat,
            path=optional_string(
                arguments,
                "path",
                optional_string(arguments, "dir_path", "."),
            ),
        )

    def to_arguments(self) -> dict[str, Any]:
        """Serialize input into tool arguments."""
        result = {"pat": self.pat}

        if self.path != ".":
            result["path"] = self.path

        return result


def execute(order: GlobInput, fs: ScopedFilesystem) -> GlobOutput:
    """Execute a scoped recursive glob search."""

    base = fs.require_allowed_path(
        fs.resolve_path(order.path)
    )

    if not base.is_dir():
        raise NotADirectoryError(
            f"Glob root is not a directory: {fs.display_path(base)}"
        )

    pattern = str(base / order.pat)

    raw_matches = list(
        globlib.glob(
            pattern,
            recursive=True,
        )
    )

    entries: list[Path] = []

    for raw in raw_matches:
        try:
            entries.append(
                fs.require_allowed_path(raw)
            )
        except ValueError:
            continue

    entries = sorted(
        set(entries),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )

    paths = tuple(
        fs.display_path(path)
        for path in entries
    )

    return GlobOutput(paths=paths)
