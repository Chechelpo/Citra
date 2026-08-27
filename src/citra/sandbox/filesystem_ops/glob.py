from __future__ import annotations

from dataclasses import dataclass
import glob as globlib
from pathlib import Path
from typing import Any

from .base import FilesystemInput, FilesystemOutput, optional_string, require_payload_dict, require_string
from .scope import ScopedFilesystem


@dataclass(frozen=True, slots=True)
class GlobOutput(FilesystemOutput):
    paths: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Any) -> "GlobOutput":
        raw = require_payload_dict(payload)
        paths = raw.get("paths")
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ValueError("Glob output 'paths' must be an array of strings.")
        return cls(paths=tuple(paths))

    def to_payload(self) -> dict[str, Any]:
        return {"paths": list(self.paths)}

    def render(self) -> str:
        return "\n".join(self.paths) or "none"


@dataclass(frozen=True, slots=True)
class GlobInput(FilesystemInput[GlobOutput]):
    operation = "glob"
    output_type = GlobOutput

    pat: str
    path: str = "."

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> "GlobInput":
        return cls(
            pat=require_string(arguments, "pat"),
            path=optional_string(arguments, "path", "."),
        )

    def to_arguments(self) -> dict[str, Any]:
        result = {"pat": self.pat}
        if self.path != ".":
            result["path"] = self.path
        return result


def execute(order: GlobInput, fs: ScopedFilesystem) -> GlobOutput:
    base = fs.require_allowed_path(fs.resolve_path(order.path))
    if not base.is_dir():
        raise NotADirectoryError(
            f"Glob root is not a directory: {fs.display_path(base)}"
        )

    entries: list[Path] = []
    for raw in globlib.glob(str(base / order.pat), recursive=True):
        try:
            entries.append(fs.require_allowed_path(raw))
        except ValueError:
            continue

    entries = sorted(
        set(entries),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    return GlobOutput(paths=tuple(fs.display_path(path) for path in entries))
