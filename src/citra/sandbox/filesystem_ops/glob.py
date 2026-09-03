from __future__ import annotations

from dataclasses import dataclass
import glob as globlib
from pathlib import Path
from typing import Any

from citra.logging import Logger

from .base import (
    FilesystemInput,
    FilesystemOutput,
    optional_string,
    require_payload_dict,
)
from .scope import ScopedFilesystem


_logger = Logger("glob.py")


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

        result = cls(
            pat=pat,
            path=optional_string(
                arguments,
                "path",
                optional_string(arguments, "dir_path", "."),
            ),
        )

        _logger.debug(
            "Parsed glob request",
            pattern=result.pat,
            path=result.path,
        )

        return result

    def to_arguments(self) -> dict[str, Any]:
        """Serialize input into tool arguments."""
        result = {"pat": self.pat}

        if self.path != ".":
            result["path"] = self.path

        return result


def execute(order: GlobInput, fs: ScopedFilesystem) -> GlobOutput:
    """Execute a scoped recursive glob search."""

    _logger.info(
        "Starting glob search",
        pattern=order.pat,
        path=order.path,
    )

    try:
        base = fs.require_allowed_path(
            fs.resolve_path(order.path)
        )
    except Exception as error:
        _logger.error(
            "Failed to resolve glob root",
            path=order.path,
            error=str(error),
        )
        raise

    _logger.trace(
        "Resolved glob root",
        resolved=fs.display_path(base),
    )

    if not base.is_dir():
        _logger.warning(
            "Glob root is not a directory",
            path=fs.display_path(base),
        )

        raise NotADirectoryError(
            f"Glob root is not a directory: {fs.display_path(base)}"
        )

    entries: list[Path] = []

    for raw in globlib.glob(
        str(base / order.pat),
        recursive=True,
    ):
        try:
            entries.append(
                fs.require_allowed_path(raw)
            )
        except ValueError as error:
            _logger.warning(
                "Ignoring glob match outside filesystem scope",
                path=str(raw),
                error=str(error),
            )
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

    _logger.info(
        "Glob search completed",
        pattern=order.pat,
        matches=len(paths),
    )

    return GlobOutput(paths=paths)