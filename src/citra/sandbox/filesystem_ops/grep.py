"""Ripgrep-backed content search scoped to the sandbox filesystem."""
from __future__ import annotations

import base64
import fnmatch
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import (
    FilesystemInput,
    FilesystemOutput,
    require_payload_dict,
    require_string,
)
from .scope import ScopedFilesystem


OUTPUT_MODES = ("content", "files_with_matches", "count")

DEFAULT_PATH = "."
DEFAULT_MAX_RESULTS = 100
MAX_MAX_RESULTS = 1000
MAX_LINE_CHARS = 500
MAX_FILE_BYTES = 2 * 1024 * 1024


def _require_optional_string(
    arguments: dict[str, Any],
    name: str,
) -> str | None:
    """Return an optional string argument."""
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{name}' must be a string.")
    return value


def _require_optional_bool(
    arguments: dict[str, Any],
    name: str,
    default: bool = False,
) -> bool:
    """Return an optional boolean argument."""
    value = arguments.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"'{name}' must be a boolean.")
    return value


@dataclass(frozen=True, slots=True)
class GrepMatch:
    """One sorted search hit.

    ``line_number`` is 1-based for ``content`` mode. File-level modes use
    ``line_number == 0``: ``text`` is empty for ``files_with_matches`` and
    holds the decimal per-file count for ``count`` mode.
    """

    path: str
    line_number: int
    text: str

    @classmethod
    def from_payload(cls, payload: Any) -> GrepMatch:
        """Create a match from a worker payload."""
        if not isinstance(payload, dict):
            raise ValueError("Grep match must be an object.")
        path = payload.get("path")
        line_number = payload.get("line_number")
        text = payload.get("text")
        if not isinstance(path, str):
            raise ValueError("Grep match 'path' must be a string.")
        if not isinstance(line_number, int) or line_number < 0:
            raise ValueError("Grep match 'line_number' must be >= 0.")
        if not isinstance(text, str):
            raise ValueError("Grep match 'text' must be a string.")
        return cls(path=path, line_number=line_number, text=text)

    def to_payload(self) -> dict[str, Any]:
        """Convert the match to a worker payload."""
        return {
            "path": self.path,
            "line_number": self.line_number,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class GrepOutput(FilesystemOutput):
    """Sorted search results for one output mode."""

    output_mode: str = "content"
    matches: tuple[GrepMatch, ...] = ()
    truncated: bool = False

    @classmethod
    def from_payload(cls, payload: Any) -> GrepOutput:
        """Create an instance from payload."""
        raw = require_payload_dict(payload)
        mode = raw.get("output_mode", "content")
        if not isinstance(mode, str) or mode not in OUTPUT_MODES:
            raise ValueError("Grep output 'output_mode' is invalid.")
        matches_raw = raw.get("matches", [])
        if not isinstance(matches_raw, list):
            raise ValueError("Grep output 'matches' must be an array.")
        matches = tuple(GrepMatch.from_payload(item) for item in matches_raw)
        truncated = raw.get("truncated", False)
        if not isinstance(truncated, bool):
            raise ValueError("Grep output 'truncated' must be a boolean.")
        return cls(output_mode=mode, matches=matches, truncated=truncated)

    def to_payload(self) -> dict[str, Any]:
        """Convert the value to payload."""
        return {
            "output_mode": self.output_mode,
            "matches": [match.to_payload() for match in self.matches],
            "truncated": self.truncated,
        }

    def render(self) -> str:
        """Render matches in a stable sorted text form."""
        if not self.matches:
            return "no matches"
        if self.output_mode == "files_with_matches":
            text = "\n".join(match.path for match in self.matches)
        elif self.output_mode == "count":
            text = "\n".join(
                f"{match.path}: {match.text}" for match in self.matches
            )
        else:
            text = "\n".join(
                f"{match.path}:{match.line_number}:{match.text}"
                for match in self.matches
            )
        if self.truncated:
            text += "\n... <truncated: showing first results>"
        return text


@dataclass(frozen=True, slots=True)
class GrepInput(FilesystemInput[GrepOutput]):
    """Search file contents with ripgrep, falling back to Python."""

    operation = "grep"
    output_type = GrepOutput

    pattern: str
    path: str = DEFAULT_PATH
    glob: str | None = None
    case_insensitive: bool = False
    max_results: int = DEFAULT_MAX_RESULTS
    output_mode: str = "content"
    literal: bool = False

    def __post_init__(self) -> None:
        """Validate the search request."""
        if not isinstance(self.pattern, str) or not self.pattern:
            raise ValueError("'pattern' cannot be empty.")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("'path' cannot be empty.")
        if self.glob is not None and not isinstance(self.glob, str):
            raise ValueError("'glob' must be a string.")
        if self.glob is not None and not self.glob:
            raise ValueError("'glob' cannot be empty.")
        if not isinstance(self.max_results, int) or isinstance(
            self.max_results, bool
        ):
            raise ValueError("'max_results' must be an integer.")
        if not 1 <= self.max_results <= MAX_MAX_RESULTS:
            raise ValueError(
                f"'max_results' must be between 1 and {MAX_MAX_RESULTS}."
            )
        if self.output_mode not in OUTPUT_MODES:
            raise ValueError(
                "'output_mode' must be one of: "
                + ", ".join(OUTPUT_MODES)
                + "."
            )

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> GrepInput:
        """Parse canonical and harness-alias arguments."""
        pattern = arguments.get("pattern")
        if pattern is None:
            # Gemini/Qwen spellings for the same required regex.
            pattern = arguments.get("regex", arguments.get("query"))
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("'pattern' must be a non-empty string.")

        path = arguments.get("path")
        if path is None:
            path = arguments.get(
                "dir_path", arguments.get("directory", arguments.get("dir", "."))
            )
        if not isinstance(path, str) or not path:
            raise ValueError("'path' must be a non-empty string.")

        glob = arguments.get("glob")
        if glob is None:
            glob = arguments.get("include", arguments.get("file_pattern"))
        if glob is not None and (not isinstance(glob, str) or not glob):
            raise ValueError("'glob' must be a non-empty string.")

        case_insensitive = arguments.get(
            "case_insensitive",
            arguments.get("caseInsensitive", arguments.get("-i", False)),
        )
        if not isinstance(case_insensitive, bool):
            raise ValueError("'case_insensitive' must be a boolean.")

        max_results = arguments.get(
            "max_results",
            arguments.get(
                "head_limit",
                arguments.get("limit", arguments.get("maxResults", 100)),
            ),
        )
        if not isinstance(max_results, int) or isinstance(max_results, bool):
            raise ValueError("'max_results' must be an integer.")

        output_mode = arguments.get(
            "output_mode", arguments.get("mode", "content")
        )
        if not isinstance(output_mode, str):
            raise ValueError("'output_mode' must be a string.")

        literal = arguments.get(
            "literal", arguments.get("fixed_strings", False)
        )
        if not isinstance(literal, bool):
            raise ValueError("'literal' must be a boolean.")

        return cls(
            pattern=pattern,
            path=path,
            glob=glob,
            case_insensitive=case_insensitive,
            max_results=max_results,
            output_mode=output_mode,
            literal=literal,
        )

    def to_arguments(self) -> dict[str, Any]:
        """Convert the value to arguments."""
        result: dict[str, Any] = {"pattern": self.pattern}
        if self.path != DEFAULT_PATH:
            result["path"] = self.path
        if self.glob is not None:
            result["glob"] = self.glob
        if self.case_insensitive:
            result["case_insensitive"] = True
        if self.max_results != DEFAULT_MAX_RESULTS:
            result["max_results"] = self.max_results
        if self.output_mode != "content":
            result["output_mode"] = self.output_mode
        if self.literal:
            result["literal"] = True
        return result


def execute(order: GrepInput, fs: ScopedFilesystem) -> GrepOutput:
    """Search with ``rg`` when available, else with a Python fallback."""
    base = fs.require_allowed_path(fs.resolve_path(order.path))
    if not base.exists():
        raise FileNotFoundError(f"Path not found: {fs.display_path(base)}")

    matches: list[GrepMatch]
    if shutil.which("rg") is not None:
        try:
            matches = _search_with_ripgrep(order, base, fs)
        except (ValueError, OSError):
            raise
        except Exception:
            matches = _search_with_python(order, base, fs)
    else:
        matches = _search_with_python(order, base, fs)

    matches.sort(key=lambda item: (item.path, item.line_number))

    if order.output_mode == "files_with_matches":
        seen: list[GrepMatch] = []
        known: set[str] = set()
        for match in matches:
            if match.path not in known:
                known.add(match.path)
                seen.append(GrepMatch(path=match.path, line_number=0, text=""))
        matches = seen
    elif order.output_mode == "count":
        counts: dict[str, int] = {}
        for match in matches:
            counts[match.path] = counts.get(match.path, 0) + 1
        matches = [
            GrepMatch(path=path, line_number=0, text=str(counts[path]))
            for path in sorted(counts)
        ]

    truncated = len(matches) > order.max_results
    matches = matches[: order.max_results]
    return GrepOutput(
        output_mode=order.output_mode,
        matches=tuple(matches),
        truncated=truncated,
    )


def _search_with_ripgrep(
    order: GrepInput, base: Path, fs: ScopedFilesystem
) -> list[GrepMatch]:
    """Collect content matches through ``rg --json``."""
    try:
        re.compile(order.pattern if not order.literal else re.escape(order.pattern))
    except re.error as error:
        raise ValueError(f"Invalid search pattern: {error}") from error

    command = [
        "rg",
        "--json",
        "--line-number",
        "--no-heading",
        "--color",
        "never",
        "--max-columns",
        "2000",
    ]
    if order.case_insensitive:
        command.append("--ignore-case")
    if order.literal:
        command.append("--fixed-strings")
    if order.glob:
        command.extend(["--glob", order.glob])
    command.extend(["--", order.pattern, str(base)])

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError("ripgrep executable disappeared.") from error

    # Exit 0 means matches, 1 means no matches, anything else is an error
    # such as an invalid pattern (already validated above).
    if completed.returncode not in (0, 1):
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"ripgrep failed: {detail[:500] or 'unknown error'}")

    matches: list[GrepMatch] = []
    # Cap scanning slightly above the request so truncation is detectable
    # without buffering unbounded worker output.
    scan_limit = order.max_results * 4 + 50
    for raw_line in completed.stdout.splitlines():
        if len(matches) >= scan_limit and order.output_mode == "content":
            break
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "match":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        raw_path = data.get("path")
        if not isinstance(raw_path, dict):
            continue
        shown = _decode_rg_text(raw_path.get("text"), raw_path.get("bytes"))
        if shown is None:
            continue
        try:
            candidate = Path(shown)
            if not candidate.is_absolute():
                candidate = base / shown
            allowed = fs.require_allowed_path(candidate.resolve())
        except ValueError:
            continue
        if not allowed.is_file():
            continue
        line_number = data.get("line_number")
        if not isinstance(line_number, int) or line_number < 1:
            continue
        raw_lines = data.get("lines")
        if not isinstance(raw_lines, dict):
            continue
        line_text = _decode_rg_text(
            raw_lines.get("text"), raw_lines.get("bytes")
        )
        if line_text is None:
            continue
        line_text = line_text.rstrip("\r\n")
        if len(line_text) > MAX_LINE_CHARS:
            line_text = line_text[:MAX_LINE_CHARS] + "..."
        matches.append(
            GrepMatch(
                path=fs.display_path(allowed),
                line_number=line_number,
                text=line_text,
            )
        )
    return matches


def _decode_rg_text(text: Any, raw_bytes: Any) -> str | None:
    """Decode one ``rg --json`` text-or-bytes field."""
    if isinstance(text, str):
        return text
    if isinstance(raw_bytes, str):
        try:
            return base64.b64decode(raw_bytes).decode("utf-8", errors="replace")
        except (ValueError, UnicodeDecodeError):
            return None
    return None


def _compile_pattern(order: GrepInput) -> re.Pattern[str]:
    """Compile the model-supplied pattern for the Python fallback."""
    source = re.escape(order.pattern) if order.literal else order.pattern
    flags = re.IGNORECASE if order.case_insensitive else 0
    try:
        return re.compile(source, flags)
    except re.error as error:
        raise ValueError(f"Invalid search pattern: {error}") from error


def _search_with_python(
    order: GrepInput, base: Path, fs: ScopedFilesystem
) -> list[GrepMatch]:
    """Walk and match files with the standard library."""
    expression = _compile_pattern(order)

    if base.is_file():
        candidates = [base]
    elif base.is_dir():
        candidates = _walk_files(base)
    else:
        raise ValueError(f"Path is not a file or directory: {fs.display_path(base)}")

    matches: list[GrepMatch] = []
    scan_limit = order.max_results * 4 + 50
    for candidate in candidates:
        if len(matches) >= scan_limit and order.output_mode == "content":
            break
        try:
            allowed = fs.require_allowed_path(candidate.resolve())
        except ValueError:
            continue
        if order.glob and not _glob_matches(order.glob, allowed, base):
            continue
        if not allowed.is_file():
            continue
        try:
            if allowed.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        try:
            with allowed.open("rb") as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
        except OSError:
            continue
        if len(raw) > MAX_FILE_BYTES or b"\x00" in raw[:8192]:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        display = fs.display_path(allowed)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if expression.search(line):
                shown = line
                if len(shown) > MAX_LINE_CHARS:
                    shown = shown[:MAX_LINE_CHARS] + "..."
                matches.append(
                    GrepMatch(
                        path=display,
                        line_number=line_number,
                        text=shown,
                    )
                )
                if (
                    len(matches) >= scan_limit
                    and order.output_mode == "content"
                ):
                    break
    return matches


def _walk_files(base: Path) -> list[Path]:
    """Return regular files below a directory without following symlinks."""
    results: list[Path] = []
    for root, directories, files in os.walk(base, followlinks=False):
        # Version-control metadata is never content-searchable; other hidden
        # or vendored trees are still visited so ``glob`` can select them.
        if ".git" in directories:
            directories.remove(".git")
        directories.sort()
        for name in sorted(files):
            results.append(Path(root) / name)
    return results


def _glob_matches(pattern: str, path: Path, base: Path) -> bool:
    """Match a file against a user-supplied glob filter."""
    try:
        relative = path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        relative = path.name
    return (
        fnmatch.fnmatchcase(relative, pattern)
        or fnmatch.fnmatchcase(path.name, pattern)
        or Path(relative).match(pattern)
    )
