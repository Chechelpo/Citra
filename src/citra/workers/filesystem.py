"""Sandbox-side implementation of Citra's scoped filesystem operations.

The controller sends one JSON request on stdin. This process performs the
actual filesystem traversal/read/write after Bubblewrap has installed the
mount policy, then returns one JSON response on stdout. No model-supplied
Python or shell source is evaluated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import glob as globlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, cast


MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_CHARS = 2_000_000
MAX_READ_FILES = 20
MAX_READ_REQUESTS = 20
MAX_PARENT_ENTRIES = 50
MAX_GREP_RESULTS = 50

DEFAULT_TREE_DEPTH = 3
MAX_TREE_DEPTH = 20
DEFAULT_TREE_LIMIT = 200
MAX_TREE_LIMIT = 5000
DEFAULT_TREE_SKIPS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".idea",
        ".vscode",
        "dist",
        "build",
        "coverage",
        ".coverage",
    }
)

_ALIAS = re.compile(r"^@([a-z_]+)(?:/(.*))?$")


class ScopedFilesystem:
    """Path resolver whose authority is limited to sandbox data mounts."""

    def __init__(self) -> None:
        self.workspace = self._required_path("CITRA_WORKSPACE")
        self.source_workspace = self._required_path("CITRA_SOURCE")
        self.home = self._required_path("HOME")
        self.tmp = self._required_path("CITRA_TMP")
        self.cache = self._required_path("CITRA_CACHE")
        env_raw = os.environ.get("CITRA_ENV")
        self.env = (
            Path(env_raw).resolve()
            if env_raw
            else self.workspace.parent / "env"
        )
        self.config = self._required_path("XDG_CONFIG_HOME")
        self.data = self._required_path("XDG_DATA_HOME")
        runtime_raw = os.environ.get("CITRA_RUNTIME")
        self.runtime = (
            Path(runtime_raw).resolve()
            if runtime_raw
            else self._required_path("XDG_RUNTIME_DIR")
        )
        library_raw = os.environ.get("CITRA_LIBRARY")
        self.library = (
            Path(library_raw).resolve()
            if library_raw
            else self.workspace.parent / "library"
        )

        self._denied_roots = (self.library,)

        self._aliases: dict[str, Path] = {
            "workspace": self.workspace,
            "source": self.source_workspace,
            "home": self.home,
            "tmp": self.tmp,
            "cache": self.cache,
            "env": self.env,
            "config": self.config,
            "data": self.data,
            "runtime": self.runtime,
        }
        self._read_roots = (
            self.source_workspace,
            *self._aliases.values(),
        )
        self._write_roots = tuple(
            root
            for name, root in self._aliases.items()
            if name not in {"source", "runtime"}
        )

    @staticmethod
    def _required_path(name: str) -> Path:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Sandbox environment is missing {name}.")
        return Path(value).resolve()

    @staticmethod
    def _within(root: Path, path: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def resolve_path(self, value: str | Path,) -> Path:
        raw = str(value)
        alias_raw = raw

        while alias_raw.startswith("./"):
            alias_raw = alias_raw[2:]

        match = _ALIAS.fullmatch(
            alias_raw
        )

        if match:
            alias, remainder = match.groups()

            try:
                base = self._aliases[alias]
            except KeyError as error:
                raise ValueError(
                    f"Unknown workspace path alias: @{alias}"
                ) from error

            candidate = (
                base
                if not remainder
                else base / remainder
            )

        elif raw == "~" or raw.startswith("~/"):
            remainder = (
                ""
                if raw == "~"
                else raw[2:]
            )

            candidate = (
                self.home
                if not remainder
                else self.home / remainder
            )

        else:
            candidate = Path(
                raw
            )

            if not candidate.is_absolute():
                candidate = (
                    self.workspace
                    / candidate
                )

        return candidate.resolve()

    def require_allowed_path(
        self,
        value: str | Path,
    ) -> Path:
        resolved = Path(
            value
        ).resolve()

        if any(
            self._within(
                root,
                resolved,
            )
            for root in self._denied_roots
        ):
            raise ValueError(
                "The Citra document library is accessible only "
                "through the Document tool."
            )

        if any(
            self._within(
                root,
                resolved,
            )
            for root in self._read_roots
        ):
            return resolved

        raise ValueError(
            "Path is outside the model-facing filesystem: "
            f"{resolved}"
        )

    def require_writable_path(self, value: str | Path) -> Path:
        resolved = self.resolve_path(value)
        if any(self._within(root, resolved) for root in self._write_roots):
            return resolved
        raise ValueError(f"Path is read-only: {self.display_path(resolved)}")

    def display_path(self, value: str | Path) -> str:
        resolved = Path(value).resolve()
        ordered = (
            ("", self.workspace),
            ("source", self.source_workspace),
            ("tmp", self.tmp),
            ("home", self.home),
            ("cache", self.cache),
            ("env", self.env),
            ("config", self.config),
            ("data", self.data),
            ("runtime", self.runtime),
        )
        for alias, root in ordered:
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            if not alias:
                return "." if not relative.parts else relative.as_posix()
            return f"@{alias}" if not relative.parts else f"@{alias}/{relative.as_posix()}"
        return str(resolved)

    def write_text_atomic(
        self,
        value: str | Path,
        text: str,
        *,
        encoding: str = "utf-8",
    ) -> Path:
        destination = self.require_writable_path(value)
        parent = self.require_writable_path(destination.parent)
        parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=parent
        )
        temporary = Path(raw)
        try:
            with os.fdopen(descriptor, "w", encoding=encoding) as stream:
                stream.write(text)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination


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
        parent.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.casefold())
    )
    if not entries:
        return f"Parent directory is empty: {fs.display_path(parent)}"
    visible = entries[:MAX_PARENT_ENTRIES]
    lines = [f"  {entry.name}{'/' if entry.is_dir() else ''}" for entry in visible]
    if len(entries) > len(visible):
        lines.append(f"  ... +{len(entries) - len(visible)} more")
    return f"Parent directory {fs.display_path(parent)} contains:\n" + "\n".join(lines)


def _expand_read_path(fs: ScopedFilesystem, value: str) -> list[Path]:
    if not globlib.has_magic(value):
        resolved = fs.resolve_path(value)
        if not resolved.exists():
            raise FileNotFoundError(
                f"File not found: {fs.display_path(resolved)}\n{_parent_listing(fs, resolved)}"
            )
        return [resolved]

    base, pattern = _split_glob(value)
    root = fs.resolve_path(base)
    if not root.exists():
        raise FileNotFoundError(f"Search directory does not exist: {fs.display_path(root)}")
    if not root.is_dir():
        raise NotADirectoryError(f"Glob search root is not a directory: {fs.display_path(root)}")
    matches: set[Path] = set()
    for raw in globlib.glob(
        str(root / pattern),
        recursive=True,
    ):
        resolved = Path(
            raw
        ).resolve()

        if resolved.is_file():
            matches.add(
                resolved
            )
    return sorted(matches, key=lambda path: fs.display_path(path).casefold())


def _convert_readable(fs: ScopedFilesystem, path: Path) -> Path:
    if path.suffix.lower() not in {".pdf", ".ipynb"}:
        return path
    from citra.utils.converters import convert
    # ScopedFilesystem implements the conversion workspace contract inside
    # the sandbox worker without importing controller lifecycle state.
    return fs.require_allowed_path(convert(path, workspace=cast(Any, fs)))


def _read(arguments: dict[str, Any], fs: ScopedFilesystem) -> str:
    top_path = arguments.get("path")
    requests = arguments.get("requests")
    if top_path is not None and requests is not None:
        raise ValueError("Use either 'path' or 'requests', not both.")
    single_literal = False
    if top_path is not None:
        requests = [{
            "path": top_path,
            "offset": arguments.get("offset", 0),
            "limit": arguments.get("limit"),
        }]
        single_literal = not globlib.has_magic(str(top_path))
    elif not requests:
        raise ValueError("'path' or 'requests' is required.")
    elif arguments.get("offset") is not None or arguments.get("limit") is not None:
        raise ValueError("Top-level 'offset' and 'limit' require top-level 'path'.")

    if len(requests) > MAX_READ_REQUESTS:
        raise ValueError(f"At most {MAX_READ_REQUESTS} read requests are allowed.")

    outputs: list[str] = []
    seen: set[Path] = set()
    selected = 0
    omitted = 0
    for request in requests:
        requested = str(request["path"])
        offset = request.get("offset", 0)
        limit = request.get("limit")
        if not isinstance(offset, int) or offset < 0:
            outputs.append(f"===== {requested} =====\nerror: 'offset' must be a non-negative integer.")
            continue
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            outputs.append(f"===== {requested} =====\nerror: 'limit' must be a non-negative integer.")
            continue
        try:
            matches = _expand_read_path(fs, requested)
        except Exception as error:
            outputs.append(f"===== {requested} =====\nerror: {error}")
            continue
        if not matches:
            outputs.append(f"===== {requested} =====\nerror: No files matched pattern: {requested}")
            continue
        for path in matches:
            if path in seen:
                continue
            seen.add(path)
            if selected >= MAX_READ_FILES:
                omitted += 1
                continue
            shown = fs.display_path(path)
            try:
                if not path.is_file():
                    raise IsADirectoryError(f"Path is not a file: {shown}")
                readable = _convert_readable(fs, path)
                with readable.open("r", encoding="utf-8", errors="replace") as stream:
                    lines = stream.readlines()
                chosen = lines[offset:] if limit is None else lines[offset:offset + limit]
                content = "".join(
                    f"{offset + index + 1:4}| {line}"
                    for index, line in enumerate(chosen)
                )
            except Exception as error:
                content = f"error: {error}"
            outputs.append(f"===== {shown} =====\n{content}")
            selected += 1
    if omitted:
        outputs.append(
            "===== truncated =====\n"
            f"{omitted} additional matching file(s) were not read because the "
            f"{MAX_READ_FILES}-file limit was reached."
        )
    if single_literal and selected == 1 and len(outputs) == 1:
        return outputs[0].partition("\n")[2]
    return "\n\n".join(outputs)


def _read_raw(arguments: dict[str, Any], fs: ScopedFilesystem) -> str:
    """Read UTF-8 source text for a trusted semantic client."""
    path = fs.resolve_path(arguments["path"])
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {fs.display_path(path)}")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Source file is too large for language-server synchronization.")
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        return stream.read()


def _write(arguments: dict[str, Any], fs: ScopedFilesystem) -> str:
    fs.write_text_atomic(arguments["path"], arguments["content"])
    return "ok"


def _edit(arguments: dict[str, Any], fs: ScopedFilesystem) -> str:
    path = fs.require_writable_path(arguments["path"])

    if not path.is_file():
        raise FileNotFoundError(
            f"File not found: {fs.display_path(path)}"
        )

    old = arguments.get("old")
    line = arguments.get("line")
    new = arguments["new"]

    if old is not None and line is not None:
        raise ValueError(
            "Use either 'old' for replacement or 'line' for insertion, not both."
        )

    with path.open("r", encoding="utf-8") as stream:
        text = stream.read()

    if line is not None:
        if not isinstance(line, int) or line < 1:
            raise ValueError(
                "'line' must be a positive 1-based line number."
            )

        lines = text.splitlines(keepends=True)

        if line > len(lines) + 1:
            raise ValueError(
                f"Insert line must be between 1 and "
                f"{len(lines) + 1}, got {line}."
            )

        lines.insert(
            line - 1,
            new,
        )

        fs.write_text_atomic(
            path,
            "".join(lines),
        )

        return "ok"

    if old is None:
        raise ValueError(
            "'old' is required for replacement, or use 'line' for insertion."
        )

    if not old:
        return "error: old string cannot be empty"

    count = text.count(old)

    if count == 0:
        return "error: old_string not found"

    replace_all = bool(
        arguments.get("all", False)
    )

    if count > 1 and not replace_all:
        return (
            f"error: old_string appears {count} times, "
            "must be unique (use all=true)"
        )

    replacement = text.replace(
        old,
        new,
        -1 if replace_all else 1,
    )

    fs.write_text_atomic(
        path,
        replacement,
    )

    return "ok"

def _glob(arguments: dict[str, Any], fs: ScopedFilesystem) -> str:
    base = fs.resolve_path(arguments.get("path", "."))
    if not base.is_dir():
        raise NotADirectoryError(f"Glob root is not a directory: {fs.display_path(base)}")
    entries: list[Path] = []
    for raw in globlib.glob(str(base / arguments["pat"]), recursive=True):
        try:
            entries.append(fs.require_allowed_path(raw))
        except ValueError:
            continue
    entries = sorted(set(entries), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    return "\n".join(fs.display_path(path) for path in entries) or "none"


def _grep(arguments: dict[str, Any], fs: ScopedFilesystem) -> str:
    try:
        pattern = re.compile(arguments["pat"])
    except re.error as error:
        return f"error: invalid regex: {error}"

    target = fs.resolve_path(arguments.get("path", "."))

    if not target.exists():
        raise FileNotFoundError(
            f"Grep path does not exist: {fs.display_path(target)}"
        )

    if target.is_file():
        candidates: Iterable[Path] = (target,)
    elif target.is_dir():
        candidates = target.rglob("*")
    else:
        raise ValueError(
            f"Grep path is not a file or directory: {fs.display_path(target)}"
        )

    hits: list[str] = []

    for candidate in candidates:
        try:
            path = fs.require_allowed_path(candidate)
            if not path.is_file():
                continue

            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line_number, line in enumerate(stream, 1):
                    if pattern.search(line):
                        hits.append(
                            f"{fs.display_path(path)}:"
                            f"{line_number}:"
                            f"{line.rstrip()}"
                        )
                        if len(hits) >= MAX_GREP_RESULTS:
                            return "\n".join(hits)

        except (OSError, UnicodeError, ValueError):
            continue

    return "\n".join(hits) or "none"


@dataclass
class _TreeState:
    emitted: int = 0
    files: int = 0
    directories: int = 0
    skipped: int = 0
    truncated: bool = False


def _tree_skipped(name: str, relative: Path, patterns: set[str]) -> bool:
    relative_text = relative.as_posix()
    return any(
        name == pattern
        or relative_text == pattern
        or fnmatchcase(name, pattern)
        or fnmatchcase(relative_text, pattern)
        for pattern in patterns
    )


def _walk_tree(
    directory: Path,
    relative: Path,
    prefix: str,
    depth: int,
    *,
    max_depth: int,
    directories_only: bool,
    hidden: bool,
    patterns: set[str],
    limit: int,
    lines: list[str],
    state: _TreeState,
) -> None:
    if state.truncated or depth >= max_depth:
        return
    entries: list[tuple[Path, bool, bool, Path]] = []
    try:
        candidates = list(directory.iterdir())
    except OSError as error:
        lines.append(f"{prefix}[error reading directory: {error}]")
        return
    for path in candidates:
        child_relative = relative / path.name
        if (not hidden and path.name.startswith(".")) or _tree_skipped(path.name, child_relative, patterns):
            state.skipped += 1
            continue
        try:
            is_link = path.is_symlink()
            is_dir = path.is_dir()
        except OSError:
            is_link = is_dir = False
        if directories_only and not is_dir:
            continue
        entries.append((path, is_dir, is_link, child_relative))
    entries.sort(key=lambda item: (not item[1], item[0].name.casefold(), item[0].name))
    for index, (path, is_dir, is_link, child_relative) in enumerate(entries):
        if state.emitted >= limit:
            state.truncated = True
            return
        last = index == len(entries) - 1
        suffix = "/" if is_dir else ""
        if is_link:
            try:
                suffix += f" -> {path.readlink()}"
            except OSError:
                suffix += " -> ?"
        lines.append(f"{prefix}{'└── ' if last else '├── '}{path.name}{suffix}")
        state.emitted += 1
        if is_dir:
            state.directories += 1
        else:
            state.files += 1
        if is_dir and not is_link:
            _walk_tree(
                path, child_relative, prefix + ("    " if last else "│   "), depth + 1,
                max_depth=max_depth, directories_only=directories_only, hidden=hidden,
                patterns=patterns, limit=limit, lines=lines, state=state,
            )


def _tree(arguments: dict[str, Any], fs: ScopedFilesystem) -> str:
    root = fs.resolve_path(arguments.get("path", "."))
    if not root.exists():
        raise FileNotFoundError(f"Tree root does not exist: {fs.display_path(root)}")
    if not root.is_dir():
        raise NotADirectoryError(f"Tree root is not a directory: {fs.display_path(root)}")
    max_depth = int(arguments.get("max_depth", DEFAULT_TREE_DEPTH))
    limit = int(arguments.get("limit", DEFAULT_TREE_LIMIT))
    if not 0 <= max_depth <= MAX_TREE_DEPTH:
        raise ValueError(f"'max_depth' must be between 0 and {MAX_TREE_DEPTH}.")
    if not 1 <= limit <= MAX_TREE_LIMIT:
        raise ValueError(f"'limit' must be between 1 and {MAX_TREE_LIMIT}.")
    directories_only = bool(arguments.get("directories_only", False))
    patterns = {str(value) for value in arguments.get("skip", [])}
    if arguments.get("use_default_skips", True):
        patterns.update(DEFAULT_TREE_SKIPS)
    lines = [fs.display_path(root)]
    state = _TreeState()
    _walk_tree(
        root, Path(), "", 0, max_depth=max_depth,
        directories_only=directories_only, hidden=bool(arguments.get("hidden", False)),
        patterns=patterns, limit=limit, lines=lines, state=state,
    )
    if state.truncated:
        lines.extend(("", f"[truncated after {state.emitted} entries]"))
    summary = f"{state.directories} {'directory' if state.directories == 1 else 'directories'}"
    if not directories_only:
        summary += f", {state.files} {'file' if state.files == 1 else 'files'}"
    if state.skipped:
        summary += f", {state.skipped} skipped"
    lines.extend(("", summary))
    return "\n".join(lines)


_OPERATIONS = {
    "read": _read,
    "read_raw": _read_raw,
    "write": _write,
    "edit": _edit,
    "glob": _glob,
    "grep": _grep,
    "tree": _tree,
}


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("Filesystem request is too large.")
        request = json.loads(raw.decode("utf-8"))
        if not isinstance(request, dict):
            raise ValueError("Filesystem request must be a JSON object.")
        operation = request.get("operation")
        arguments = request.get("arguments")
        if operation not in _OPERATIONS:
            raise ValueError(f"Unsupported filesystem operation: {operation!r}")
        if not isinstance(arguments, dict):
            raise ValueError("Filesystem arguments must be a JSON object.")
        result = _OPERATIONS[operation](arguments, ScopedFilesystem())
        output_limit = 16 * 1024 * 1024 if operation == "read_raw" else MAX_OUTPUT_CHARS
        if len(result) > output_limit:
            omitted = len(result) - output_limit
            result = result[:output_limit] + f"\n... <truncated {omitted} characters>"
        response = {"ok": True, "result": result}
    except Exception as error:
        response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    sys.stdout.write(json.dumps(response, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
