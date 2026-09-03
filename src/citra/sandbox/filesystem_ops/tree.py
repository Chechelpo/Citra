from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from .base import FilesystemInput, FilesystemOutput, optional_string, require_payload_dict
from .scope import ScopedFilesystem


DEFAULT_TREE_DEPTH = 3
MAX_TREE_DEPTH = 20
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


@dataclass(frozen=True, slots=True)
class TreeOutput(FilesystemOutput):
    """Represent TreeOutput."""
    root: str
    lines: tuple[str, ...]
    directories: int
    files: int
    skipped: int
    directories_only: bool

    @classmethod
    def from_payload(cls, payload: Any) -> "TreeOutput":
        """Create an instance from payload."""
        raw = require_payload_dict(payload)
        root = raw.get("root")
        lines = raw.get("lines")
        directories = raw.get("directories")
        files = raw.get("files")
        skipped = raw.get("skipped")
        directories_only = raw.get("directories_only")
        if not isinstance(root, str):
            raise ValueError("Tree output 'root' must be a string.")
        if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
            raise ValueError("Tree output 'lines' must be an array of strings.")
        for name, value in (
            ("directories", directories),
            ("files", files),
            ("skipped", skipped),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"Tree output '{name}' must be a non-negative integer.")
        if not isinstance(directories_only, bool):
            raise ValueError("Tree output 'directories_only' must be boolean.")
        return cls(
            root=root,
            lines=tuple(lines),
            directories=directories,
            files=files,
            skipped=skipped,
            directories_only=directories_only,
        )

    def to_payload(self) -> dict[str, Any]:
        """Convert the value to payload."""
        return {
            "root": self.root,
            "lines": list(self.lines),
            "directories": self.directories,
            "files": self.files,
            "skipped": self.skipped,
            "directories_only": self.directories_only,
        }

    def render(self) -> str:
        """Handle render."""
        summary = (
            f"{self.directories} "
            f"{'directory' if self.directories == 1 else 'directories'}"
        )
        if not self.directories_only:
            summary += (
                f", {self.files} "
                f"{'file' if self.files == 1 else 'files'}"
            )
        if self.skipped:
            summary += f", {self.skipped} skipped"
        return "\n".join((self.root, *self.lines, "", summary))


@dataclass(frozen=True, slots=True)
class TreeInput(FilesystemInput[TreeOutput]):
    """Represent TreeInput."""
    operation = "tree"
    output_type = TreeOutput

    path: str = "."
    max_depth: int = DEFAULT_TREE_DEPTH
    directories_only: bool = False
    hidden: bool = False
    skip: tuple[str, ...] = ()
    use_default_skips: bool = True

    def __post_init__(self) -> None:
        """Validate and initialize the instance after construction."""
        if not 0 <= self.max_depth <= MAX_TREE_DEPTH:
            raise ValueError(f"'max_depth' must be between 0 and {MAX_TREE_DEPTH}.")

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> "TreeInput":
        """Handle parse."""
        path = optional_string(arguments, "path", ".")
        raw_depth = arguments.get("max_depth", DEFAULT_TREE_DEPTH)
        try:
            max_depth = int(raw_depth)
        except (TypeError, ValueError) as error:
            raise ValueError("'max_depth' must be an integer.") from error

        raw_skip = arguments.get("skip", [])
        if not isinstance(raw_skip, list):
            raise ValueError("'skip' must be an array.")

        return cls(
            path=path,
            max_depth=max_depth,
            directories_only=bool(arguments.get("directories_only", False)),
            hidden=bool(arguments.get("hidden", False)),
            skip=tuple(str(value) for value in raw_skip),
            use_default_skips=bool(arguments.get("use_default_skips", True)),
        )

    def to_arguments(self) -> dict[str, Any]:
        """Convert the value to arguments."""
        result: dict[str, Any] = {}
        if self.path != ".":
            result["path"] = self.path
        if self.max_depth != DEFAULT_TREE_DEPTH:
            result["max_depth"] = self.max_depth
        if self.directories_only:
            result["directories_only"] = True
        if self.hidden:
            result["hidden"] = True
        if self.skip:
            result["skip"] = list(self.skip)
        if not self.use_default_skips:
            result["use_default_skips"] = False
        return result


@dataclass
class _TreeState:
    """Represent TreeState."""
    files: int = 0
    directories: int = 0
    skipped: int = 0


def _tree_skipped(name: str, relative: Path, patterns: set[str]) -> bool:
    """Handle tree skipped."""
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
    lines: list[str],
    state: _TreeState,
) -> None:
    """Handle walk tree."""
    if depth >= max_depth:
        return

    entries: list[tuple[Path, bool, bool, Path]] = []
    try:
        candidates = list(directory.iterdir())
    except OSError as error:
        lines.append(f"{prefix}[error reading directory: {error}]")
        return

    for path in candidates:
        child_relative = relative / path.name
        if (
            (not hidden and path.name.startswith("."))
            or _tree_skipped(path.name, child_relative, patterns)
        ):
            state.skipped += 1
            continue

        try:
            is_link = path.is_symlink()
            is_dir = path.is_dir()
        except OSError:
            is_link = False
            is_dir = False

        if directories_only and not is_dir:
            continue
        entries.append((path, is_dir, is_link, child_relative))

    entries.sort(
        key=lambda item: (not item[1], item[0].name.casefold(), item[0].name)
    )

    for index, (path, is_dir, is_link, child_relative) in enumerate(entries):
        last = index == len(entries) - 1
        suffix = "/" if is_dir else ""
        if is_link:
            try:
                suffix += f" -> {path.readlink()}"
            except OSError:
                suffix += " -> ?"

        lines.append(
            f"{prefix}{'└── ' if last else '├── '}{path.name}{suffix}"
        )
        if is_dir:
            state.directories += 1
        else:
            state.files += 1

        if is_dir and not is_link:
            _walk_tree(
                path,
                child_relative,
                prefix + ("    " if last else "│   "),
                depth + 1,
                max_depth=max_depth,
                directories_only=directories_only,
                hidden=hidden,
                patterns=patterns,
                lines=lines,
                state=state,
            )


def execute(order: TreeInput, fs: ScopedFilesystem) -> TreeOutput:
    """Execute the execute operation."""
    root = fs.require_allowed_path(fs.resolve_path(order.path))
    if not root.exists():
        raise FileNotFoundError(
            f"Tree root does not exist: {fs.display_path(root)}"
        )
    if not root.is_dir():
        raise NotADirectoryError(
            f"Tree root is not a directory: {fs.display_path(root)}"
        )

    patterns = set(order.skip)
    if order.use_default_skips:
        patterns.update(DEFAULT_TREE_SKIPS)

    lines: list[str] = []
    state = _TreeState()
    _walk_tree(
        root,
        Path(),
        "",
        0,
        max_depth=order.max_depth,
        directories_only=order.directories_only,
        hidden=order.hidden,
        patterns=patterns,
        lines=lines,
        state=state,
    )
    return TreeOutput(
        root=fs.display_path(root),
        lines=tuple(lines),
        directories=state.directories,
        files=state.files,
        skipped=state.skipped,
        directories_only=order.directories_only,
    )
