from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Iterable

from citra.context.workspace import WorkspaceContext


DEFAULT_MAX_DEPTH = 3
DEFAULT_LIMIT = 200

MAX_MAX_DEPTH = 20
MAX_LIMIT = 5000

DEFAULT_SKIPS = frozenset(
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


@dataclass(frozen=True)
class _TreeEntry:
    path: Path
    relative: Path
    is_directory: bool
    is_symlink: bool


@dataclass
class _TreeState:
    limit: int

    emitted: int = 0
    files: int = 0
    directories: int = 0
    skipped: int = 0
    truncated: bool = False


def render_tree(
    workspace: WorkspaceContext,
    *,
    path: str | Path = ".",
    max_depth: int = DEFAULT_MAX_DEPTH,
    directories_only: bool = False,
    skip: Iterable[str] = (),
    hidden: bool = False,
    limit: int = DEFAULT_LIMIT,
    use_default_skips: bool = True,
) -> str:
    """
    Render a bounded, deterministic directory tree.

    Relative paths and workspace aliases are resolved through the supplied
    WorkspaceContext.

    Directory symlinks are displayed but never traversed.
    """

    _validate_options(
        max_depth=max_depth,
        limit=limit,
    )

    root = workspace.resolve_path(
        path
    )

    if not root.exists():
        raise FileNotFoundError(
            "Tree root does not exist: "
            f"{workspace.display_path(root)}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            "Tree root is not a directory: "
            f"{workspace.display_path(root)}"
        )

    skip_patterns = _normalize_skip_patterns(
        skip
    )

    if use_default_skips:
        skip_patterns.update(
            DEFAULT_SKIPS
        )

    lines: list[str] = [
        _root_label(
            workspace,
            root,
        )
    ]

    state = _TreeState(
        limit=limit,
    )

    _walk(
        directory=root,
        relative_directory=Path(),
        prefix="",
        depth=0,
        max_depth=max_depth,
        directories_only=directories_only,
        hidden=hidden,
        skip=skip_patterns,
        lines=lines,
        state=state,
    )

    if state.truncated:
        lines.extend(
            (
                "",
                (
                    f"[truncated after "
                    f"{state.emitted} entries]"
                ),
            )
        )

    lines.append(
        ""
    )

    summary = (
        f"{state.directories} "
        f"{'directory' if state.directories == 1 else 'directories'}"
    )

    if not directories_only:
        summary += (
            f", {state.files} "
            f"{'file' if state.files == 1 else 'files'}"
        )

    if state.skipped:
        summary += (
            f", {state.skipped} skipped"
        )

    lines.append(
        summary
    )

    return "\n".join(
        lines
    )


def _walk(
    *,
    directory: Path,
    relative_directory: Path,
    prefix: str,
    depth: int,
    max_depth: int,
    directories_only: bool,
    hidden: bool,
    skip: set[str],
    lines: list[str],
    state: _TreeState,
) -> None:
    if state.truncated:
        return

    if depth >= max_depth:
        return

    try:
        paths = list(
            directory.iterdir()
        )

    except OSError as error:
        lines.append(
            f"{prefix}"
            f"[error reading directory: {error}]"
        )
        return

    entries: list[_TreeEntry] = []

    for path in paths:
        relative = (
            relative_directory
            / path.name
        )

        if (
            not hidden
            and path.name.startswith(".")
        ):
            state.skipped += 1
            continue

        if _should_skip(
            name=path.name,
            relative=relative,
            patterns=skip,
        ):
            state.skipped += 1
            continue

        try:
            is_symlink = path.is_symlink()
            is_directory = path.is_dir()

        except OSError:
            is_symlink = False
            is_directory = False

        if (
            directories_only
            and not is_directory
        ):
            continue

        entries.append(
            _TreeEntry(
                path=path,
                relative=relative,
                is_directory=is_directory,
                is_symlink=is_symlink,
            )
        )

    entries.sort(
        key=lambda entry: (
            not entry.is_directory,
            entry.path.name.casefold(),
            entry.path.name,
        )
    )

    for index, entry in enumerate(
        entries
    ):
        if state.emitted >= state.limit:
            state.truncated = True
            return

        is_last = (
            index == len(entries) - 1
        )

        connector = (
            "└── "
            if is_last
            else "├── "
        )

        suffix = _entry_suffix(
            entry
        )

        lines.append(
            f"{prefix}"
            f"{connector}"
            f"{entry.path.name}"
            f"{suffix}"
        )

        state.emitted += 1

        if entry.is_directory:
            state.directories += 1
        else:
            state.files += 1

        if (
            entry.is_directory
            and not entry.is_symlink
        ):
            child_prefix = (
                prefix
                + (
                    "    "
                    if is_last
                    else "│   "
                )
            )

            _walk(
                directory=entry.path,
                relative_directory=entry.relative,
                prefix=child_prefix,
                depth=depth + 1,
                max_depth=max_depth,
                directories_only=directories_only,
                hidden=hidden,
                skip=skip,
                lines=lines,
                state=state,
            )

            if state.truncated:
                return


def _entry_suffix(
    entry: _TreeEntry,
) -> str:
    suffix = (
        "/"
        if entry.is_directory
        else ""
    )

    if not entry.is_symlink:
        return suffix

    try:
        target = entry.path.readlink()

    except OSError:
        target = "?"

    return (
        f"{suffix} -> {target}"
    )


def _root_label(
    workspace: WorkspaceContext,
    root: Path,
) -> str:
    shown = workspace.display_path(
        root
    )

    if shown == ".":
        return "."

    if shown.endswith(
        "/"
    ):
        return shown

    return (
        f"{shown}/"
    )


def _normalize_skip_patterns(
    values: Iterable[str],
) -> set[str]:
    patterns: set[str] = set()

    for value in values:
        value = value.strip()

        if not value:
            raise ValueError(
                "'skip' cannot contain empty patterns."
            )

        value = value.replace(
            "\\",
            "/",
        )

        while value.startswith(
            "./"
        ):
            value = value[2:]

        value = value.rstrip(
            "/"
        )

        if not value:
            raise ValueError(
                "'skip' cannot contain an empty path."
            )

        patterns.add(
            value
        )

    return patterns


def _should_skip(
    *,
    name: str,
    relative: Path,
    patterns: set[str],
) -> bool:
    relative_text = relative.as_posix()

    for pattern in patterns:
        # Patterns without slashes match basenames at any depth.
        if "/" not in pattern:
            if fnmatchcase(
                name,
                pattern,
            ):
                return True

            continue

        # Path-shaped patterns match relative to the requested tree root.
        if fnmatchcase(
            relative_text,
            pattern,
        ):
            return True

        # fnmatch("foo", "**/foo") is false, although users generally
        # expect **/foo to also match foo at the tree root.
        if (
            pattern.startswith("**/")
            and fnmatchcase(
                relative_text,
                pattern[3:],
            )
        ):
            return True

    return False


def _validate_options(
    *,
    max_depth: int,
    limit: int,
) -> None:
    if max_depth < 0:
        raise ValueError(
            "'max_depth' cannot be negative."
        )

    if max_depth > MAX_MAX_DEPTH:
        raise ValueError(
            f"'max_depth' cannot exceed "
            f"{MAX_MAX_DEPTH}."
        )

    if limit <= 0:
        raise ValueError(
            "'limit' must be greater than zero."
        )

    if limit > MAX_LIMIT:
        raise ValueError(
            f"'limit' cannot exceed "
            f"{MAX_LIMIT}."
        )