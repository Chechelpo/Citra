"""Controller-only source baseline used by the user-facing apply command."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess


@dataclass(frozen=True)
class SourceEntry:
    """Content and mode identity for one project-relative filesystem entry."""

    kind: str
    mode: int = 0
    size: int = 0
    digest: str = ""


MISSING_SOURCE_ENTRY = SourceEntry("missing")


def capture_source_baseline(root: Path) -> dict[str, SourceEntry]:
    """Snapshot tracked and non-ignored files from the initial checkout."""
    return {
        path: snapshot_source_entry(project_entry_path(root, path))
        for path in git_project_inventory(root)
    }


def git_project_inventory(root: Path) -> tuple[str, ...]:
    """Return exact tracked and non-ignored untracked project paths."""
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "--no-pager",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        env={
            **os.environ,
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stdout.strip() or "Git inventory failed."
        raise RuntimeError(detail)
    return tuple(
        sorted(
            {
                normalize_project_path(path)
                for path in completed.stdout.split("\0")
                if path
            }
        )
    )


def snapshot_source_entry(path: Path) -> SourceEntry:
    """Hash one regular file or symlink without following the final symlink."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return MISSING_SOURCE_ENTRY
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path).encode("utf-8", errors="surrogateescape")
        return SourceEntry(
            "symlink",
            mode=mode,
            size=len(target),
            digest=hashlib.sha256(target).hexdigest(),
        )
    if stat.S_ISREG(metadata.st_mode):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return SourceEntry(
            "file",
            mode=mode,
            size=metadata.st_size,
            digest=digest.hexdigest(),
        )
    return SourceEntry("unsupported", mode=mode, size=metadata.st_size)


def normalize_project_path(raw: str) -> str:
    """Validate and normalize an exact Git project-relative path."""
    if not isinstance(raw, str) or not raw:
        raise ValueError("Project paths must be non-empty strings.")
    candidate = PurePosixPath(raw)
    if (
        candidate.is_absolute()
        or candidate in {PurePosixPath("."), PurePosixPath("..")}
        or ".." in candidate.parts
        or ".git" in candidate.parts
        or "\\" in raw
        or "\x00" in raw
    ):
        raise ValueError("Project paths must be exact project-relative paths.")
    return candidate.as_posix()


def project_entry_path(root: Path, relative: str) -> Path:
    """Resolve an entry lexically while rejecting escaping parent symlinks."""
    candidate = root / normalize_project_path(relative)
    try:
        candidate.parent.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"Path traverses a symlink outside the project: {relative}"
        ) from error
    return candidate


__all__ = [
    "MISSING_SOURCE_ENTRY",
    "SourceEntry",
    "capture_source_baseline",
    "git_project_inventory",
    "normalize_project_path",
    "project_entry_path",
    "snapshot_source_entry",
]
