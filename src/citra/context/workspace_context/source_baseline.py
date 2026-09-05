"""Controller-only source snapshots used by the user-facing apply command."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess

from citra.logging import Logger


_logger = Logger(__name__)
_INVENTORY_EXCLUSIONS = frozenset({".git", ".hg", ".svn", ".citra.logs"})


@dataclass(frozen=True)
class SourceEntry:
    """Store the content and mode identity of one project-relative entry."""

    kind: str
    mode: int = 0
    size: int = 0
    digest: str = ""


MISSING_SOURCE_ENTRY = SourceEntry("missing")


def capture_source_baseline(root: Path) -> dict[str, SourceEntry]:
    """Snapshot every applicable file and symlink in a copied workspace."""
    inventory = project_inventory(root)
    baseline = {
        path: snapshot_source_entry(project_entry_path(root, path))
        for path in inventory
    }
    _logger.info(
        "Captured source apply baseline",
        root=str(root),
        entries=len(baseline),
    )
    return baseline


def project_inventory(root: Path) -> tuple[str, ...]:
    """List workspace entries with Git ignores when ``root`` is a repo root.

    A copied repository root retains its ``.git`` metadata, so Git remains the
    authoritative inventory there. A plain directory or a copied subdirectory
    of a larger repository has no local repository metadata; those trees use a
    filesystem inventory and therefore remain fully applyable.
    """
    resolved = root.resolve()
    if not resolved.is_dir():
        _logger.error("Cannot inventory a missing workspace", root=str(resolved))
        raise NotADirectoryError(f"Workspace does not exist: {resolved}")
    repository = git_repository_root(resolved)
    if repository == resolved:
        inventory = _git_project_inventory(resolved)
        _logger.debug(
            "Inventoried repository-root workspace",
            root=str(resolved),
            entries=len(inventory),
        )
        return inventory
    inventory = _filesystem_project_inventory(resolved)
    _logger.debug(
        "Inventoried repository-independent workspace",
        root=str(resolved),
        entries=len(inventory),
        containing_repository=str(repository) if repository is not None else None,
    )
    return inventory


def git_repository_root(root: Path) -> Path | None:
    """Return the containing Git worktree root, or ``None`` when unavailable."""
    resolved = root.resolve()
    try:
        completed = _git(
            resolved,
            "rev-parse",
            "--path-format=absolute",
            "--show-toplevel",
        )
    except OSError as error:
        _logger.warning(
            "Git repository discovery is unavailable",
            root=str(resolved),
            error=str(error),
        )
        return None
    if completed.returncode != 0:
        _logger.trace(
            "Workspace is not contained in a Git worktree",
            root=str(resolved),
        )
        return None
    output = completed.stdout.strip().splitlines()
    if not output:
        _logger.warning(
            "Git repository discovery returned no root",
            root=str(resolved),
        )
        return None
    repository = Path(output[-1]).resolve()
    try:
        resolved.relative_to(repository)
    except ValueError:
        _logger.error(
            "Git returned a repository outside the selected workspace ancestry",
            root=str(resolved),
            repository=str(repository),
        )
        return None
    _logger.trace(
        "Discovered containing Git worktree",
        root=str(resolved),
        repository=str(repository),
    )
    return repository


def snapshot_source_entry(path: Path) -> SourceEntry:
    """Hash one regular file or symlink without following its final symlink."""
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
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as error:
            _logger.error(
                "Could not snapshot source entry",
                path=str(path),
                error=str(error),
            )
            raise
        return SourceEntry(
            "file",
            mode=mode,
            size=metadata.st_size,
            digest=digest.hexdigest(),
        )
    _logger.warning(
        "Source baseline found an unsupported filesystem entry",
        path=str(path),
        mode=mode,
    )
    return SourceEntry("unsupported", mode=mode, size=metadata.st_size)


def normalize_project_path(raw: str) -> str:
    """Validate and normalize an exact project-relative path."""
    if not isinstance(raw, str) or not raw:
        _logger.warning("Rejected an empty or non-string project path")
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
        _logger.warning("Rejected an unsafe project path", path=raw)
        raise ValueError("Project paths must be exact project-relative paths.")
    return candidate.as_posix()


def project_entry_path(root: Path, relative: str) -> Path:
    """Resolve an entry lexically while rejecting escaping parent symlinks."""
    candidate = root / normalize_project_path(relative)
    try:
        candidate.parent.resolve().relative_to(root.resolve())
    except ValueError as error:
        _logger.error(
            "Project path traverses a symlink outside its workspace",
            root=str(root),
            path=relative,
        )
        raise ValueError(
            f"Path traverses a symlink outside the project: {relative}"
        ) from error
    return candidate


def _git_project_inventory(root: Path) -> tuple[str, ...]:
    """Return tracked and non-ignored untracked paths from a Git root."""
    completed = _git(
        root,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    )
    if completed.returncode != 0:
        detail = completed.stdout.strip() or "Git inventory failed."
        _logger.error("Git workspace inventory failed", root=str(root), error=detail)
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


def _filesystem_project_inventory(root: Path) -> tuple[str, ...]:
    """Return file, symlink, and unsupported entries without following links."""
    discovered: list[str] = []

    def visit(directory: Path, relative_directory: Path) -> None:
        """Walk one directory while pruning controller and VCS metadata."""
        try:
            with os.scandir(directory) as entries:
                ordered = sorted(entries, key=lambda entry: entry.name)
        except OSError as error:
            _logger.error(
                "Could not scan workspace for apply inventory",
                directory=str(directory),
                error=str(error),
            )
            raise
        for entry in ordered:
            if entry.name in _INVENTORY_EXCLUSIONS:
                _logger.trace(
                    "Excluded workspace metadata from apply inventory",
                    entry=str(relative_directory / entry.name),
                )
                continue
            relative = relative_directory / entry.name
            if entry.is_dir(follow_symlinks=False):
                visit(Path(entry.path), relative)
                continue
            discovered.append(normalize_project_path(relative.as_posix()))

    visit(root, Path())
    return tuple(discovered)


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run a non-interactive Git command for controller-side discovery."""
    environment = os.environ.copy()
    environment["GIT_PAGER"] = "cat"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    _logger.trace("Running Git discovery command", root=str(root), command=arguments)
    return subprocess.run(
        ["git", "-C", str(root), "--no-pager", *arguments],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


__all__ = [
    "MISSING_SOURCE_ENTRY",
    "SourceEntry",
    "capture_source_baseline",
    "git_repository_root",
    "normalize_project_path",
    "project_entry_path",
    "project_inventory",
    "snapshot_source_entry",
]
