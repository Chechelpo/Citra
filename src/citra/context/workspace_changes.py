from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
from typing import Sequence


class WorkspaceConflictError(RuntimeError):
    """The source no longer matches the lifecycle materialization baseline."""


class GitCommandError(RuntimeError):
    """A local Git operation used by the workspace controller failed."""


@dataclass(frozen=True)
class MaterializationResult:
    preview: bool
    materialized: tuple[str, ...]
    planned: tuple[str, ...]
    already_materialized: tuple[str, ...]
    absent_from_source: tuple[str, ...]
    ignored: tuple[str, ...]
    hard_excluded: tuple[str, ...]
    unsupported: tuple[str, ...]
    total_bytes: int
    limit_exceeded: bool

    def format(self) -> str:
        selected = (
            self.planned
            if self.preview
            else self.materialized
        )
        verb = (
            "Would materialize"
            if self.preview
            else "Materialized"
        )
        lines = [
            f"{verb} {len(selected)} file(s) "
            f"({_format_bytes(self.total_bytes)}).",
        ]

        if selected:
            lines.extend(
                f"  + {path}"
                for path in selected[:50]
            )

            if len(selected) > 50:
                lines.append(
                    "  ... "
                    f"{len(selected) - 50} more"
                )

        if self.already_materialized:
            lines.append(
                "Already materialized: "
                f"{len(self.already_materialized)} file(s)."
            )

        if self.absent_from_source:
            lines.append(
                "Not found in @source: "
                f"{len(self.absent_from_source)} file(s)."
            )

        if self.ignored:
            lines.append(
                "Ignored during directory/glob expansion: "
                f"{len(self.ignored)} file(s)."
            )

        if self.hard_excluded:
            lines.append(
                "Hard-excluded: "
                + ", ".join(self.hard_excluded[:20])
            )

        if self.unsupported:
            lines.append(
                "Skipped unsupported entries: "
                + ", ".join(self.unsupported[:20])
            )

        if self.limit_exceeded:
            lines.append(
                "The normal materialization limit is exceeded. Narrow the "
                "scope or repeat the copy with allow_large=true."
            )

        return "\n".join(lines)


@dataclass(frozen=True)
class SourceSnapshot:
    kind: str
    mode: int
    size: int
    sha256: str | None = None
    symlink_target: str | None = None


@dataclass(frozen=True)
class _MaterializationPlan:
    copyable: tuple[str, ...]
    already_materialized: tuple[str, ...]
    absent_from_source: tuple[str, ...]
    ignored: tuple[str, ...]
    hard_excluded: tuple[str, ...]
    unsupported: tuple[str, ...]
    total_bytes: int


def _format_bytes(size: int) -> str:
    value = float(size)

    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return (
                f"{int(value)} {unit}"
                if unit == "B"
                else f"{value:.1f} {unit}"
            )

        value /= 1024

    return f"{size} B"


class WorkspaceChanges:
    """
    Owns the private baseline and staging index for one Citra lifecycle.

    The private Git repository is stored outside the model-visible working
    directory. It never writes the source repository's index or history.
    """

    BASELINE_REF = "refs/citra/materialized-baseline"
    MAX_PATCH_BYTES = 2_000_000
    MAX_DIFF_CHARS = 100_000
    MAX_MATERIALIZE_FILES = 10_000
    MAX_MATERIALIZE_BYTES = 512 * 1024 * 1024

    HARD_EXCLUDED_PARTS = frozenset(
        {
            ".git",
            ".hg",
            ".svn",
            "@source",
        }
    )
    SOFT_EXCLUDED_PARTS = frozenset(
        {
            ".mypy_cache",
            ".nox",
            ".pytest_cache",
            ".ruff_cache",
            ".tox",
            ".venv",
            "__pycache__",
            "build",
            "dist",
            "node_modules",
            "venv",
        }
    )

    def __init__(
        self,
        *,
        source_workspace: Path,
        workspace: Path,
        state: Path,
        home: Path,
        git: str,
    ) -> None:
        self.__source_workspace = source_workspace
        self.__workspace = workspace
        self.__git_dir = state / "workspace.git"
        self.__baseline_index = state / "baseline.index"
        self.__staging_index = state / "staging.index"
        self.__inventory_index = state / "inventory.index"
        self.__home = home
        self.__git = git
        self.__materialized: set[str] = set()
        self.__source_snapshots: dict[str, SourceSnapshot] = {}
        self.__source_is_git_repository = False

    @classmethod
    def create(
        cls,
        *,
        source_workspace: Path,
        workspace: Path,
        state: Path,
        home: Path,
    ) -> WorkspaceChanges:
        git = shutil.which("git")

        if git is None:
            raise RuntimeError(
                "Git is required for Citra's private staging index. The "
                "source workspace itself does not need to be a repository."
            )

        instance = cls(
            source_workspace=source_workspace,
            workspace=workspace,
            state=state,
            home=home,
            git=git,
        )
        instance._initialize_private_repository()
        instance.__source_is_git_repository = (
            instance._detect_source_repository()
        )
        return instance

    @property
    def source_workspace(self) -> Path:
        return self.__source_workspace

    @property
    def workspace(self) -> Path:
        return self.__workspace

    @property
    def source_is_git_repository(self) -> bool:
        return self.__source_is_git_repository

    def materialize(
        self,
        pathspecs: Sequence[str],
        *,
        preview: bool = False,
        include_ignored: bool = False,
        allow_large: bool = False,
    ) -> MaterializationResult:
        plan = self._plan_materialization(
            pathspecs,
            include_ignored=include_ignored,
        )
        limit_exceeded = self._materialization_limit_exceeded(
            plan
        )

        if preview:
            return MaterializationResult(
                preview=True,
                materialized=(),
                planned=plan.copyable,
                already_materialized=plan.already_materialized,
                absent_from_source=plan.absent_from_source,
                ignored=plan.ignored,
                hard_excluded=plan.hard_excluded,
                unsupported=plan.unsupported,
                total_bytes=plan.total_bytes,
                limit_exceeded=limit_exceeded,
            )

        if limit_exceeded and not allow_large:
            raise ValueError(
                "Materialization would copy "
                f"{len(plan.copyable)} files "
                f"({_format_bytes(plan.total_bytes)}), exceeding the normal "
                "limit. Narrow the scope, preview it first, or set "
                "allow_large=true."
            )

        copied: list[Path] = []
        snapshots: dict[str, SourceSnapshot] = {}

        try:
            for path in plan.copyable:
                relative = self._validate_repository_path(path)
                source = self.__source_workspace / relative
                destination = self.__workspace / relative

                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(
                        "Materialization would overwrite an existing agent "
                        f"workspace path: {path}"
                    )

                self._validate_destination_parent(relative)
                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                before = self._snapshot_source_path(source)

                if source.is_symlink():
                    destination.symlink_to(
                        os.readlink(source)
                    )
                else:
                    shutil.copy2(
                        source,
                        destination,
                        follow_symlinks=False,
                    )

                copied.append(destination)
                after = self._snapshot_source_path(source)

                if before != after:
                    raise WorkspaceConflictError(
                        "Source changed while it was being materialized: "
                        f"{path}"
                    )

                snapshots[path] = after

            if plan.copyable:
                self._add_to_index(
                    self.__baseline_index,
                    plan.copyable,
                )
                self._add_to_index(
                    self.__staging_index,
                    plan.copyable,
                )
                self._commit_current_baseline_index(
                    "Materialize selected source files"
                )

        except Exception:
            for destination in reversed(copied):
                destination.unlink(missing_ok=True)
            raise

        self.__materialized.update(
            plan.copyable
        )
        self.__source_snapshots.update(snapshots)

        return MaterializationResult(
            preview=False,
            materialized=plan.copyable,
            planned=(),
            already_materialized=plan.already_materialized,
            absent_from_source=plan.absent_from_source,
            ignored=plan.ignored,
            hard_excluded=plan.hard_excluded,
            unsupported=plan.unsupported,
            total_bytes=plan.total_bytes,
            limit_exceeded=(
                limit_exceeded
                and not allow_large
            ),
        )

    def _plan_materialization(
        self,
        pathspecs: Sequence[str],
        *,
        include_ignored: bool,
    ) -> _MaterializationPlan:
        normalized_specs = self._validate_source_pathspecs(
            pathspecs
        )
        discovered = set(
            self._inventory_paths(
                normalized_specs,
                include_ignored=include_ignored,
            )
        )
        source_tracked: set[str] = set()

        if self.__source_is_git_repository:
            source_tracked.update(
                self._source_tracked_paths(
                    normalized_specs
                )
            )
            discovered.update(source_tracked)

        ignored: set[str] = set()

        if not include_ignored:
            ignored.update(
                self._ignored_inventory_paths(
                    normalized_specs
                )
            )
            ignored.difference_update(
                source_tracked
            )
            ignored.update(
                self._citra_ignored_paths(
                    normalized_specs
                )
            )

        explicit_files: set[str] = set()
        absent: set[str] = set()
        unsupported: set[str] = set()
        hard_excluded: set[str] = set()

        for spec in normalized_specs:
            if spec == "." or self._has_pathspec_magic(spec):
                continue

            relative = self._validate_repository_path(
                spec
            )

            if self._is_hard_excluded(relative):
                hard_excluded.add(spec)
                continue

            source = self.__source_workspace / relative

            if source.is_symlink() or source.is_file():
                # An exact file request intentionally overrides soft and
                # ignore-file exclusions.
                explicit_files.add(spec)
            elif source.is_dir():
                continue
            elif os.path.lexists(source):
                unsupported.add(spec)
            else:
                absent.add(spec)

        discovered.update(explicit_files)
        ignored.difference_update(explicit_files)

        copyable: list[str] = []
        already: list[str] = []
        total_bytes = 0

        for path in sorted(discovered):
            relative = self._validate_repository_path(path)

            if self._is_hard_excluded(relative):
                hard_excluded.add(path)
                continue

            if (
                not include_ignored
                and path not in explicit_files
                and path in ignored
            ):
                continue

            if (
                not include_ignored
                and path not in explicit_files
                and self._is_soft_excluded(relative)
            ):
                ignored.add(path)
                continue

            source = self.__source_workspace / relative

            if path in self.__materialized:
                already.append(path)
                continue

            if source.is_symlink() or source.is_file():
                copyable.append(path)
                total_bytes += source.lstat().st_size
            elif not os.path.lexists(source):
                absent.add(path)
            else:
                unsupported.add(path)

        return _MaterializationPlan(
            copyable=tuple(copyable),
            already_materialized=tuple(already),
            absent_from_source=tuple(sorted(absent)),
            ignored=tuple(sorted(ignored)),
            hard_excluded=tuple(sorted(hard_excluded)),
            unsupported=tuple(sorted(unsupported)),
            total_bytes=total_bytes,
        )

    def _inventory_paths(
        self,
        pathspecs: Sequence[str],
        *,
        include_ignored: bool,
    ) -> tuple[str, ...]:
        arguments = [
            *self._inventory_ignore_config(),
            "ls-files",
            "-z",
            "--others",
        ]

        if not include_ignored:
            arguments.append(
                "--exclude-standard"
            )

        arguments.extend(
            [
                "--",
                *pathspecs,
            ]
        )
        result = self._private_git(
            arguments,
            index=self.__inventory_index,
            work_tree=self.__source_workspace,
            text=False,
        )
        return self._decode_git_paths(
            result.stdout
        )

    def _ignored_inventory_paths(
        self,
        pathspecs: Sequence[str],
    ) -> tuple[str, ...]:
        result = self._private_git(
            [
                *self._inventory_ignore_config(),
                "ls-files",
                "-z",
                "--others",
                "--ignored",
                "--exclude-standard",
                "--",
                *pathspecs,
            ],
            index=self.__inventory_index,
            work_tree=self.__source_workspace,
            text=False,
        )
        return self._decode_git_paths(
            result.stdout
        )

    def _inventory_ignore_config(self) -> tuple[str, ...]:
        citra_ignore = self.__source_workspace / ".citraignore"

        if not citra_ignore.is_file():
            return ()

        return (
            "-c",
            f"core.excludesFile={citra_ignore}",
        )

    def _citra_ignored_paths(
        self,
        pathspecs: Sequence[str],
    ) -> tuple[str, ...]:
        citra_ignore = self.__source_workspace / ".citraignore"

        if not citra_ignore.is_file():
            return ()

        result = self._private_git(
            [
                "ls-files",
                "-z",
                "--others",
                "--ignored",
                f"--exclude-from={citra_ignore}",
                "--",
                *pathspecs,
            ],
            index=self.__inventory_index,
            work_tree=self.__source_workspace,
            text=False,
        )
        return self._decode_git_paths(
            result.stdout
        )

    def _source_tracked_paths(
        self,
        pathspecs: Sequence[str],
    ) -> tuple[str, ...]:
        result = self._source_git(
            [
                "ls-files",
                "-z",
                "--cached",
                "--",
                *pathspecs,
            ],
            text=False,
        )
        return self._decode_git_paths(
            result.stdout
        )

    @staticmethod
    def _decode_git_paths(
        raw: bytes,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    os.fsdecode(value)
                    for value in raw.split(b"\0")
                    if value
                }
            )
        )

    @classmethod
    def _materialization_limit_exceeded(
        cls,
        plan: _MaterializationPlan,
    ) -> bool:
        return (
            len(plan.copyable) > cls.MAX_MATERIALIZE_FILES
            or plan.total_bytes > cls.MAX_MATERIALIZE_BYTES
        )

    @staticmethod
    def _has_pathspec_magic(
        pathspec: str,
    ) -> bool:
        return any(
            character in pathspec
            for character in "*?["
        )

    @classmethod
    def _is_hard_excluded(
        cls,
        relative: Path,
    ) -> bool:
        return any(
            part in cls.HARD_EXCLUDED_PARTS
            for part in relative.parts
        )

    @classmethod
    def _is_soft_excluded(
        cls,
        relative: Path,
    ) -> bool:
        return any(
            part in cls.SOFT_EXCLUDED_PARTS
            for part in relative.parts
        )

    def status(self) -> str:
        result = self._private_git(
            [
                "status",
                "--short",
                "--untracked-files=all",
            ],
            index=self.__staging_index,
        )
        output = result.stdout.strip()
        return output or "(clean)"

    def diff(
        self,
        *,
        staged: bool,
        paths: Sequence[str] = (),
    ) -> str:
        arguments = [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
        ]

        if staged:
            arguments.extend(
                [
                    "--cached",
                    "HEAD",
                ]
            )

        if paths:
            arguments.extend(
                [
                    "--",
                    *self._validate_pathspecs(
                        paths,
                        argument_name="paths",
                    ),
                ]
            )

        result = self._private_git(
            arguments,
            index=self.__staging_index,
        )
        output = result.stdout

        if not output:
            return "(no diff)"

        if len(output) <= self.MAX_DIFF_CHARS:
            return output.rstrip()

        omitted = len(output) - self.MAX_DIFF_CHARS
        return (
            output[:self.MAX_DIFF_CHARS]
            + f"\n... <truncated {omitted} characters>"
        )

    def stage(
        self,
        paths: Sequence[str],
    ) -> str:
        normalized = self._validate_pathspecs(
            paths,
            argument_name="paths",
        )
        self._private_git(
            [
                "add",
                "-A",
                "-f",
                "--",
                *normalized,
            ],
            index=self.__staging_index,
        )
        return self.status()

    def stage_patch(
        self,
        patch: str,
    ) -> str:
        if not patch.strip():
            raise ValueError(
                "'patch' cannot be empty."
            )

        encoded = patch.encode("utf-8")

        if len(encoded) > self.MAX_PATCH_BYTES:
            raise ValueError(
                "'patch' exceeds the 2 MB limit."
            )

        arguments = [
            "apply",
            "--cached",
            "--recount",
            "--whitespace=nowarn",
            "-",
        ]

        self._private_git(
            [
                *arguments[:-1],
                "--check",
                arguments[-1],
            ],
            index=self.__staging_index,
            input_data=encoded,
            text=False,
        )
        self._private_git(
            arguments,
            index=self.__staging_index,
            input_data=encoded,
            text=False,
        )
        return self.status()

    def unstage(
        self,
        paths: Sequence[str],
    ) -> str:
        normalized = self._validate_pathspecs(
            paths,
            argument_name="paths",
        )
        self._private_git(
            [
                "reset",
                "--quiet",
                "HEAD",
                "--",
                *normalized,
            ],
            index=self.__staging_index,
        )
        return self.status()

    def apply(self) -> str:
        summary = self._private_git(
            [
                "diff",
                "--cached",
                "--name-status",
                "--find-renames",
                "HEAD",
            ],
            index=self.__staging_index,
        ).stdout.strip()

        patch_result = self._private_git(
            [
                "diff",
                "--cached",
                "--binary",
                "--full-index",
                "--no-color",
                "--no-ext-diff",
                "HEAD",
            ],
            index=self.__staging_index,
            text=False,
        )
        patch = patch_result.stdout

        if not patch:
            return "No staged file updates to apply."

        staged_paths = self._staged_paths()
        self._verify_staged_source_paths(
            staged_paths
        )

        apply_arguments = [
            "apply",
            "--binary",
            "--whitespace=nowarn",
            "-",
        ]

        try:
            self._private_git(
                [
                    *apply_arguments[:-1],
                    "--check",
                    apply_arguments[-1],
                ],
                work_tree=self.__source_workspace,
                input_data=patch,
                text=False,
            )
        except GitCommandError as error:
            raise WorkspaceConflictError(
                "Cannot apply staged updates because the source workspace "
                "no longer matches the materialized baseline. "
                f"Git reported: {error}"
            ) from error

        self._private_git(
            apply_arguments,
            work_tree=self.__source_workspace,
            input_data=patch,
            text=False,
        )

        self._advance_baseline_to_staging()
        self._refresh_source_snapshots(
            staged_paths
        )

        return (
            "Applied staged file updates to @source."
            + (f"\n{summary}" if summary else "")
        )

    def _staged_paths(self) -> tuple[str, ...]:
        result = self._private_git(
            [
                "diff",
                "--cached",
                "--name-only",
                "--no-renames",
                "-z",
                "HEAD",
            ],
            index=self.__staging_index,
            text=False,
        )
        paths = self._decode_git_paths(
            result.stdout
        )

        for path in paths:
            relative = self._validate_repository_path(path)

            if self._is_hard_excluded(relative):
                raise ValueError(
                    "Staged updates cannot address a protected path: "
                    f"{path}"
                )

        return paths

    def _verify_staged_source_paths(
        self,
        paths: Sequence[str],
    ) -> None:
        conflicts: list[str] = []

        for path in paths:
            relative = self._validate_repository_path(path)
            source = self.__source_workspace / relative
            expected = self.__source_snapshots.get(path)

            if expected is None:
                if os.path.lexists(source):
                    conflicts.append(
                        f"{path} (target appeared after the lifecycle started)"
                    )
                continue

            try:
                current = self._snapshot_source_path(source)
            except (FileNotFoundError, ValueError):
                conflicts.append(
                    f"{path} (deleted or replaced externally)"
                )
                continue

            if current != expected:
                conflicts.append(
                    f"{path} (content, mode, or link target changed)"
                )

        if conflicts:
            detail = "; ".join(conflicts[:20])

            if len(conflicts) > 20:
                detail += f"; ... {len(conflicts) - 20} more"

            raise WorkspaceConflictError(
                "Cannot apply staged updates because @source no longer "
                f"matches the materialized baseline: {detail}"
            )

    def _refresh_source_snapshots(
        self,
        paths: Sequence[str],
    ) -> None:
        self.__materialized.update(paths)

        for path in paths:
            source = (
                self.__source_workspace
                / self._validate_repository_path(path)
            )

            if os.path.lexists(source):
                self.__source_snapshots[path] = (
                    self._snapshot_source_path(source)
                )
            else:
                self.__source_snapshots.pop(
                    path,
                    None,
                )

    @staticmethod
    def _snapshot_source_path(
        path: Path,
    ) -> SourceSnapshot:
        metadata = path.lstat()
        mode = stat.S_IMODE(
            metadata.st_mode
        )

        if stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(path)
            return SourceSnapshot(
                kind="symlink",
                mode=mode,
                size=metadata.st_size,
                symlink_target=target,
            )

        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                "Only regular files and symbolic links can be "
                f"materialized: {path}"
            )

        digest = hashlib.sha256()

        with path.open("rb") as source:
            for block in iter(
                lambda: source.read(1024 * 1024),
                b"",
            ):
                digest.update(block)

        return SourceSnapshot(
            kind="file",
            mode=mode,
            size=metadata.st_size,
            sha256=digest.hexdigest(),
        )

    def _detect_source_repository(self) -> bool:
        try:
            result = self._source_git(
                [
                    "rev-parse",
                    "--is-inside-work-tree",
                ]
            )
        except GitCommandError:
            return False

        return result.stdout.strip() == "true"

    def _initialize_private_repository(self) -> None:
        self.__git_dir.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._raw_git(
            [
                "init",
                "--bare",
                "--quiet",
                str(self.__git_dir),
            ]
        )
        self._private_git(
            [
                "read-tree",
                "--empty",
            ],
            index=self.__baseline_index,
        )
        initial_commit = self._commit_current_baseline_index(
            "Initialize empty materialization baseline",
            parent=None,
            update_existing=False,
        )
        self._private_git(
            [
                "symbolic-ref",
                "HEAD",
                self.BASELINE_REF,
            ]
        )
        self._private_git(
            [
                "read-tree",
                initial_commit,
            ],
            index=self.__staging_index,
        )
        self._private_git(
            [
                "read-tree",
                "--empty",
            ],
            index=self.__inventory_index,
            work_tree=self.__source_workspace,
        )

    def _add_to_index(
        self,
        index: Path,
        paths: Sequence[str],
    ) -> None:
        self._private_git(
            [
                "add",
                "-A",
                "-f",
                "--",
                *paths,
            ],
            index=index,
        )

    def _commit_current_baseline_index(
        self,
        message: str,
        *,
        parent: str | None = None,
        update_existing: bool = True,
    ) -> str:
        tree = self._private_git(
            [
                "write-tree",
            ],
            index=self.__baseline_index,
        ).stdout.strip()

        if parent is None and update_existing:
            parent = self._baseline_commit()

        arguments = [
            "commit-tree",
            tree,
            "-m",
            message,
        ]

        if parent is not None:
            arguments.extend(
                [
                    "-p",
                    parent,
                ]
            )

        commit = self._private_git(
            arguments
        ).stdout.strip()

        update_arguments = [
            "update-ref",
            self.BASELINE_REF,
            commit,
        ]

        if parent is not None:
            update_arguments.append(parent)

        self._private_git(
            update_arguments
        )
        return commit

    def _advance_baseline_to_staging(self) -> None:
        old_commit = self._baseline_commit()
        tree = self._private_git(
            [
                "write-tree",
            ],
            index=self.__staging_index,
        ).stdout.strip()
        commit = self._private_git(
            [
                "commit-tree",
                tree,
                "-p",
                old_commit,
                "-m",
                "Apply staged file updates",
            ]
        ).stdout.strip()
        self._private_git(
            [
                "update-ref",
                self.BASELINE_REF,
                commit,
                old_commit,
            ]
        )
        self._private_git(
            [
                "read-tree",
                commit,
            ],
            index=self.__baseline_index,
        )

    def _baseline_commit(self) -> str:
        return self._private_git(
            [
                "rev-parse",
                self.BASELINE_REF,
            ]
        ).stdout.strip()

    def _validate_destination_parent(
        self,
        relative: Path,
    ) -> None:
        parent = self.__workspace / relative.parent
        existing = parent

        while not existing.exists() and existing != self.__workspace:
            existing = existing.parent

        resolved = existing.resolve()

        try:
            resolved.relative_to(
                self.__workspace
            )
        except ValueError as error:
            raise ValueError(
                "Materialization destination escapes the agent workspace: "
                f"{relative.as_posix()}"
            ) from error

        if existing.exists() and not existing.is_dir():
            raise NotADirectoryError(
                "Materialization destination parent is not a directory: "
                f"{relative.parent.as_posix()}"
            )

    @staticmethod
    def _validate_repository_path(
        path: str,
    ) -> Path:
        pure = PurePosixPath(path)

        if (
            not path
            or pure.is_absolute()
            or ".." in pure.parts
            or not pure.parts
        ):
            raise ValueError(
                f"Unsafe source-relative path: {path!r}"
            )

        return Path(*pure.parts)

    @classmethod
    def _validate_pathspecs(
        cls,
        values: Sequence[str],
        *,
        argument_name: str,
    ) -> tuple[str, ...]:
        if not values:
            raise ValueError(
                f"'{argument_name}' must contain at least one path."
            )

        result: list[str] = []

        for raw in values:
            value = raw.strip()

            if not value:
                raise ValueError(
                    f"'{argument_name}' cannot contain empty paths."
                )

            if "\x00" in value:
                raise ValueError(
                    f"'{argument_name}' contains a NUL byte."
                )

            normalized = value.replace("\\", "/")

            while normalized.startswith("./"):
                normalized = normalized[2:]

            pure = PurePosixPath(normalized)

            if (
                pure.is_absolute()
                or normalized == ".."
                or normalized.startswith("../")
                or "/../" in normalized
                or normalized.startswith("@source")
                or normalized.startswith(":")
                or any(
                    part in cls.HARD_EXCLUDED_PARTS
                    for part in pure.parts
                )
            ):
                raise ValueError(
                    "Paths must remain inside the agent workspace and may "
                    f"not address @source or VCS internals: {value!r}"
                )

            result.append(
                normalized or "."
            )

        return tuple(result)

    @classmethod
    def _validate_source_pathspecs(
        cls,
        values: Sequence[str],
    ) -> tuple[str, ...]:
        if not values:
            raise ValueError(
                "'paths' must contain at least one path."
            )

        result: list[str] = []

        for raw in values:
            value = raw.strip()

            if not value:
                raise ValueError(
                    "'paths' cannot contain empty paths."
                )

            if "\x00" in value:
                raise ValueError(
                    "'paths' contains a NUL byte."
                )

            normalized = value.replace("\\", "/")

            while normalized.startswith("./"):
                normalized = normalized[2:]

            normalized = normalized or "."
            pure = PurePosixPath(normalized)

            if (
                pure.is_absolute()
                or normalized == ".."
                or normalized.startswith("../")
                or "/../" in normalized
                or normalized.startswith("@source")
                or normalized.startswith(":")
            ):
                raise ValueError(
                    "Source paths must remain inside @source and use plain "
                    f"filesystem paths or glob patterns: {value!r}"
                )

            result.append(normalized)

        return tuple(result)

    def _source_git(
        self,
        arguments: Sequence[str],
        *,
        text: bool = True,
    ) -> subprocess.CompletedProcess:
        return self._raw_git(
            [
                "-C",
                str(self.__source_workspace),
                *arguments,
            ],
            text=text,
        )

    def _private_git(
        self,
        arguments: Sequence[str],
        *,
        index: Path | None = None,
        work_tree: Path | None = None,
        input_data: str | bytes | None = None,
        text: bool = True,
    ) -> subprocess.CompletedProcess:
        environment = self._git_environment()

        if index is not None:
            environment["GIT_INDEX_FILE"] = str(index)

        return self._raw_git(
            [
                "--git-dir",
                str(self.__git_dir),
                "--work-tree",
                str(work_tree or self.__workspace),
                "-c",
                "core.bare=false",
                "-c",
                "core.autocrlf=false",
                *arguments,
            ],
            environment=environment,
            input_data=input_data,
            text=text,
            cwd=work_tree,
        )

    def _raw_git(
        self,
        arguments: Sequence[str],
        *,
        environment: dict[str, str] | None = None,
        input_data: str | bytes | None = None,
        text: bool = True,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess:
        env = (
            self._git_environment()
            if environment is None
            else environment
        )

        result = subprocess.run(
            [
                self.__git,
                "--no-pager",
                *arguments,
            ],
            env=env,
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            check=False,
            cwd=cwd,
        )

        if result.returncode == 0:
            return result

        stderr = result.stderr

        if isinstance(stderr, bytes):
            detail = stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()
        else:
            detail = stderr.strip()

        raise GitCommandError(
            detail
            or f"Git exited with code {result.returncode}."
        )

    def _git_environment(self) -> dict[str, str]:
        environment: dict[str, str] = {}

        for name in (
            "PATH",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TZ",
        ):
            value = os.environ.get(name)
            if value is not None:
                environment[name] = value

        environment.update(
            {
                "HOME": str(self.__home),
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_ATTR_NOSYSTEM": "1",
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_ASKPASS": "/bin/false",
                "GIT_AUTHOR_NAME": "Citra Workspace",
                "GIT_AUTHOR_EMAIL": "citra@localhost",
                "GIT_COMMITTER_NAME": "Citra Workspace",
                "GIT_COMMITTER_EMAIL": "citra@localhost",
            }
        )
        return environment
