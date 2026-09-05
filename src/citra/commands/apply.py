"""Apply reviewed checkout changes back to the original source tree."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from citra.context.source_baseline import (
    MISSING_SOURCE_ENTRY,
    SourceEntry,
    git_repository_root,
    normalize_project_path,
    project_entry_path,
    project_inventory,
    snapshot_source_entry,
)
from citra.logging import Logger

from .command import Command, CommandResult


_logger = Logger(__name__)


class ApplyCommand(Command):
    """Apply checkout changes and stage them when a Git worktree is available."""

    id = "apply"
    description = "Preview and apply checkout changes to the original source."

    def _run(self, args: str) -> CommandResult:
        """Preview and apply selected checkout changes after confirmation."""
        _logger.info("Starting source apply command", arguments=args)
        include_dirty, force_conflicts, requested = self._parse_args(args)
        source, checkout = self._roots()
        repository_root = git_repository_root(source)

        baseline = self.context.workspace.source_baseline
        if baseline is None:
            _logger.error("Source apply baseline is unavailable")
            raise RuntimeError(
                "The startup source baseline is unavailable; Citra cannot "
                "safely distinguish model changes from later source edits."
            )
        available, unsupported = self._changed_paths(baseline, checkout)
        selected = self._select_paths(available, requested)
        if not selected:
            detail = (
                " Unsupported entries: " + ", ".join(unsupported)
                if unsupported
                else ""
            )
            return CommandResult(
                output="No applicable checkout changes are available." + detail
            )

        original_snapshots = {
            path: snapshot_source_entry(project_entry_path(source, path))
            for path in selected
        }
        conflicts = tuple(
            path
            for path in selected
            if original_snapshots[path]
            != baseline.get(path, MISSING_SOURCE_ENTRY)
            and original_snapshots[path]
            != snapshot_source_entry(project_entry_path(checkout, path))
        )
        dirty = (
            self._dirty_source_paths(source, selected)
            if repository_root is not None
            else frozenset()
        )
        ignored = (
            self._ignored_source_paths(source, selected)
            if repository_root is not None
            else frozenset()
        )
        safe_to_stage = (
            tuple(
                path
                for path in selected
                if path not in ignored and (include_dirty or path not in dirty)
            )
            if repository_root is not None
            else ()
        )
        skipped_dirty = tuple(
            path for path in selected if path in dirty and path not in safe_to_stage
        )
        skipped_ignored = tuple(path for path in selected if path in ignored)

        self._print_preview(
            source=source,
            checkout=checkout,
            repository_root=repository_root,
            selected=selected,
            safe_to_stage=safe_to_stage,
            skipped_dirty=skipped_dirty,
            skipped_ignored=skipped_ignored,
            conflicts=conflicts,
            unsupported=unsupported,
        )
        if conflicts and not force_conflicts:
            _logger.warning(
                "Blocked apply because source paths changed after startup",
                conflicts=conflicts,
            )
            return CommandResult(
                output=(
                    "Nothing was applied because the original source changed "
                    "after the checkout was created. Review the conflicts "
                    "above, then use /apply --force-conflicts only if the "
                    "checkout should overwrite them."
                )
            )
        if not self._confirm(
            "Apply these changes to the original source? [Y/n] "
        ):
            _logger.info("Source apply was cancelled", selected=len(selected))
            return CommandResult(output="Apply cancelled; no files were changed.")

        changed_during_review = tuple(
            path
            for path, snapshot in original_snapshots.items()
            if snapshot_source_entry(project_entry_path(source, path)) != snapshot
        )
        if changed_during_review:
            _logger.error(
                "Source changed during apply review",
                paths=changed_during_review,
            )
            raise RuntimeError(
                "Original source changed during review; nothing was applied: "
                + ", ".join(changed_during_review)
            )

        self._apply_transaction(
            source=source,
            checkout=checkout,
            selected=selected,
        )
        self._advance_baseline(
            baseline=baseline,
            checkout=checkout,
            selected=selected,
        )

        staged_count = 0
        staging_error: str | None = None
        if safe_to_stage:
            try:
                self._git_text(
                    source,
                    "add",
                    "--all",
                    "--",
                    *(f":(literal){path}" for path in safe_to_stage),
                )
                staged_count = len(safe_to_stage)
            except RuntimeError as error:
                staging_error = str(error)
                _logger.error(
                    "Applied source changes but automatic Git staging failed",
                    repository=str(repository_root),
                    error=staging_error,
                )

        lines = [f"Applied {len(selected)} file change(s) to: {source}"]
        if repository_root is None:
            lines.append(
                "Git staging was skipped because the source is not inside a "
                "Git worktree."
            )
        elif staging_error is None:
            lines.append(
                f"Staged {staged_count} path(s) in repository: {repository_root}"
            )
        else:
            lines.extend(
                (
                    "The files were applied, but automatic Git staging failed:",
                    f"  {staging_error}",
                )
            )
        if skipped_dirty:
            lines.extend(
                (
                    "Did not alter staging state for dirty source paths:",
                    *(f"  - {path}" for path in skipped_dirty),
                    "Stage those paths manually if they should be included.",
                )
            )
        if skipped_ignored:
            lines.extend(
                (
                    "Applied but did not stage Git-ignored source paths:",
                    *(f"  - {path}" for path in skipped_ignored),
                )
            )
        if repository_root is None:
            lines.extend(
                (
                    "",
                    "Review the applied files in the original source:",
                    f"  cd -- {shlex.quote(str(source))}",
                )
            )
        else:
            lines.extend(
                (
                    "",
                    "Review and commit from the containing repository:",
                    f"  cd -- {shlex.quote(str(repository_root))}",
                    "  git diff --cached",
                    '  git commit -m "Describe the changes"',
                )
            )
        _logger.info(
            "Completed source apply command",
            selected=len(selected),
            staged=staged_count,
            repository=str(repository_root) if repository_root is not None else None,
        )
        return CommandResult(output="\n".join(lines))

    def _roots(self) -> tuple[Path, Path]:
        """Return distinct, available source and copied-workspace roots."""
        workspace = self.context.workspace
        source = Path(workspace.source_workspace).resolve()
        checkout = Path(workspace.workspace).resolve()
        if source == checkout:
            _logger.error("Apply source and checkout resolve to the same path")
            raise RuntimeError(
                "Apply requires a copied checkout distinct from the original source."
            )
        if not source.is_dir() or not checkout.is_dir():
            _logger.error(
                "Apply source or checkout is unavailable",
                source=str(source),
                checkout=str(checkout),
            )
            raise NotADirectoryError("Source or project checkout is unavailable.")
        _logger.debug(
            "Resolved apply roots",
            source=str(source),
            checkout=str(checkout),
        )
        return source, checkout

    @staticmethod
    def _parse_args(args: str) -> tuple[bool, bool, tuple[str, ...]]:
        """Parse safety flags and exact project-relative path selections."""
        try:
            tokens = shlex.split(args)
        except ValueError as error:
            raise ValueError(f"Invalid arguments: {error}") from error
        include_dirty = False
        force_conflicts = False
        paths: list[str] = []
        for token in tokens:
            if token == "--include-dirty":
                include_dirty = True
            elif token == "--force-conflicts":
                force_conflicts = True
            elif token.startswith("--"):
                raise ValueError(
                    "Usage: /apply [--include-dirty] [--force-conflicts] "
                    "[path ...]"
                )
            else:
                paths.append(normalize_project_path(token))
        requested = tuple(dict.fromkeys(paths))
        _logger.trace(
            "Parsed source apply arguments",
            include_dirty=include_dirty,
            force_conflicts=force_conflicts,
            paths=requested,
        )
        return include_dirty, force_conflicts, requested

    def _changed_paths(
        self,
        baseline: dict[str, SourceEntry],
        checkout: Path,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return supported changes and unsupported entries since startup."""
        candidates = set(baseline) | set(project_inventory(checkout))
        changed: list[str] = []
        unsupported: list[str] = []
        for path in sorted(candidates):
            baseline_entry = baseline.get(path, MISSING_SOURCE_ENTRY)
            checkout_entry = snapshot_source_entry(
                project_entry_path(checkout, path)
            )
            if "unsupported" in {baseline_entry.kind, checkout_entry.kind}:
                unsupported.append(path)
            elif baseline_entry != checkout_entry:
                changed.append(path)
        _logger.debug(
            "Compared checkout with source baseline",
            candidates=len(candidates),
            changed=len(changed),
            unsupported=len(unsupported),
        )
        return tuple(changed), tuple(unsupported)

    @staticmethod
    def _select_paths(
        available: tuple[str, ...],
        requested: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Select requested changed paths or return every available change."""
        if not requested:
            return available
        available_set = set(available)
        missing = tuple(path for path in requested if path not in available_set)
        if missing:
            _logger.warning(
                "Requested apply paths have no checkout changes",
                paths=missing,
            )
            raise ValueError(
                "Requested paths have no applicable checkout changes: "
                + ", ".join(missing)
            )
        return tuple(path for path in available if path in set(requested))

    def _dirty_source_paths(
        self,
        source: Path,
        selected: tuple[str, ...],
    ) -> frozenset[str]:
        """Return selected paths already dirty in the containing repository."""
        dirty: set[str] = set()
        for path in selected:
            output = self._git_text(
                source,
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--",
                f":(literal){path}",
            )
            if output:
                dirty.add(path)
        _logger.debug("Detected pre-existing dirty source paths", paths=tuple(dirty))
        return frozenset(dirty)

    def _ignored_source_paths(
        self,
        source: Path,
        selected: tuple[str, ...],
    ) -> frozenset[str]:
        """Return untracked selected paths ignored by the containing repository."""
        completed = self._git(
            source,
            "check-ignore",
            "-z",
            "--stdin",
            input_text="\0".join(selected) + "\0",
        )
        if completed.returncode not in {0, 1}:
            detail = completed.stdout.strip() or "Git ignore check failed."
            _logger.error("Could not check Git ignore state", error=detail)
            raise RuntimeError(detail)
        ignored = frozenset(
            normalize_project_path(path)
            for path in completed.stdout.split("\0")
            if path
        )
        _logger.debug("Detected Git-ignored source paths", paths=tuple(ignored))
        return ignored

    def _print_preview(
        self,
        *,
        source: Path,
        checkout: Path,
        repository_root: Path | None,
        selected: tuple[str, ...],
        safe_to_stage: tuple[str, ...],
        skipped_dirty: tuple[str, ...],
        skipped_ignored: tuple[str, ...],
        conflicts: tuple[str, ...],
        unsupported: tuple[str, ...],
    ) -> None:
        """Print diffs, conflicts, and optional Git staging decisions."""
        print("\nCheckout changes relative to the original source\n")
        print(f"Original: {source}")
        print(f"Checkout: {checkout}\n")
        for path in selected:
            print(self._diff(source, checkout, path).rstrip())
            print()
        if repository_root is None:
            print(f"Selected: {len(selected)} | Git staging: unavailable")
            print("Changes can still be applied directly to the original source.")
        else:
            print(
                f"Selected: {len(selected)} | stage by default: "
                f"{len(safe_to_stage)} | pre-existing dirty: "
                f"{len(skipped_dirty)} | Git-ignored: {len(skipped_ignored)}"
            )
            print(f"Containing repository: {repository_root}")
        if skipped_dirty:
            print("Pre-existing dirty source paths will be applied but not staged:")
            for path in skipped_dirty:
                print(f"  - {path}")
            print(
                "Cancel and rerun with --include-dirty to stage them explicitly."
            )
        if skipped_ignored:
            print("Git-ignored paths will be applied but not staged:")
            for path in skipped_ignored:
                print(f"  - {path}")
        if conflicts:
            print("Conflicts: original source changed after checkout creation:")
            for path in conflicts:
                print(f"  - {path}")
        if unsupported:
            print("Unsupported entries were excluded:")
            for path in unsupported:
                print(f"  - {path}")
        print()
        _logger.debug(
            "Rendered source apply preview",
            selected=len(selected),
            conflicts=len(conflicts),
            staging_available=repository_root is not None,
        )

    def _diff(self, source: Path, checkout: Path, relative: str) -> str:
        """Render a binary-safe no-index diff for one selected path."""
        source_path = project_entry_path(source, relative)
        checkout_path = project_entry_path(checkout, relative)
        left = source_path if _exists(source_path) else Path("/dev/null")
        right = checkout_path if _exists(checkout_path) else Path("/dev/null")
        completed = self._git(
            source,
            "diff",
            "--no-index",
            "--no-ext-diff",
            "--binary",
            "--",
            str(left),
            str(right),
        )
        if completed.returncode not in {0, 1}:
            detail = completed.stdout.strip() or f"Could not diff {relative}"
            _logger.error("Could not render source apply diff", path=relative)
            raise RuntimeError(detail)
        rendered = completed.stdout
        for prefix in ("a", "b"):
            rendered = rendered.replace(
                f"{prefix}{source_path}",
                f"{prefix}/source/{relative}",
            )
            rendered = rendered.replace(
                f"{prefix}{checkout_path}",
                f"{prefix}/checkout/{relative}",
            )
        rendered = rendered.replace(str(source_path), f"source/{relative}")
        rendered = rendered.replace(str(checkout_path), f"checkout/{relative}")
        return rendered or f"diff -- source/{relative} checkout/{relative}"

    def _apply_transaction(
        self,
        *,
        source: Path,
        checkout: Path,
        selected: tuple[str, ...],
    ) -> None:
        """Apply all selected entries and roll filesystem writes back on error."""
        with tempfile.TemporaryDirectory(prefix="citra-apply-") as raw_backup:
            backup = Path(raw_backup)
            existed: dict[str, bool] = {}
            for path in selected:
                original = project_entry_path(source, path)
                existed[path] = _exists(original)
                if existed[path]:
                    self._copy_entry(original, backup / path)
            applied: list[str] = []
            try:
                for path in selected:
                    destination = project_entry_path(source, path)
                    candidate = project_entry_path(checkout, path)
                    if _exists(candidate):
                        self._copy_entry(candidate, destination)
                    else:
                        destination.unlink(missing_ok=True)
                    applied.append(path)
            except Exception as error:
                _logger.error(
                    "Source apply transaction failed; restoring prior entries",
                    applied=tuple(applied),
                    error=str(error),
                )
                for path in reversed(applied):
                    destination = project_entry_path(source, path)
                    if existed[path]:
                        self._copy_entry(backup / path, destination)
                    else:
                        destination.unlink(missing_ok=True)
                raise

    @staticmethod
    def _advance_baseline(
        *,
        baseline: dict[str, SourceEntry],
        checkout: Path,
        selected: tuple[str, ...],
    ) -> None:
        """Make a successful apply the baseline for subsequent model edits."""
        for path in selected:
            entry = snapshot_source_entry(project_entry_path(checkout, path))
            if entry == MISSING_SOURCE_ENTRY:
                baseline.pop(path, None)
            else:
                baseline[path] = entry
        _logger.debug("Advanced source apply baseline", paths=selected)

    @staticmethod
    def _copy_entry(source: Path, destination: Path) -> None:
        """Atomically replace one destination with a copied file or symlink."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_dir() and not destination.is_symlink():
            raise IsADirectoryError(destination)
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.citra-",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(raw_temporary)
        temporary.unlink(missing_ok=True)
        try:
            if source.is_symlink():
                temporary.symlink_to(os.readlink(source))
            else:
                shutil.copy2(source, temporary, follow_symlinks=False)
            os.replace(temporary, destination)
            _logger.trace(
                "Copied source apply entry",
                source=str(source),
                destination=str(destination),
            )
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _confirm(prompt: str) -> bool:
        """Request an affirmative terminal confirmation from the user."""
        try:
            answer = input(prompt).strip().casefold()
        except (EOFError, KeyboardInterrupt):
            _logger.warning("Source apply confirmation was interrupted")
            return False
        return answer in {"", "y", "yes"}

    @staticmethod
    def _git(
        root: Path,
        *arguments: str,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a non-interactive controller-side Git command."""
        environment = os.environ.copy()
        environment["GIT_PAGER"] = "cat"
        environment["GIT_TERMINAL_PROMPT"] = "0"
        _logger.trace("Running source apply Git command", command=arguments)
        return subprocess.run(
            ["git", "-C", str(root), "--no-pager", *arguments],
            env=environment,
            text=True,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def _git_text(self, root: Path, *arguments: str) -> str:
        """Run Git and return output, raising a logged error on failure."""
        completed = self._git(root, *arguments)
        if completed.returncode != 0:
            detail = completed.stdout.strip() or "Git command failed."
            _logger.error(
                "Source apply Git command failed",
                command=arguments,
                error=detail,
            )
            raise RuntimeError(detail)
        return completed.stdout


def _exists(path: Path) -> bool:
    """Return whether a path or dangling symlink occupies an entry."""
    return path.exists() or path.is_symlink()


__all__ = ["ApplyCommand"]
