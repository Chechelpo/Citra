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
    git_project_inventory,
    normalize_project_path,
    project_entry_path,
    snapshot_source_entry,
)

from .command import Command, CommandResult


class ApplyCommand(Command):
    """Preview, apply, and safely stage model-produced file changes."""

    id = "apply"
    description = "Preview and apply checkout changes to the original source."

    def _run(self, args: str) -> CommandResult:
        """Execute the run operation."""
        include_dirty, force_conflicts, requested = self._parse_args(args)
        source, checkout = self._roots()
        self._require_repository_root(source, label="Original source")
        self._require_repository_root(checkout, label="Project checkout")

        baseline = self.context.workspace.source_baseline
        if baseline is None:
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
                output="No non-ignored checkout changes are available." + detail
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
        dirty = self._dirty_source_paths(source, selected)
        safe_to_stage = tuple(
            path
            for path in selected
            if include_dirty or path not in dirty
        )
        skipped_dirty = tuple(
            path for path in selected if path not in safe_to_stage
        )

        self._print_preview(
            source=source,
            checkout=checkout,
            selected=selected,
            safe_to_stage=safe_to_stage,
            skipped_dirty=skipped_dirty,
            conflicts=conflicts,
            unsupported=unsupported,
        )
        if conflicts and not force_conflicts:
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
            return CommandResult(output="Apply cancelled; no files were changed.")

        changed_during_review = tuple(
            path
            for path, snapshot in original_snapshots.items()
            if snapshot_source_entry(project_entry_path(source, path)) != snapshot
        )
        if changed_during_review:
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

        if safe_to_stage:
            self._git_text(
                source,
                "add",
                "--all",
                "--",
                *(f":(literal){path}" for path in safe_to_stage),
            )

        lines = [
            f"Applied {len(selected)} file change(s) to: {source}",
            f"Staged {len(safe_to_stage)} path(s) in the original repository.",
        ]
        if skipped_dirty:
            lines.extend(
                (
                    "Did not alter staging state for dirty source paths:",
                    *(f"  - {path}" for path in skipped_dirty),
                    "Stage those paths manually if they should be included.",
                )
            )
        lines.extend(
            (
                "",
                "Review and commit from the original repository:",
                f"  cd -- {shlex.quote(str(source))}",
                "  git diff --cached",
                '  git commit -m "Describe the changes"',
            )
        )
        return CommandResult(output="\n".join(lines))

    def _roots(self) -> tuple[Path, Path]:
        """Handle roots."""
        workspace = self.context.workspace
        source = Path(workspace.source_workspace).resolve()
        checkout = Path(workspace.workspace).resolve()
        if source == checkout:
            raise RuntimeError(
                "Apply requires a copied checkout distinct from the original source."
            )
        if not source.is_dir() or not checkout.is_dir():
            raise NotADirectoryError("Source or project checkout is unavailable.")
        return source, checkout

    @staticmethod
    def _parse_args(args: str) -> tuple[bool, bool, tuple[str, ...]]:
        """Handle parse args."""
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
        return include_dirty, force_conflicts, tuple(dict.fromkeys(paths))

    def _changed_paths(
        self,
        baseline: dict[str, SourceEntry],
        checkout: Path,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Handle changed paths."""
        candidates = set(baseline) | set(git_project_inventory(checkout))
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
        return tuple(changed), tuple(unsupported)

    @staticmethod
    def _select_paths(
        available: tuple[str, ...],
        requested: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Handle select paths."""
        if not requested:
            return available
        available_set = set(available)
        missing = tuple(path for path in requested if path not in available_set)
        if missing:
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
        """Handle dirty source paths."""
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
        return frozenset(dirty)

    def _print_preview(
        self,
        *,
        source: Path,
        checkout: Path,
        selected: tuple[str, ...],
        safe_to_stage: tuple[str, ...],
        skipped_dirty: tuple[str, ...],
        conflicts: tuple[str, ...],
        unsupported: tuple[str, ...],
    ) -> None:
        """Handle print preview."""
        print("\nCheckout changes relative to the original source\n")
        print(f"Original: {source}")
        print(f"Checkout: {checkout}\n")
        for path in selected:
            print(self._diff(source, checkout, path).rstrip())
            print()
        print(
            f"Selected: {len(selected)} | stage by default: "
            f"{len(safe_to_stage)} | pre-existing dirty: {len(skipped_dirty)}"
        )
        if skipped_dirty:
            print("Pre-existing dirty source paths will be applied but not staged:")
            for path in skipped_dirty:
                print(f"  - {path}")
            print(
                "Cancel and rerun with --include-dirty to stage them explicitly."
            )
        if conflicts:
            print("Conflicts: original source changed after checkout creation:")
            for path in conflicts:
                print(f"  - {path}")
        if unsupported:
            print("Unsupported entries were excluded:")
            for path in unsupported:
                print(f"  - {path}")
        print()

    def _diff(self, source: Path, checkout: Path, relative: str) -> str:
        """Handle diff."""
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
            raise RuntimeError(detail)
        rendered = completed.stdout
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
        """Handle apply transaction."""
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
            except Exception:
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

    @staticmethod
    def _copy_entry(source: Path, destination: Path) -> None:
        """Handle copy entry."""
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
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _confirm(prompt: str) -> bool:
        """Handle confirm."""
        try:
            answer = input(prompt).strip().casefold()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in {"", "y", "yes"}

    def _require_repository_root(self, root: Path, *, label: str) -> None:
        """Handle require repository root."""
        discovered = self._git_text(root, "rev-parse", "--show-toplevel").strip()
        if not discovered or Path(discovered.splitlines()[-1]).resolve() != root:
            raise RuntimeError(f"{label} must be a Git repository root: {root}")

    @staticmethod
    def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Handle git."""
        environment = os.environ.copy()
        environment["GIT_PAGER"] = "cat"
        environment["GIT_TERMINAL_PROMPT"] = "0"
        return subprocess.run(
            ["git", "-C", str(root), "--no-pager", *arguments],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def _git_text(self, root: Path, *arguments: str) -> str:
        """Handle git text."""
        completed = self._git(root, *arguments)
        if completed.returncode != 0:
            detail = completed.stdout.strip() or "Git command failed."
            raise RuntimeError(detail)
        return completed.stdout


def _exists(path: Path) -> bool:
    """Handle exists."""
    return path.exists() or path.is_symlink()


__all__ = ["ApplyCommand"]
