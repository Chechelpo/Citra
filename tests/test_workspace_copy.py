from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from citra.context.workspace_context import RuntimeState, WorkspaceContext


def _git(project: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(project), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def test_project_is_copied_without_legacy_aliases_and_preserved(
    tmp_path: Path,
) -> None:
    original = tmp_path / "input"
    original.mkdir()
    (original / "tracked.txt").write_text("original\n", encoding="utf-8")
    _git(original, "init", "-q")
    _git(original, "config", "user.name", "test")
    _git(original, "config", "user.email", "test@example.invalid")
    _git(original, "add", "tracked.txt")
    _git(original, "commit", "-qm", "initial")

    context = WorkspaceContext.create(
        original,
        temporary_workspace=tmp_path / "runtimes",
        tool_definitions=(),
        runtime_assets=(),
    )
    project = context.workspace
    root = context.root

    assert project != original
    assert (project / ".git").is_dir()
    assert context.source_baseline is not None
    assert "tracked.txt" in context.source_baseline
    assert not (project / "@source").exists()
    assert context.display_path(project) == "."
    assert "CITRA_SOURCE" not in context.environment()
    assert "CITRA_WORKSPACE" not in context.environment()
    for legacy in ("@source", "@workspace"):
        with pytest.raises(ValueError):
            context.resolve_path(legacy)

    (project / "tracked.txt").write_text("changed\n", encoding="utf-8")
    assert (original / "tracked.txt").read_text(encoding="utf-8") == "original\n"

    context.cleanup(preserve_workspace=True)
    assert context.lifecycle_state is RuntimeState.CLOSED
    assert (project / "tracked.txt").read_text(encoding="utf-8") == "changed\n"
    assert (root / ".citra-workspace").is_file()
    assert not (root / "runtime").exists()
