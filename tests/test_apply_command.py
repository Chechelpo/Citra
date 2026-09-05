from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest import mock

from citra.commands.apply import ApplyCommand
from citra.commands.default_registry import COMMAND_REGISTRY
from citra.context.source_baseline import SourceEntry, capture_source_baseline


def _git(project: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(project), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def _repositories(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, SourceEntry]]:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.name", "test")
    _git(source, "config", "user.email", "test@example.invalid")
    (source / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (source / "tracked.py").write_text("before = 1\n", encoding="utf-8")
    _git(source, "add", ".gitignore", "tracked.py")
    _git(source, "commit", "-qm", "initial")
    checkout = tmp_path / "checkout"
    shutil.copytree(source, checkout, symlinks=True)
    return source, checkout, capture_source_baseline(checkout)


def _command(
    source: Path,
    checkout: Path,
    baseline: dict[str, SourceEntry],
) -> ApplyCommand:
    context = SimpleNamespace(
        workspace=SimpleNamespace(
            source_workspace=source,
            workspace=checkout,
            source_baseline=baseline,
        )
    )
    return ApplyCommand(context)


def _plain_workspaces(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, SourceEntry]]:
    source = tmp_path / "plain-source"
    source.mkdir()
    (source / "changed.txt").write_text("before\n", encoding="utf-8")
    (source / "deleted.txt").write_text("remove me\n", encoding="utf-8")
    checkout = tmp_path / "plain-checkout"
    shutil.copytree(source, checkout, symlinks=True)
    return source, checkout, capture_source_baseline(checkout)


def test_apply_command_is_registered() -> None:
    assert COMMAND_REGISTRY.contains("apply")


def test_apply_previews_copies_and_stages_nonignored_changes(
    tmp_path: Path,
    capsys,
) -> None:
    source, checkout, baseline = _repositories(tmp_path)
    (checkout / "tracked.py").write_text("after = 2\n", encoding="utf-8")
    (checkout / "new.py").write_text("created = True\n", encoding="utf-8")
    (checkout / "ignored.txt").write_text("ignore me\n", encoding="utf-8")

    with mock.patch("builtins.input", return_value=""):
        result = _command(source, checkout, baseline).run("")

    preview = capsys.readouterr().out
    assert "relative to the original source" in preview
    assert "tracked.py" in preview
    assert "new.py" in preview
    assert "ignored.txt" not in preview
    assert (source / "tracked.py").read_text(encoding="utf-8") == "after = 2\n"
    assert (source / "new.py").read_text(encoding="utf-8") == "created = True\n"
    assert not (source / "ignored.txt").exists()
    staged = _git(source, "diff", "--cached", "--name-only").splitlines()
    assert staged == ["new.py", "tracked.py"]
    assert "Staged 2 path(s)" in result.output
    assert _git(source, "log", "-1", "--format=%s").strip() == "initial"


def test_apply_does_not_change_index_for_preexisting_dirty_source(
    tmp_path: Path,
) -> None:
    source, checkout, baseline = _repositories(tmp_path)
    (source / "tracked.py").write_text("user = 1\n", encoding="utf-8")
    (checkout / "tracked.py").write_text("model = 2\n", encoding="utf-8")

    with mock.patch("builtins.input", return_value="yes"):
        result = _command(source, checkout, baseline).run("")

    assert (source / "tracked.py").read_text(encoding="utf-8") == "user = 1\n"
    assert _git(source, "diff", "--cached") == ""
    assert "Nothing was applied" in result.output
    assert "--force-conflicts" in result.output


def test_apply_include_dirty_stages_explicitly(tmp_path: Path) -> None:
    source, checkout, baseline = _repositories(tmp_path)
    (source / "tracked.py").write_text("user = 1\n", encoding="utf-8")
    (checkout / "tracked.py").write_text("model = 2\n", encoding="utf-8")

    with mock.patch("builtins.input", return_value="yes"):
        result = _command(source, checkout, baseline).run(
            "--include-dirty --force-conflicts tracked.py"
        )

    assert "Staged 1 path(s)" in result.output
    assert "model = 2" in _git(source, "diff", "--cached")


def test_apply_cancellation_is_non_mutating(tmp_path: Path) -> None:
    source, checkout, baseline = _repositories(tmp_path)
    (checkout / "tracked.py").write_text("after = 2\n", encoding="utf-8")

    with mock.patch("builtins.input", return_value="no"):
        result = _command(source, checkout, baseline).run("")

    assert "no files were changed" in result.output
    assert (source / "tracked.py").read_text(encoding="utf-8") == "before = 1\n"
    assert _git(source, "diff", "--cached") == ""


def test_apply_preserves_source_file_created_after_checkout(
    tmp_path: Path,
) -> None:
    source, checkout, baseline = _repositories(tmp_path)
    (source / "later.py").write_text("user_created = True\n", encoding="utf-8")
    (checkout / "tracked.py").write_text("after = 2\n", encoding="utf-8")

    with mock.patch("builtins.input", return_value="yes"):
        _command(source, checkout, baseline).run("")

    assert (source / "later.py").read_text(encoding="utf-8") == (
        "user_created = True\n"
    )


def test_apply_stages_checkout_deletion(tmp_path: Path) -> None:
    source, checkout, baseline = _repositories(tmp_path)
    (checkout / "tracked.py").unlink()

    with mock.patch("builtins.input", return_value="yes"):
        result = _command(source, checkout, baseline).run("tracked.py")

    assert not (source / "tracked.py").exists()
    assert "tracked.py" in _git(source, "diff", "--cached", "--name-only")
    assert "Staged 1 path(s)" in result.output


def test_apply_advances_baseline_for_later_model_edits(tmp_path: Path) -> None:
    source, checkout, baseline = _repositories(tmp_path)
    (checkout / "tracked.py").write_text("after = 2\n", encoding="utf-8")

    with mock.patch("builtins.input", return_value="yes"):
        first = _command(source, checkout, baseline).run("")
    assert "Staged 1 path(s)" in first.output

    assert "No applicable" in _command(
        source,
        checkout,
        baseline,
    ).run("").output

    (checkout / "tracked.py").write_text("later = 3\n", encoding="utf-8")
    with mock.patch("builtins.input", return_value="yes"):
        second = _command(source, checkout, baseline).run("")

    assert "Applied 1 file change" in second.output
    assert (source / "tracked.py").read_text(encoding="utf-8") == "later = 3\n"


def test_apply_supports_plain_directories_without_git(
    tmp_path: Path,
    capsys,
) -> None:
    source, checkout, baseline = _plain_workspaces(tmp_path)
    (checkout / "changed.txt").write_text("after\n", encoding="utf-8")
    (checkout / "deleted.txt").unlink()
    (checkout / "created.txt").write_text("created\n", encoding="utf-8")

    with mock.patch("builtins.input", return_value="yes"):
        result = _command(source, checkout, baseline).run("")

    preview = capsys.readouterr().out
    assert "Git staging: unavailable" in preview
    assert "Changes can still be applied" in preview
    assert (source / "changed.txt").read_text(encoding="utf-8") == "after\n"
    assert not (source / "deleted.txt").exists()
    assert (source / "created.txt").read_text(encoding="utf-8") == "created\n"
    assert "Applied 3 file change(s)" in result.output
    assert "Git staging was skipped" in result.output
    assert "git commit" not in result.output


def test_apply_plain_directory_retains_conflict_detection(tmp_path: Path) -> None:
    source, checkout, baseline = _plain_workspaces(tmp_path)
    (source / "changed.txt").write_text("user edit\n", encoding="utf-8")
    (checkout / "changed.txt").write_text("model edit\n", encoding="utf-8")

    with mock.patch("builtins.input", return_value="yes") as confirmation:
        result = _command(source, checkout, baseline).run("")

    assert "Nothing was applied" in result.output
    assert (source / "changed.txt").read_text(encoding="utf-8") == "user edit\n"
    confirmation.assert_not_called()


def test_apply_uses_repository_containing_selected_workspace(
    tmp_path: Path,
    capsys,
) -> None:
    repository = tmp_path / "repository"
    source = repository / "packages" / "selected"
    source.mkdir(parents=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "test")
    _git(repository, "config", "user.email", "test@example.invalid")
    (source / "tracked.py").write_text("before = 1\n", encoding="utf-8")
    _git(repository, "add", "packages/selected/tracked.py")
    _git(repository, "commit", "-qm", "initial")
    checkout = tmp_path / "nested-checkout"
    shutil.copytree(source, checkout, symlinks=True)
    baseline = capture_source_baseline(checkout)
    (checkout / "tracked.py").write_text("after = 2\n", encoding="utf-8")
    (checkout / "created.py").write_text("created = True\n", encoding="utf-8")

    with mock.patch("builtins.input", return_value="yes"):
        result = _command(source, checkout, baseline).run("")

    preview = capsys.readouterr().out
    assert f"Containing repository: {repository}" in preview
    staged = _git(repository, "diff", "--cached", "--name-only").splitlines()
    assert staged == [
        "packages/selected/created.py",
        "packages/selected/tracked.py",
    ]
    assert "Staged 2 path(s)" in result.output
    assert f"repository: {repository}" in result.output


def test_apply_nested_workspace_does_not_stage_ignored_paths(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    source = repository / "packages" / "selected"
    source.mkdir(parents=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "test")
    _git(repository, "config", "user.email", "test@example.invalid")
    (repository / ".gitignore").write_text(
        "packages/selected/generated.txt\n",
        encoding="utf-8",
    )
    (source / "tracked.py").write_text("before = 1\n", encoding="utf-8")
    _git(repository, "add", ".gitignore", "packages/selected/tracked.py")
    _git(repository, "commit", "-qm", "initial")
    checkout = tmp_path / "nested-checkout"
    shutil.copytree(source, checkout, symlinks=True)
    baseline = capture_source_baseline(checkout)
    (checkout / "tracked.py").write_text("after = 2\n", encoding="utf-8")
    (checkout / "generated.txt").write_text("generated\n", encoding="utf-8")

    with mock.patch("builtins.input", return_value="yes"):
        result = _command(source, checkout, baseline).run("")

    assert (source / "generated.txt").read_text(encoding="utf-8") == "generated\n"
    staged = _git(repository, "diff", "--cached", "--name-only").splitlines()
    assert staged == ["packages/selected/tracked.py"]
    assert "Applied but did not stage Git-ignored source paths" in result.output
    assert "generated.txt" in result.output
