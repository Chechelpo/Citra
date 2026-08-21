from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from citra.context.libraries.repository_library import RepositoryLibrary
from citra.utils.git_utility import GitRepositoryUtility, GitUtilityError


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.strip()


def _create_remote(tmp_path: Path) -> tuple[str, str, str, str]:
    work = tmp_path / "work"
    remote = tmp_path / "remote.git"

    _git("init", "-b", "main", str(work))
    _git("config", "user.email", "test@example.com", cwd=work)
    _git("config", "user.name", "Test", cwd=work)

    (work / "old.txt").write_text("one\n", encoding="utf-8")
    (work / "src").mkdir()
    (work / "src/app.py").write_text("print('one')\n", encoding="utf-8")
    _git("add", ".", cwd=work)
    _git("commit", "-m", "first", cwd=work)
    first = _git("rev-parse", "HEAD", cwd=work)

    _git("branch", "feature", cwd=work)

    _git("mv", "old.txt", "renamed.txt", cwd=work)
    (work / "src/app.py").write_text("print('two')\n", encoding="utf-8")
    _git("add", ".", cwd=work)
    _git("commit", "-m", "second", cwd=work)
    second = _git("rev-parse", "HEAD", cwd=work)

    _git("init", "--bare", str(remote))
    _git("remote", "add", "origin", str(remote), cwd=work)
    _git("push", "origin", "main", "feature", cwd=work)
    _git("symbolic-ref", "HEAD", "refs/heads/main", cwd=remote)

    return remote.as_uri(), first, second, str(work)


@pytest.fixture
def git_utility(tmp_path: Path) -> GitRepositoryUtility:
    return GitRepositoryUtility(
        tmp_path / ".citra/git/repos",
        allow_file_urls=True,
    )


def test_resolve_default_and_explicit_branch(tmp_path: Path, git_utility):
    url, first, second, _ = _create_remote(tmp_path)

    resolved = git_utility.resolve_revision(url)
    assert resolved.branch == "main"
    assert resolved.commit == second

    feature = git_utility.resolve_revision(url, branch="feature")
    assert feature.branch == "feature"
    assert feature.commit == first


def test_commit_without_branch_prefers_default_branch(tmp_path: Path, git_utility):
    url, first, _, _ = _create_remote(tmp_path)

    resolved = git_utility.resolve_revision(url, commit=first[:12])
    assert resolved.branch == "main"
    assert resolved.commit == first


def test_branch_plus_unreachable_commit_is_rejected(tmp_path: Path, git_utility):
    url, _, second, _ = _create_remote(tmp_path)

    with pytest.raises(GitUtilityError, match="not reachable"):
        git_utility.resolve_revision(
            url,
            branch="feature",
            commit=second,
        )


def test_changed_files_includes_both_sides_of_rename(tmp_path: Path, git_utility):
    url, first, second, _ = _create_remote(tmp_path)

    changed = git_utility.changed_files(url, first, second)
    assert Path("old.txt") in changed
    assert Path("renamed.txt") in changed
    assert Path("src/app.py") in changed


def test_cache_is_persistent_and_removeable(tmp_path: Path, git_utility):
    url, _, _, _ = _create_remote(tmp_path)

    assert not git_utility.is_cached(url)
    cache = git_utility.refresh(url)
    assert cache.root.is_dir()
    assert git_utility.is_cached(url)

    git_utility.remove_cache(url)
    assert not git_utility.is_cached(url)


def test_remote_policy_rejects_non_github_https(tmp_path: Path):
    utility = GitRepositoryUtility(tmp_path / "cache")

    with pytest.raises(ValueError, match="github.com"):
        utility.cache_path("https://example.com/foo/bar")

    with pytest.raises(ValueError, match="file://"):
        utility.cache_path((tmp_path / "repo.git").as_uri())


def test_repository_library_git_hooks_end_to_end(tmp_path: Path, monkeypatch):
    url, first, second, _ = _create_remote(tmp_path)
    utility = GitRepositoryUtility(
        tmp_path / ".citra/git/repos",
        allow_file_urls=True,
    )

    # RepositoryLibrary intentionally only accepts GitHub URLs, so for this
    # offline integration test we preserve its persistence behavior while
    # substituting its URL canonicalization boundary with the local file URL.
    library = RepositoryLibrary(
        tmp_path / ".citra/library/repos",
        git_utility=utility,
    )

    monkeypatch.setattr(library, "_canonicalize_repo_url", lambda _: url)
    monkeypatch.setattr(library, "_parse_repo_identity", lambda _: ("local", "repo"))
    monkeypatch.setattr(
        library,
        "_get_repo_folder_name",
        lambda _: "repo-localtest",
    )

    created = library.create_repo_from_git(url)
    assert "Created documented repository" in created
    assert second in created

    changed = library.changed_source_files(url, first, second)
    assert "old.txt" in changed
    assert "renamed.txt" in changed
    assert "src/app.py" in changed
