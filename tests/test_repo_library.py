from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from citra.context.libraries.repository_library import RepoFileEdit
from citra.tools.transient.repo_library import RepoLibrary


REPO = "https://github.com/acme/widget"
COMMIT = "a" * 40
NEXT_COMMIT = "b" * 40


class FakeRepositoryLibrary:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def _call(self, name: str, *args, **kwargs) -> str:
        self.calls.append((name, args, kwargs))
        return name

    def list_repositories(self):
        return self._call("list_repositories")

    def get_repo(self, *args, **kwargs):
        return self._call("get_repo", *args, **kwargs)

    def list_versions(self, *args, **kwargs):
        return self._call("list_versions", *args, **kwargs)

    def tree_docs(self, *args, **kwargs):
        return self._call("tree_docs", *args, **kwargs)

    def search_docs(self, *args, **kwargs):
        return self._call("search_docs", *args, **kwargs)

    def read_files(self, *args, **kwargs):
        return self._call("read_files", *args, **kwargs)

    def create_repo_from_git(self, *args, **kwargs):
        return self._call("create_repo_from_git", *args, **kwargs)

    def create_version_from_git(self, *args, **kwargs):
        return self._call("create_version_from_git", *args, **kwargs)

    def set_preferred_version(self, *args, **kwargs):
        return self._call("set_preferred_version", *args, **kwargs)

    def delete_repo(self, *args, **kwargs):
        return self._call("delete_repo", *args, **kwargs)

    def delete_version(self, *args, **kwargs):
        return self._call("delete_version", *args, **kwargs)

    def changed_source_files(self, *args, **kwargs):
        return self._call("changed_source_files", *args, **kwargs)

    def add_files(self, *args, **kwargs):
        return self._call("add_files", *args, **kwargs)

    def replace_files(self, *args, **kwargs):
        return self._call("replace_files", *args, **kwargs)

    def edit_files(self, *args, **kwargs):
        return self._call("edit_files", *args, **kwargs)

    def move_paths(self, *args, **kwargs):
        return self._call("move_paths", *args, **kwargs)

    def delete_paths(self, *args, **kwargs):
        return self._call("delete_paths", *args, **kwargs)


@dataclass
class FakeLibraries:
    repositories: FakeRepositoryLibrary


@dataclass
class FakeContext:
    libraries: FakeLibraries


@pytest.fixture
def library() -> FakeRepositoryLibrary:
    return FakeRepositoryLibrary()


@pytest.fixture
def tool(library: FakeRepositoryLibrary) -> RepoLibrary:
    return RepoLibrary(FakeContext(FakeLibraries(library)))  # type: ignore[arg-type]


def test_definition_is_valid(tool: RepoLibrary) -> None:
    assert tool.id == "repo_library"
    assert tool.get_as_tool()["function"]["parameters"]["type"] == "object"


def test_create_uses_git_resolution(
    tool: RepoLibrary,
    library: FakeRepositoryLibrary,
) -> None:
    assert tool.execute(
        {
            "action": "create",
            "repo_url": REPO,
            "branch": "main",
            "commit": COMMIT,
        }
    ) == "create_repo_from_git"

    assert library.calls == [
        (
            "create_repo_from_git",
            (REPO,),
            {"branch": "main", "commit": COMMIT},
        )
    ]


def test_read_converts_paths(
    tool: RepoLibrary,
    library: FakeRepositoryLibrary,
) -> None:
    tool.execute(
        {
            "action": "read",
            "repo_url": REPO,
            "files": ["index.md", "api/auth.md"],
        }
    )

    _, args, kwargs = library.calls[-1]
    assert args == (
        REPO,
        [Path("index.md"), Path("api/auth.md")],
    )
    assert kwargs == {"commit": None}


def test_add_converts_documents_to_path_map(
    tool: RepoLibrary,
    library: FakeRepositoryLibrary,
) -> None:
    tool.execute(
        {
            "action": "add",
            "repo_url": REPO,
            "documents": [
                {"path": "api/index.md", "content": "# API"},
                {"path": "api/auth.md", "content": "# Auth"},
            ],
        }
    )

    _, args, kwargs = library.calls[-1]
    assert args == (
        REPO,
        {
            Path("api/index.md"): "# API",
            Path("api/auth.md"): "# Auth",
        },
    )
    assert kwargs == {"commit": None}


def test_edit_converts_to_repo_file_edits(
    tool: RepoLibrary,
    library: FakeRepositoryLibrary,
) -> None:
    tool.execute(
        {
            "action": "edit",
            "repo_url": REPO,
            "commit": COMMIT,
            "edits": [
                {
                    "path": "api/auth.md",
                    "content": "replacement",
                    "at_line": 2,
                    "to_line": 4,
                }
            ],
        }
    )

    _, args, kwargs = library.calls[-1]
    assert args == (
        REPO,
        [
            RepoFileEdit(
                path=Path("api/auth.md"),
                content="replacement",
                at_line=2,
                to_line=4,
            )
        ],
    )
    assert kwargs == {"commit": COMMIT}


def test_move_preserves_order(
    tool: RepoLibrary,
    library: FakeRepositoryLibrary,
) -> None:
    tool.execute(
        {
            "action": "move",
            "repo_url": REPO,
            "moves": [
                {"source": "a.md", "destination": "x/a.md"},
                {"source": "b.md", "destination": "x/b.md"},
            ],
        }
    )

    _, args, _ = library.calls[-1]
    assert list(args[1].items()) == [
        (Path("a.md"), Path("x/a.md")),
        (Path("b.md"), Path("x/b.md")),
    ]


def test_action_specific_argument_rejection(tool: RepoLibrary) -> None:
    with pytest.raises(ValueError, match="Arguments not valid"):
        tool.execute(
            {
                "action": "versions",
                "repo_url": REPO,
                "query": "unused",
            }
        )


def test_duplicate_document_path_rejected(tool: RepoLibrary) -> None:
    with pytest.raises(ValueError, match="Duplicate documentation path"):
        tool.execute(
            {
                "action": "add",
                "repo_url": REPO,
                "documents": [
                    {"path": "same.md", "content": "a"},
                    {"path": "same.md", "content": "b"},
                ],
            }
        )


def test_invalid_edit_range_rejected(tool: RepoLibrary) -> None:
    with pytest.raises(ValueError, match="to_line must be >= at_line"):
        tool.execute(
            {
                "action": "edit",
                "repo_url": REPO,
                "edits": [
                    {
                        "path": "a.md",
                        "content": "x",
                        "at_line": 5,
                        "to_line": 2,
                    }
                ],
            }
        )
