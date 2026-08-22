from __future__ import annotations

from types import SimpleNamespace

import pytest

from citra.tools.transient import Edit, Glob, Grep, Read, Tree, Write
from citra.utils.sandbox import WorkspaceSandbox
from citra.workers.filesystem import ScopedFilesystem


class SpyFilesystem:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, operation: str, arguments: dict) -> str:
        self.calls.append((operation, arguments))
        return "sandbox-result"


@pytest.mark.parametrize(
    ("tool_type", "operation", "arguments"),
    [
        (Read, "read", {"path": "a.py"}),
        (Write, "write", {"path": "a.py", "content": "x\n"}),
        (Edit, "edit", {"path": "a.py", "old": "x", "new": "y"}),
        (Glob, "glob", {"pat": "**/*.py"}),
        (Grep, "grep", {"pat": "symbol"}),
        (Tree, "tree", {"path": "."}),
    ],
)
def test_scoped_filesystem_tools_only_delegate_to_sandbox(
    tool_type,
    operation,
    arguments,
) -> None:
    filesystem = SpyFilesystem()
    context = SimpleNamespace(filesystem=filesystem)
    result = tool_type(context).execute(arguments)
    assert result == "sandbox-result"
    assert filesystem.calls == [(operation, arguments)]


def test_worker_rejects_control_plane_alias(monkeypatch, tmp_path) -> None:
    roots = {
        "CITRA_WORKSPACE": tmp_path / "workspace",
        "CITRA_SOURCE": tmp_path / "source",
        "HOME": tmp_path / "home",
        "CITRA_TMP": tmp_path / "tmp",
        "CITRA_CACHE": tmp_path / "cache",
        "XDG_CONFIG_HOME": tmp_path / "config",
        "XDG_DATA_HOME": tmp_path / "data",
        "XDG_RUNTIME_DIR": tmp_path / "runtime",
    }
    for name, path in roots.items():
        path.mkdir()
        monkeypatch.setenv(name, str(path))
    filesystem = ScopedFilesystem()
    with pytest.raises(ValueError, match="Unknown workspace path alias"):
        filesystem.resolve_path("@state/workspace.git")
    with pytest.raises(ValueError, match="Unknown workspace path alias"):
        filesystem.resolve_path("@agent/state/workspace.git")


def test_resolver_under_run_is_reopened_from_an_inherited_fd() -> None:
    arguments: list[str] = []
    WorkspaceSandbox._append_resolver_bind(
        arguments,
        resolver_fd=17,
        target=Path("/run/systemd/resolve/resolv.conf"),
    )
    assert arguments == [
        "--dir",
        "/run/systemd",
        "--dir",
        "/run/systemd/resolve",
        "--ro-bind-fd",
        "17",
        "/run/systemd/resolve/resolv.conf",
    ]
