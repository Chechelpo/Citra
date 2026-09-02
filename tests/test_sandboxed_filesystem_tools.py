from __future__ import annotations

from types import SimpleNamespace

import pytest

from citra.tools.transient import Edit, Glob, Read, Tree, Write


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
