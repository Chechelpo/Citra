from __future__ import annotations

from types import SimpleNamespace

import pytest

from citra.tools.transient import Edit, Find, Glob, Read, Tree, Write


class SpyFilesystem:
    """In-memory stand-in for ``SandboxedFilesystem``.

    Tools always invoke ``self.context.filesystem.execute(SomeInput.parse(args))``
    so the spy must accept the parsed input and return a budgeted result.
    The spy's ``to_budgeted`` returns a constant string so the dispatch
    contract stays easy to assert on.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, operation):
        self.calls.append((operation.operation, operation.to_arguments()))
        return SimpleNamespace(
            to_budgeted=lambda model_id, token_count: "sandbox-result"
        )


def _make_context(filesystem: SpyFilesystem) -> SimpleNamespace:
    """Build a minimal context compatible with the transient tool base class.

    ``Tool.__init__`` resolves the model-facing definition, which calls
    ``context.config.model().id``; supplying a stub here keeps the test
    focused on the dispatch contract instead of the model's identity.
    """
    model = SimpleNamespace(id="test-model")
    config = SimpleNamespace(
        model=lambda: model,
        model_config=lambda: model,
    )
    return SimpleNamespace(
        config=config,
        model_config=config.model_config,
        filesystem=filesystem,
    )


@pytest.mark.parametrize(
    ("tool_type", "operation", "arguments"),
    [
        (Read, "read", {"path": "a.py"}),
        (Write, "write", {"path": "a.py", "content": "x\n"}),
        (Edit, "edit", {"path": "a.py", "old": "x", "new": "y"}),
        (Glob, "glob", {"pat": "**/*.py"}),
        (Tree, "tree", {"path": "."}),
        (Find, "find", {"paths": ["src"]}),
    ],
)
def test_scoped_filesystem_tools_only_delegate_to_sandbox(
    tool_type,
    operation,
    arguments,
) -> None:
    filesystem = SpyFilesystem()
    context = _make_context(filesystem)
    result = tool_type(context).execute(arguments)
    assert result == "sandbox-result"
    assert filesystem.calls == [(operation, arguments)]
