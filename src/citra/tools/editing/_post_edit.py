"""Shared post-mutation verification for Edit and Write."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citra.context import ExecutionContext


def post_edit_result(
    context: ExecutionContext,
    path: str,
    *,
    auto_fix: bool | None = None,
) -> str:
    """Run post-edit checks with an optional per-operation fixer override."""
    sections: list[str] = []

    # Linters may rewrite the file (for example Ruff import sorting and
    # formatting), so run them before asking the language server to inspect
    # the final contents.
    lint_for_path = getattr(context, "lint_for_path", None)
    if callable(lint_for_path):
        lint = (
            lint_for_path(path)
            if auto_fix is None
            else lint_for_path(path, auto_fix=auto_fix)
        )
        if lint:
            sections.append(f"Lint checks after edit:\n{lint}")

    diagnostics_for_path = getattr(context, "diagnostics_for_path", None)
    if callable(diagnostics_for_path):
        diagnostics = diagnostics_for_path(path)
        if diagnostics:
            sections.append(f"LSP diagnostics after edit:\n{diagnostics}")

    if not sections:
        return "ok"

    return "ok\n\n" + "\n\n".join(sections)
