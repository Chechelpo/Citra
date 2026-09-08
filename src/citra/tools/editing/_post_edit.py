"""Shared post-mutation verification for Edit and Write."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from citra.context import ExecutionContext

def post_edit_result(context: ExecutionContext, path: str) -> str:
    """Handle post edit result."""
    sections: list[str] = []

    if callable(context.diagnostics_for_path):
        diagnostics = context.diagnostics_for_path(path)
        if diagnostics:
            sections.append(
                "LSP diagnostics after edit:\n"
                f"{diagnostics}"
            )

    if callable(context.lint_for_path):
        lint = context.lint_for_path(path)
        if lint:
            sections.append(
                "Lint checks after edit:\n"
                f"{lint}"
            )

    if not sections:
        return "ok"

    return "ok\n\n" + "\n\n".join(sections)
