"""Shared post-mutation verification for Edit and Write."""

from __future__ import annotations

from typing import Any


def post_edit_result(context: Any, path: str) -> str:
    sections: list[str] = []

    diagnostics_for_path = getattr(
        context,
        "diagnostics_for_path",
        None,
    )
    if callable(diagnostics_for_path):
        diagnostics = diagnostics_for_path(path)
        if diagnostics:
            sections.append(
                "LSP diagnostics after edit:\n"
                f"{diagnostics}"
            )

    lint_for_path = getattr(
        context,
        "lint_for_path",
        None,
    )
    if callable(lint_for_path):
        lint = lint_for_path(path)
        if lint:
            sections.append(
                "Lint checks after edit:\n"
                f"{lint}"
            )

    if not sections:
        return "ok"

    return "ok\n\n" + "\n\n".join(sections)
