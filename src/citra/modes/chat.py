"""Conversational mode with an ordinary writable project directory."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from citra.modes.mode import SandboxConfig, SandboxMode, StaticMode
from citra.tools.default_registry import ToolSet
from citra.tools.transient import (
    Bash,
    Browser,
    Diagram,
    Document,
    Edit,
    Git,
    Glob,
    Lsp,
    PromptUser,
    Read,
    ReadImage,
    Subprocess,
    Tree,
    WebSearch,
    Workspace,
    Write,
)
from citra.utils.prompt import collect_environment

if TYPE_CHECKING:
    from citra.context import ExecutionContext


class ChatMode(StaticMode):
    """A helpful chat assistant that can work with the current project."""

    _NAME = "chat"
    _DESCRIPTION = (
        "Conversational assistant with a writable project and optional tools."
    )
    _TOOLS = ToolSet(
        core_tools=(
            Read,
            Write,
            Edit,
            Glob,
            Tree,
            Workspace,
            PromptUser,
        ),
        deferred_tools=(
            Bash,
            Git,
            Lsp,
            WebSearch,
            Browser,
            Subprocess,
            Document,
            Diagram,
            ReadImage,
        ),
    )
    _SANDBOX_CONFIG = SandboxConfig(
        mode=SandboxMode.FULL_SANDBOX,
    )

    @override
    def get_system_prompt(self, context: ExecutionContext) -> str:
        environment = collect_environment(context)
        return f"""
# Role

You are a helpful conversational assistant. Respond directly to the user's
actual request and match the level of detail they need. Do not turn a simple
question into a software-engineering workflow.

# Environment

{environment.as_prompt_section()}

You have a writable current project rooted at `.`. Treat it as an ordinary
project directory. Inspect or modify it only when the request benefits from
doing so; otherwise answer conversationally without unnecessary tool calls.

When changing files, inspect the relevant context, preserve unrelated work,
and perform practical verification. Use the `workspace` tool to roll back an
exact tracked file when an attempted change is wrong. Do not create Git
commits, stage files, or alter repository history; the user owns commits.

Prefer dedicated tools over Bash when one exists. Use Bash for focused command
execution, tests, builds, and similar project work—not for Git mutation.
""".strip()


__all__ = ["ChatMode"]
