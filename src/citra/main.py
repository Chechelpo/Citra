#!/usr/bin/env python3
"""Thin executable facade for Citra's modular application runtime."""

from __future__ import annotations

from .agent.response import (
    execute_tool_call,
    get_assistant_message,
    serialize_tool_result,
)
from .agent.runner import AgentRunner, run_agent_turn
from .application import CitraApplication
from .cli.rendering import (
    argument_preview,
    print_header,
    render_markdown,
    result_preview,
)
from .cli.repl import is_command
from .cli.repl import main as _repl_main
from .context import ExecutionContext, WorkspaceContext
from .utils.chat_completions_api import call_api
from .utils.terminal_input import terminal_input


def main() -> None:
    """Run the terminal application.

    Passing the facade's ``call_api`` binding preserves a convenient patch
    seam for embedders and tests while orchestration remains outside this
    module.
    """
    _repl_main(
        api_call=call_api,
        input_service=terminal_input,
    )


__all__ = [
    "AgentRunner",
    "CitraApplication",
    "ExecutionContext",
    "WorkspaceContext",
    "argument_preview",
    "call_api",
    "execute_tool_call",
    "get_assistant_message",
    "is_command",
    "main",
    "print_header",
    "render_markdown",
    "result_preview",
    "run_agent_turn",
    "serialize_tool_result",
    "terminal_input",
]


if __name__ == "__main__":
    main()
