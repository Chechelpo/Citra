"""Terminal rendering kept separate from agent orchestration."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from openai.types.chat import ChatCompletionMessageFunctionToolCallParam
from rich.console import Console
from rich.markdown import Markdown
from rich.status import Status
from rich.text import Text

from ..context import CitraConfig
from ..tools.session_memory import MemoryTool
from ..tools.tool import Tool
from ..utils.chat_completions_api import build_memory_context

console = Console(
    highlight=False,
)


def format_elapsed(seconds: float) -> str:
    """Format elapsed work time compactly for terminal status messages."""
    total_seconds = max(0, round(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


@contextmanager
def working_animation():
    """Animate while the model is working and report the elapsed duration."""
    started = perf_counter()
    status = Status(
        "[dim]Working…[/dim]",
        console=console,
        spinner="dots",
        refresh_per_second=12.5,
    )
    status.start()
    try:
        yield
    finally:
        status.stop()
        elapsed = format_elapsed(perf_counter() - started)
        line = Text("✻ ", style="magenta")
        line.append(f"Worked for {elapsed}", style="dim")
        console.print(line)


def render_markdown(text: str) -> None:
    """Handle render markdown."""
    console.print(
        Markdown(text)
    )


def argument_preview(
    arguments: dict[str, Any],
    limit: int = 50,
) -> str:
    """Handle argument preview."""
    if not arguments:
        return ""

    text = str(
        next(iter(arguments.values()))
    ).replace(
        "\n",
        " ",
    )

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def result_preview(
    result: str,
    line_limit: int = 60,
) -> str:
    """Handle result preview."""
    lines = result.splitlines()

    if not lines:
        return "(empty)"

    preview = lines[0][
        :line_limit
    ]

    if len(lines) > 1:
        preview += (
            f" ... +{len(lines) - 1} lines"
        )
    elif len(lines[0]) > line_limit:
        preview += "..."

    return preview


def render_tool_call_start(
    tool_call: ChatCompletionMessageFunctionToolCallParam,
    tool: Tool | None = None,
) -> dict[str, Any] | None:
    """Handle render tool call start."""
    function = tool_call["function"]

    name = function.get(
        "name",
        "unknown",
    )

    raw = function.get(
        "arguments",
        "{}",
    )

    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError:
        arguments = None

    if (
        isinstance(arguments, dict)
        and tool is not None
        and (
            tool.id in {"edit", "read", "write"}
            or isinstance(tool, MemoryTool)
        )
    ):
        preview = tool.format_call_log(arguments)
    elif isinstance(arguments, dict):
        preview = argument_preview(
            arguments
        )
    else:
        preview = raw[:50]

    shown = (
        name
        .replace("_", " ")
        .title()
    )

    line = Text()

    line.append(
        "⏺ ",
        style="green",
    )

    line.append(
        shown,
        style="green bold",
    )

    if "\n" in preview:
        line.append("\n")
        line.append(preview, style="dim")
    else:
        line.append("(")
        line.append(preview, style="dim")
        line.append(")")

    console.print()
    console.print(line)

    return arguments


def render_tool_call_result(
    result: str,
    tool: Tool | None = None,
) -> None:
    """Handle render tool call result."""
    line = Text()

    line.append(
        "  ⎿  ",
        style="dim",
    )

    shown = (
        tool.format_result_log(result)
        if isinstance(tool, MemoryTool)
        else result_preview(result)
    )
    line.append(shown, style="dim")

    console.print(line)


def render_assistant_text(
    text: str,
) -> None:
    """Handle render assistant text."""
    if not text.strip():
        return

    console.print(
        Text(
            "⏺",
            style="cyan",
        )
    )

    console.print(
        Markdown(text)
    )


def render_memory_change(
    tools: dict[str, Tool],
    before: str | None,
) -> None:
    """Handle render memory change."""
    after = build_memory_context(
        tools
    )

    if after == before:
        return

    content = (
        after
        or "# Conversation Memory\n\n(empty)"
    )

    console.print()

    console.print(
        Text(
            content,
            style="dim",
        )
    )


def memory_tool_for_call(
    tools: dict[str, Tool],
    tool_call: ChatCompletionMessageFunctionToolCallParam,
) -> MemoryTool[Any] | None:
    """Handle memory tool for call."""
    name = tool_call[
        "function"
    ].get(
        "name"
    )

    tool = (
        tools.get(name)
        if name
        else None
    )

    return (
        tool
        if isinstance(
            tool,
            MemoryTool,
        )
        else None
    )


def print_header(
    config: CitraConfig,
    workspace: Path,
) -> None:
    """Handle print header."""
    line = Text()

    line.append(
        "citra",
        style="bold",
    )

    line.append(
        " | ",
        style="dim",
    )

    line.append(
        config.model().id,
        style="dim",
    )

    line.append(
        " | ",
        style="dim",
    )

    line.append(
        str(workspace),
        style="dim",
    )

    console.print(line)
    console.print()
