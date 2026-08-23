"""Terminal rendering kept separate from agent orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletionMessageFunctionToolCallParam
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from ..context import CitraConfig
from ..tools.session_memory import MemoryTool
from ..tools.tool import Tool
from ..utils.chat_completions_api import build_memory_context


console = Console(
    highlight=False,
    soft_wrap=True,
)


def render_markdown(text: str) -> None:
    console.print(
        Markdown(text)
    )


def argument_preview(
    arguments: dict[str, Any],
    limit: int = 50,
) -> str:
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
) -> dict[str, Any] | None:
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

    if isinstance(arguments, dict):
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

    line.append("(")

    line.append(
        preview,
        style="dim",
    )

    line.append(")")

    console.print()
    console.print(line)

    return arguments


def render_tool_call_result(
    result: str,
) -> None:
    line = Text()

    line.append(
        "  ⎿  ",
        style="dim",
    )

    line.append(
        result_preview(result),
        style="dim",
    )

    console.print(line)


def render_assistant_text(
    text: str,
) -> None:
    console.print()

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
    source_workspace: Path,
) -> None:
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
        str(source_workspace),
        style="dim",
    )

    console.print(line)
    console.print()