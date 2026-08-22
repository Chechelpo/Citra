"""Terminal rendering kept separate from agent orchestration."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from openai.types.chat import ChatCompletionMessageFunctionToolCallParam

from ..context import CitraConfig
from ..tools.session_memory import MemoryTool
from ..tools.tool import Tool
from ..utils.chat_completions_api import build_memory_context
from ..utils.terminal import BOLD, CYAN, DIM, GREEN, RESET


def render_markdown(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*", f"{BOLD}\\1{RESET}", text)


def argument_preview(arguments: dict[str, Any], limit: int = 50) -> str:
    if not arguments:
        return ""
    text = str(next(iter(arguments.values()))).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def result_preview(result: str, line_limit: int = 60) -> str:
    lines = result.splitlines()
    if not lines:
        return "(empty)"

    preview = lines[0][:line_limit]
    if len(lines) > 1:
        preview += f" ... +{len(lines) - 1} lines"
    elif len(lines[0]) > line_limit:
        preview += "..."
        
    return preview


def render_tool_call_start(
    tool_call: ChatCompletionMessageFunctionToolCallParam,
) -> dict[str, Any] | None:
    function = tool_call["function"]
    name = function.get("name", "unknown")
    raw = function.get("arguments", "{}")
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError:
        arguments = None
    preview = argument_preview(arguments) if isinstance(arguments, dict) else raw[:50]
    shown = name.replace("_", " ").title()
    print(f"\n{GREEN}⏺ {shown}{RESET}({DIM}{preview}{RESET})")
    return arguments


def render_tool_call_result(result: str) -> None:
    print(f"  {DIM}⎿  {result_preview(result)}{RESET}")


def render_assistant_text(text: str) -> None:
    print(f"\n{CYAN}⏺{RESET} {render_markdown(text)}")


def render_memory_change(tools: dict[str, Tool], before: str | None) -> None:
    after = build_memory_context(tools)
    if after == before:
        return
    print(f"\n{DIM}{after or '# Conversation Memory\n\n(empty)'}{RESET}")


def memory_tool_for_call(
    tools: dict[str, Tool],
    tool_call: ChatCompletionMessageFunctionToolCallParam,
) -> MemoryTool[Any] | None:
    name = tool_call["function"].get("name")
    tool = tools.get(name) if name else None
    return tool if isinstance(tool, MemoryTool) else None


def print_header(config: CitraConfig, source_workspace: Path) -> None:
    print(
        f"{BOLD}citra{RESET} | {DIM}{config.model.id} | "
        f"{source_workspace}{RESET}\n"
    )

