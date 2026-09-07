"""Professional, semantic terminal rendering for Citra's agentic CLI."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from openai.types.chat import ChatCompletionMessageFunctionToolCallParam
from rich import box
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from ..context import CitraConfig
from ..tools.tool import Tool
from .theme import console

_MAX_PANEL_WIDTH = 120


def _panel_width() -> int:
    """Keep cards readable instead of stretching across ultrawide terminals."""
    return min(console.width, _MAX_PANEL_WIDTH)


def format_elapsed(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, seconds = divmod(total, 60)
    return f"{minutes}m {seconds}s" if minutes else f"{seconds}s"


@contextmanager
def working_animation(label: str = "Working") -> Iterator[None]:
    """Show a quiet live spinner, then retain a compact duration receipt."""
    started = perf_counter()
    with Status(
        Text.assemble(("● ", "citra.accent"), (f"{label}…", "citra.muted")),
        console=console,
        spinner="dots12",
        spinner_style="citra.accent",
        refresh_per_second=12.5,
    ):
        yield
    console.print(
        Text.assemble(
            ("• ", "citra.accent"),
            (f"Worked for {format_elapsed(perf_counter() - started)}", "citra.muted"),
        )
    )


def render_markdown(text: str) -> None:
    console.print(Markdown(text, code_theme="monokai"))


def argument_preview(arguments: dict[str, Any], limit: int = 50) -> str:
    if not arguments:
        return ""
    value = str(next(iter(arguments.values()))).replace("\n", " ")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def result_preview(result: str, line_limit: int = 72) -> str:
    lines = result.splitlines()
    if not lines:
        return "(empty)"
    preview = lines[0][:line_limit]
    if len(lines) > 1:
        preview += f"  … +{len(lines) - 1} lines"
    elif len(lines[0]) > line_limit:
        preview += "…"
    return preview


def _tool_details(name: str, preview: str) -> object:
    """Choose a readable Rich primitive for structured tool details."""
    if not preview:
        return Text("No arguments", style="citra.muted")
    if name == "edit" or preview.lstrip().startswith(("--- ", "*** ")):
        return Syntax(preview, "diff", theme="monokai", word_wrap=True, padding=(0, 1))
    if name == "write":
        return Syntax(preview, "text", theme="monokai", word_wrap=True, padding=(0, 1))
    if "\n" in preview:
        tree = Tree("files", guide_style="citra.border", hide_root=True)
        for line in preview.splitlines():
            tree.add(Text(line, style="citra.muted"))
        return tree
    return Text(preview, style="citra.muted", overflow="ellipsis")


def render_tool_call_start(
    tool_call: ChatCompletionMessageFunctionToolCallParam,
    tool: Tool | None = None,
) -> dict[str, Any] | None:
    function = tool_call["function"]
    name = str(function.get("name", "unknown"))
    raw = str(function.get("arguments", "{}"))
    try:
        decoded = json.loads(raw)
        arguments = decoded if isinstance(decoded, dict) else None
    except json.JSONDecodeError:
        arguments = None
    if arguments is not None and tool is not None:
        preview = tool.format_call_log(arguments)
    elif arguments is not None:
        preview = argument_preview(arguments)
    else:
        preview = raw[:72]

    title = name.replace("_", " ").capitalize()
    console.print()
    console.print(Text.assemble(("• ", "citra.tool"), (title, "citra.tool")))
    semantic_name = str(getattr(tool, "id", name)).lower()
    console.print(_tool_details(semantic_name, preview))
    return arguments


def render_tool_call_result(result: str, tool: Tool | None = None) -> None:
    shown = (
        tool.format_result_log(result) if tool is not None else result_preview(result)
    )
    console.print(Text.assemble(("  └ ", "citra.border"), (shown, "citra.muted")))


def render_assistant_text(text: str) -> None:
    if text.strip():
        console.print()
        console.print(Text("•", style="citra.accent"))
        console.print(Markdown(text, code_theme="monokai"))


def render_notice(message: str, *, level: str = "info") -> None:
    styles = {
        "info": "citra.accent",
        "success": "citra.success",
        "warning": "citra.warning",
        "error": "citra.error",
    }
    icons = {"info": "•", "success": "✓", "warning": "!", "error": "×"}
    style = styles.get(level, styles["info"])
    console.print(
        Text.assemble((f"{icons.get(level, '•')} ", style), (message, "default"))
    )


def render_command_output(output: str) -> None:
    """Render command results without flattening Markdown, tables, or diffs."""
    if not output.strip():
        return
    console.print()
    if output.lstrip().startswith(("diff --git", "--- ")):
        console.print(Syntax(output.rstrip(), "diff", theme="monokai", word_wrap=True))
    else:
        console.print(Markdown(output, code_theme="monokai"))


def render_question(question: str, options: tuple[str, ...] | list[str] = ()) -> None:
    body: list[object] = [Text(question, style="bold")]
    if options:
        choices = Table.grid(padding=(0, 1))
        choices.add_column(style="citra.accent", justify="right")
        choices.add_column()
        for index, option in enumerate(options, 1):
            choices.add_row(f"{index}.", option)
        body.extend(
            [
                Text(),
                choices,
                Text("Type a number or a free-form answer.", style="citra.muted"),
            ]
        )
    else:
        body.append(Text("Open-ended question", style="citra.muted"))
    console.print(
        Panel(
            Group(*body),
            border_style="citra.border",
            box=box.ROUNDED,
            padding=(1, 2),
            width=_panel_width(),
        )
    )


def render_workflow_picker(workflows: list[tuple[str, str, bool]]) -> None:
    table = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    table.add_column(style="citra.accent", justify="right", width=3)
    table.add_column()
    table.add_column(style="citra.muted", ratio=2)
    table.add_column(style="citra.success", justify="right")
    for index, (name, description, default) in enumerate(workflows, 1):
        table.add_row(f"{index}.", name, description, "default" if default else "")
    console.print(
        Panel(
            table,
            title="[citra.brand]Choose a workflow[/]",
            border_style="citra.border",
            box=box.ROUNDED,
            width=_panel_width(),
        )
    )


def print_header(config: CitraConfig, workspace: Path) -> None:
    model = config.model().id
    grid = Table.grid(expand=True)
    grid.add_column(min_width=18)
    grid.add_column(justify="right", style="citra.muted", ratio=1)
    grid.add_row(Text("CITRA", style="citra.brand"), "agentic coding workspace")
    grid.add_row(Text(model, style="citra.accent"), str(workspace))
    console.print(
        Panel(
            grid,
            border_style="citra.border",
            box=box.ROUNDED,
            padding=(0, 1),
            width=_panel_width(),
        )
    )
