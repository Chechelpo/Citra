"""Professional, semantic terminal rendering for Citra's agentic CLI."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any

from rich import box
from rich.console import Group, RenderableType
from citra.sandbox import SandboxMode
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from ..agent.chat_message import ToolCall
from ..tools.tool import InvalidToolArguments, Tool
from .theme import console
from .input import terminal_ui_state

_MAX_PANEL_WIDTH = 120
_FILE_LIST_TOOLS = {"read", "glob", "tree"}
_DIFF_TOOLS = {"edit", "apply_patch"}
_SHELL_TOOLS = {"bash", "subprocess"}
_MAX_FALLBACK_PREVIEW_LENGTH = 1_000


@dataclass(frozen=True)
class SessionHeader:
    """Represent stable connection, workflow, and sandbox facts for a session."""

    workflow: str
    connection: str
    sandbox_mode: SandboxMode
    workspace: Path

    def to_header(self) -> str:
        """Return the stable session facts rendered inside the header box."""
        return "\n".join(
            (
                "CITRA",
                f"connection:   {self.connection}",
                f"workflow:     {self.workflow}",
                f"sandbox mode: {self.sandbox_mode.name}",
                f"workspace:    {self.workspace}",
            )
        )


@dataclass(frozen=True)
class SessionFooter:
    """Represent the static environment details beneath the prompt composer."""

    model_selection: str
    source_directory: str
    workspace_directory: str
    process_name: str

    def render(self) -> str:
        """
        Return the compact, single-line footer shown for every prompt.

        returns:
        ``` text
        model: <model_profile> (<model_id>) | <source> -> <workspace_directory>
        ```

        """
        return (
            f"model: {self.model_selection}  ·  source: {self.source_directory}  ·  "
            f"workspace: {self.workspace_directory}  ·  process: {self.process_name}"
        )


@dataclass(frozen=True)
class CliSessionLayout:
    """Centralize static header and footer objects for one rendered session."""

    header: SessionHeader
    footer: SessionFooter


class ToolCallGroup(Enum):
    """Describe the user-facing activity represented by a tool call."""

    EXPLORED = "Explored"
    CHANGED = "Changed"
    RAN_COMMANDS = "Ran commands"
    UPDATED_MEMORY = "Updated memory"
    USED_TOOLS = "Used tools"


# Tool identities assigned to the same activity category share one heading.
_TOOL_GROUPS: dict[str, ToolCallGroup] = {
    **dict.fromkeys(
        (
            "browser",
            "find",
            "glob",
            "grep",
            "lsp",
            "read",
            "read_image",
            "tree",
            "web-search",
            "web_search",
        ),
        ToolCallGroup.EXPLORED,
    ),
    **dict.fromkeys(
        ("apply_patch", "diagram", "document", "edit", "write"),
        ToolCallGroup.CHANGED,
    ),
    **dict.fromkeys(
        ("bash", "commit", "git", "python", "subprocess"),
        ToolCallGroup.RAN_COMMANDS,
    ),
    **dict.fromkeys(
        (
            "acceptance_criteria",
            "change",
            "checkpoint",
            "constraint",
            "decision",
            "fact",
            "issue",
            "requirement",
            "scope",
            "todo",
            "verification",
            "working_state",
        ),
        ToolCallGroup.UPDATED_MEMORY,
    ),
}


@dataclass
class ToolCallRenderState:
    """Keep adjacent tool calls under one activity heading for a single run."""

    active_group: ToolCallGroup | None = None
    active_arguments: dict[str, Any] | None = None

    def render_batch(
        self,
        calls: Iterable[tuple[ToolCall, Tool | None, str]],
    ) -> None:
        """Render a completed assistant tool-call batch in execution order."""
        for tool_call, tool, result in calls:
            self.render_start(tool_call, tool)
            self.render_result(result, tool)

    def render_start(
        self,
        tool_call: ToolCall,
        tool: Tool | None = None,
    ) -> dict[str, Any] | None:
        """Render a grouped call and return its decoded argument object, if valid."""
        group = tool_call_group(tool_call, tool)
        if group is not self.active_group:
            render_tool_group_heading(group)
            self.active_group = group
        self.active_arguments = render_tool_call_start(
            tool_call,
            tool,
            nested=True,
        )
        return self.active_arguments

    def render_result(self, result: str, tool: Tool | None = None) -> None:
        """Render a result aligned beneath the most recently started grouped call."""
        render_tool_call_result(
            result,
            tool,
            arguments=self.active_arguments,
            nested=True,
        )


def _panel_width() -> int:
    """Keep cards readable instead of stretching across ultrawide terminals."""
    return min(console.width, _MAX_PANEL_WIDTH)


def tool_call_group(
    tool_call: ToolCall,
    tool: Tool | None = None,
) -> ToolCallGroup:
    """Map a model-facing call to its semantic UI activity category."""
    model_name = tool_call.name
    semantic_name = str(getattr(tool, "id", model_name)).lower()
    return _TOOL_GROUPS.get(semantic_name, ToolCallGroup.USED_TOOLS)


def render_tool_group_heading(group: ToolCallGroup) -> None:
    """Start a semantic tool activity group in the terminal."""
    console.print()
    console.print(Text.assemble(("• ", "citra.tool"), (group.value, "citra.tool")))


def format_elapsed(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, seconds = divmod(total, 60)
    return f"{minutes}m {seconds}s" if minutes else f"{seconds}s"


@contextmanager
def working_animation(label: str = "Working") -> Iterator[None]:
    """Expose model activity in the stable composer instead of a live Rich refresh."""
    terminal_ui_state.begin_working(label)
    try:
        yield
    finally:
        terminal_ui_state.finish_working()


def render_markdown(text: str) -> None:
    console.print(Markdown(text, code_theme="monokai"))


def argument_preview(arguments: dict[str, Any], limit: int = 50) -> str:
    if not arguments:
        return ""
    value = str(next(iter(arguments.values()))).replace("\n", " ")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _fallback_call_preview(arguments: dict[str, Any]) -> str:
    """Keep unknown tools useful without dumping an unreadable Python repr."""
    if not arguments:
        return ""
    rendered = json.dumps(arguments, ensure_ascii=False, indent=2, default=str)
    if len(rendered) <= _MAX_FALLBACK_PREVIEW_LENGTH:
        return rendered
    omitted = len(rendered) - _MAX_FALLBACK_PREVIEW_LENGTH
    return (
        rendered[:_MAX_FALLBACK_PREVIEW_LENGTH]
        + f"\n… {omitted} characters omitted"
    )


def _safe_call_preview(tool: Tool | None, arguments: dict[str, Any]) -> str:
    """A presentation hook must never prevent the tool itself from running."""
    if tool is None:
        return _fallback_call_preview(arguments)
    try:
        return str(tool.format_call_log(arguments))
    except Exception:  # noqa: BLE001 - third-party tool presentation hook
        return _fallback_call_preview(arguments)


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


def _mutation_result_preview(
    name: str,
    result: str,
    arguments: dict[str, Any] | None,
) -> str:
    """Describe a file mutation using its target and post-edit check output."""
    if not result.startswith("ok"):
        return result_preview(result)

    path = _argument_path(arguments)
    verb = "Wrote" if name == "write" else "Updated"
    summary = f"{verb} {path}" if path else verb
    diagnostics = result.removeprefix("ok").strip()
    if not diagnostics:
        return summary
    return f"{summary}\n{diagnostics}"


def _argument_path(arguments: dict[str, Any] | None) -> str:
    """Return the first supported path spelling from decoded call arguments."""
    if arguments is None:
        return ""
    for key in ("path", "file_path", "filename", "filePath"):
        value = arguments.get(key)
        if isinstance(value, str):
            return value
    return ""


def _exploration_result_preview(name: str, result: str) -> str:
    """Summarize discovery results without hiding empty-result semantics."""
    if not result or result in {"none", "no matches", "(empty)"}:
        return "No results"
    line_count = len(result.splitlines())
    if name == "read":
        return f"Read {line_count} line(s) · {len(result)} chars"
    if name in {"find", "glob", "grep"}:
        return f"Found {line_count} match(es)"
    return result_preview(result)


def _command_result_preview(result: str) -> str:
    """Summarize foreground command completion and retain failure status."""
    if not result or result == "(empty)":
        return "Completed · no output"
    if result == "ok" or result.startswith("Started subprocess "):
        return result
    if result.startswith(("error:", "permission-denied:", "cancelled:")):
        return result_preview(result)
    lines = result.splitlines()
    final_line = lines[-1]
    if final_line.startswith("(exit code "):
        return f"Failed {final_line} · {max(0, len(lines) - 1)} output line(s)"
    if final_line.startswith("(timed out after "):
        return f"Timed out · {max(0, len(lines) - 1)} output line(s)"
    return f"Completed · {len(lines)} output line(s)"


def tool_result_preview(
    name: str,
    result: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Build the terminal result receipt independently from legacy log hooks."""
    normalized_name = name.lower()
    if normalized_name in _DIFF_TOOLS | {"write"}:
        return _mutation_result_preview(normalized_name, result, arguments)
    if normalized_name in {
        "find",
        "glob",
        "grep",
        "read",
        "tree",
        "web-search",
        "web_search",
    }:
        return _exploration_result_preview(normalized_name, result)
    if normalized_name in _SHELL_TOOLS | {"python"}:
        return _command_result_preview(result)
    return result_preview(result)


def _tool_details(name: str, preview: str) -> RenderableType:
    """Choose a readable Rich primitive for structured tool details."""
    if not preview:
        return Text("No arguments", style="citra.muted")
    if name in _DIFF_TOOLS or preview.lstrip().startswith(("--- ", "*** ")):
        return Syntax(preview, "diff", theme="monokai", word_wrap=True, padding=(0, 1))
    if name == "write":
        return Syntax(preview, "text", theme="monokai", word_wrap=True, padding=(0, 1))
    if name in _SHELL_TOOLS:
        return Syntax(preview, "shell", theme="monokai", word_wrap=True, padding=(0, 1))
    if name in _FILE_LIST_TOOLS and "\n" in preview:
        tree = Tree("paths", guide_style="citra.border", hide_root=True)
        for line in preview.splitlines():
            tree.add(Text(line, style="citra.muted"))
        return tree
    if "\n" in preview:
        return Text(preview, style="citra.muted", overflow="fold")
    return Text(preview, style="citra.muted", overflow="ellipsis")


def render_tool_call_start(
    tool_call: ToolCall,
    tool: Tool | None = None,
    *,
    nested: bool = False,
) -> dict[str, Any] | None:
    """Render one tool invocation and return decoded object arguments when valid."""
    name = tool_call.name
    raw = tool_call.arguments
    arguments: dict[str, Any] | None = None
    if tool is None:
        preview = raw[:_MAX_FALLBACK_PREVIEW_LENGTH]
    else:
        try:
            arguments = tool.parse_arguments(raw)
            preview = _safe_call_preview(tool, arguments)
        except InvalidToolArguments as error:
            preview = str(error)

    title = name.replace("_", " ").capitalize()
    if not nested:
        console.print()
    prefix = "  └ " if nested else "• "
    console.print(Text.assemble((prefix, "citra.tool"), (title, "citra.tool")))
    semantic_name = str(getattr(tool, "id", name)).lower()
    details = _tool_details(semantic_name, preview)
    console.print(Padding(details, (0, 0, 0, 4)) if nested else details)
    return arguments


def render_tool_call_result(
    result: str,
    tool: Tool | None = None,
    *,
    arguments: dict[str, Any] | None = None,
    nested: bool = False,
) -> None:
    """Render a compact, safely formatted receipt for a completed tool call."""
    semantic_name = str(getattr(tool, "id", "unknown"))
    shown = tool_result_preview(semantic_name, result, arguments).strip() or "(empty)"
    lines = shown.splitlines()
    indent = "    " if nested else "  "
    receipt = Text.assemble((f"{indent}└ ", "citra.border"), (lines[0], "citra.muted"))
    continuation = "\n".join(f"{indent}  {line}" for line in lines[1:])
    if continuation:
        receipt.append(f"\n{continuation}", style="citra.muted")
    console.print(receipt)


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


def render_model_debug(message: str) -> None:
    """Render optional model-request diagnostics without disrupting live output."""
    console.print(
        Text.assemble(("  · ", "citra.border"), (message, "citra.muted"))
    )


def render_model_retry(
    reason: str,
    *,
    delay: float,
    attempt: int,
    max_attempts: int,
) -> None:
    """Render the reason and schedule for a model-request retry."""
    render_notice(
        f"Model request {reason}. Retrying in {delay:.1f}s "
        f"(attempt {attempt}/{max_attempts})…",
        level="warning",
    )


def render_model_warning(message: str) -> None:
    """Render a recoverable model-provider warning."""
    render_notice(message, level="warning")


def render_model_error(message: str) -> None:
    """Render a model-provider failure."""
    render_notice(message, level="error")


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
    body: list[RenderableType] = [Text(question, style="bold")]
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


def print_header(header: SessionHeader) -> None:
    """Render immutable session connection, workflow, and sandbox information."""
    console.print(
        Panel(
            Text(header.to_header()),
            subtitle="agentic coding workspace",
            border_style="citra.border",
            box=box.ROUNDED,
            padding=(0, 1),
            width=_panel_width(),
        )
    )
