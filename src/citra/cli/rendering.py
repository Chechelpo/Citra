"""Professional, semantic terminal rendering for Citra's agentic CLI."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

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
from ..tools.tool import Tool
from ..tools.tool_group import GenericToolGroup, ToolGroup
from .theme import console
from .input import terminal_ui_state

if TYPE_CHECKING:
    from ..commands.command import CommandResult, CommandUsage

_MAX_PANEL_WIDTH = 120
_FILE_LIST_TOOLS = {"read", "glob", "tree"}
_DIFF_TOOLS = {"edit", "apply_patch"}
_SHELL_TOOLS = {"bash", "subprocess"}
_MAX_FALLBACK_PREVIEW_LENGTH = 1_000
_DIFF_HUNK = re.compile(
    r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@"
)


@dataclass(frozen=True)
class PreparedToolCall:
    """Hold presentation data captured before a tool mutates its target."""

    arguments: dict[str, Any] | None
    preview: str


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
            f"workspace: {self.workspace_directory}"
        )


@dataclass(frozen=True)
class CliSessionLayout:
    """Centralize static header and footer objects for one rendered session."""

    header: SessionHeader
    footer: SessionFooter


@dataclass
class ToolCallRenderState:
    """Keep adjacent tool calls under one activity heading for a single run."""

    active_group: type[ToolGroup] | None = None
    active_arguments: dict[str, Any] | None = None
    prepared_calls: dict[str, PreparedToolCall] | None = None

    def prepare_call(
        self,
        tool_call: ToolCall,
        tool: Tool | None,
    ) -> None:
        """Capture an invocation preview before the tool changes any files."""
        if self.prepared_calls is None:
            self.prepared_calls = {}
        arguments, preview = _prepare_tool_call(tool_call, tool)
        self.prepared_calls[tool_call.id] = PreparedToolCall(arguments, preview)

    def render_batch(
        self,
        calls: Iterable[tuple[ToolCall, Tool | None, str]],
    ) -> None:
        """Render a completed assistant tool-call batch in execution order."""
        materialized = list(calls)
        index = 0
        while index < len(materialized):
            edit_calls: list[tuple[ToolCall, Tool | None, str]] = []
            while index < len(materialized):
                tool_call, tool, result = materialized[index]
                prepared = self._prepared(tool_call)
                semantic_name = str(getattr(tool, "id", tool_call.name)).lower()
                if (
                    semantic_name != "edit"
                    or not result.startswith("ok")
                    or prepared is None
                    or not prepared.preview.startswith("--- ")
                ):
                    break
                edit_calls.append(materialized[index])
                index += 1
            if edit_calls:
                self._render_edit_batch(edit_calls)
                self.active_group = None
                continue

            tool_call, tool, result = materialized[index]
            self.render_start(tool_call, tool)
            self.render_result(result, tool)
            index += 1
        if self.prepared_calls is not None:
            for tool_call, _tool, _result in materialized:
                self.prepared_calls.pop(tool_call.id, None)

    def _prepared(self, tool_call: ToolCall) -> PreparedToolCall | None:
        if self.prepared_calls is None:
            return None
        return self.prepared_calls.get(tool_call.id)

    def _render_edit_batch(
        self,
        calls: list[tuple[ToolCall, Tool | None, str]],
    ) -> None:
        """Render successful edits as one contextual, line-numbered diff tree."""
        entries: list[tuple[str, str, str, int, int]] = []
        for tool_call, _tool, result in calls:
            prepared = self._prepared(tool_call)
            assert prepared is not None
            path = _argument_path(prepared.arguments) or tool_call.name
            added, deleted = _diff_line_counts(prepared.preview)
            entries.append((path, prepared.preview, result, added, deleted))

        total_added = sum(entry[3] for entry in entries)
        total_deleted = sum(entry[4] for entry in entries)
        file_count = len({entry[0] for entry in entries})
        noun = "file" if file_count == 1 else "files"
        console.print()
        console.print(
            Text.assemble(
                ("• ", "citra.tool"),
                (f"Edited {file_count} {noun} ", "citra.tool"),
                (f"(+{total_added} -{total_deleted})", "citra.muted"),
            )
        )
        for path, preview, result, added, deleted in entries:
            console.print(
                Text.assemble(
                    ("  └ ", "citra.border"),
                    (path, "citra.path"),
                    (f" (+{added} -{deleted})", "citra.muted"),
                )
            )
            console.print(_compact_diff(preview))
            diagnostics = result.removeprefix("ok").strip()
            receipt = f"ok (+{added}, -{deleted})"
            if diagnostics:
                receipt += f"\n{diagnostics}"
            _render_result_receipt(receipt, nested=True)

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
        prepared = self._prepared(tool_call)
        self.active_arguments = render_tool_call_start(
            tool_call,
            tool,
            nested=True,
            prepared=prepared,
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
) -> type[ToolGroup]:
    """Resolve the group registered by the tool's package."""
    return (
        ToolGroup.for_tool(tool)
        or ToolGroup.for_name(str(getattr(tool, "id", tool_call.name)))
        or GenericToolGroup
    )


def render_tool_group_heading(group: type[ToolGroup]) -> None:
    """Start a semantic tool activity group in the terminal."""
    console.print()
    console.print(Text.assemble(("• ", "citra.tool"), (group.GROUP_NAME, "citra.tool")))


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


def _safe_call_preview(tool: Tool | None, arguments: Any) -> str:
    """A presentation hook must never prevent the tool itself from running."""
    if tool is None:
        return _fallback_call_preview(dict(arguments))
    try:
        group = ToolGroup.for_tool(tool) or GenericToolGroup
        return group.format_call(tool, arguments)
    except Exception:  # noqa: BLE001 - third-party tool presentation hook
        return _fallback_call_preview(dict(arguments))


def _prepare_tool_call(
    tool_call: ToolCall,
    tool: Tool | None,
) -> tuple[dict[str, Any] | None, str]:
    """Build a presentation preview from already parsed arguments."""
    typed_arguments = tool_call.arguments
    arguments = typed_arguments.to_dict()
    if tool is None:
        preview = _fallback_call_preview(arguments)
        return None, preview[:_MAX_FALLBACK_PREVIEW_LENGTH]
    return arguments, _safe_call_preview(tool, typed_arguments)


def _diff_line_counts(diff: str) -> tuple[int, int]:
    """Count created and deleted content lines in a unified diff."""
    added = 0
    deleted = 0
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith("@@ "):
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            added += 1
        elif in_hunk and line.startswith("-"):
            deleted += 1
    return added, deleted


def _compact_diff(diff: str) -> Text:
    """Convert a unified diff into compact numbered contextual hunks."""
    rendered = Text()
    old_line = 0
    new_line = 0
    seen_hunk = False
    for line in diff.splitlines():
        match = _DIFF_HUNK.match(line)
        if match is not None:
            if seen_hunk:
                rendered.append("   ⋮\n", style="citra.muted")
            old_line = int(match.group(1))
            new_line = int(match.group(2))
            seen_hunk = True
            continue
        if not seen_hunk or line.startswith(("--- ", "+++ ")):
            continue
        marker = line[:1]
        content = line[1:]
        if marker == "-":
            rendered.append(f"{old_line:>4} -{content}\n", style="citra.diff.deleted")
            old_line += 1
        elif marker == "+":
            rendered.append(f"{new_line:>4} +{content}\n", style="citra.diff.added")
            new_line += 1
        elif marker == " ":
            rendered.append(f"{new_line:>4}  {content}\n")
            old_line += 1
            new_line += 1
        elif marker == "\\":
            rendered.append(f"     {line}\n", style="citra.muted")
    rendered.rstrip()
    return rendered


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
    if name == "edit":
        old = str((arguments or {}).get("old", ""))
        new = str((arguments or {}).get("new", ""))
        fragment = "\n".join(
            ("@@ -1 +1 @@",)
            + tuple(f"-{line}" for line in old.splitlines())
            + tuple(f"+{line}" for line in new.splitlines())
        )
        added, deleted = _diff_line_counts(fragment)
        summary = f"ok (+{added}, -{deleted})"
    else:
        summary = f"Wrote {path}" if path else "Wrote"
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
    prepared: PreparedToolCall | None = None,
) -> dict[str, Any] | None:
    """Render one tool invocation and return decoded object arguments when valid."""
    name = tool_call.name
    if prepared is None:
        arguments, preview = _prepare_tool_call(tool_call, tool)
    else:
        arguments = prepared.arguments
        preview = prepared.preview

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
    if tool is None:
        shown = result_preview(result)
    elif not isinstance(tool, Tool):
        shown = tool_result_preview(str(getattr(tool, "id", "unknown")), result, arguments)
    else:
        group = ToolGroup.for_tool(tool) or GenericToolGroup
        shown = group.format_result(tool, result, arguments)
    shown = shown.strip() or "(empty)"
    _render_result_receipt(shown, nested=nested)


def _render_result_receipt(shown: str, *, nested: bool) -> None:
    """Render a result receipt with aligned continuation lines."""
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
    """Replace the bounded latest-request diagnostics in the composer."""
    terminal_ui_state.record_model_debug(message)


def clear_model_debug() -> None:
    """Clear transient model diagnostics from the composer."""
    terminal_ui_state.clear_model_debug()


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
        # Command output often contains absolute paths and shell snippets.  It
        # is still parsed as Markdown, but Rich must not inject hard newlines
        # into copyable values merely because the current terminal is narrow.
        console.print(Markdown(output, code_theme="monokai"), soft_wrap=True)


def render_command_result(result: CommandResult) -> None:
    """Render every presentational field of a command result in order."""
    render_command_output(result.output)
    render_command_usage(result.usage)


def render_command_usage(usages: tuple[CommandUsage, ...]) -> None:
    """Render declared slash-command forms and arguments as a Rich tree."""
    if not usages:
        return
    tree = Tree(
        "Commands" if len(usages) > 1 else "Usage",
        guide_style="citra.border",
    )
    for usage in usages:
        command_node = tree.add(
            Text.assemble(
                (f"/{usage.command}", "citra.accent"),
                (f" — {usage.description}", "default"),
            )
        )
        for form in usage.forms:
            suffix = form.suffix
            label = f"/{usage.command}{f' {suffix}' if suffix else ''}"
            form_node = command_node.add(
                Text.assemble(
                    (label, "citra.tool"),
                    (
                        (f" — {form.description}", "citra.muted")
                        if form.description
                        else ("", "")
                    ),
                )
            )
            for option in form.options:
                form_node.add(
                    Text.assemble(
                        (option.syntax, "citra.accent"),
                        (f" — {option.description}", "citra.muted"),
                    )
                )
            for argument in form.arguments:
                form_node.add(
                    Text.assemble(
                        (argument.syntax, "citra.accent"),
                        (f" — {argument.description}", "citra.muted"),
                    )
                )
    console.print()
    console.print(tree)


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
