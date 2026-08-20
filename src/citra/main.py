#!/usr/bin/env python3
from __future__ import annotations

"""
Citra - minimal agentic coding assistant.
"""

from citra.context.workspace import WorkspaceContext

import json
import random
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any
import os

from .agent import AgentSession
from .commands import COMMAND_REGISTRY
from .context import ExecutionContext
from .tools.default_registry import TOOL_REGISTRY
from .tools.tool import Tool
from .utils.api import chat_completions_url
from .utils.prompt import build_system_prompt
from .utils.terminal import (
    BLUE,
    BOLD,
    CYAN,
    DIM,
    GREEN,
    RED,
    RESET,
    YELLOW,
    separator,
)
from .utils.terminal_input import terminal_input
from .tools.session_memory import MemoryTool

_CANCELLED_BY_STEERING = (
    "cancelled: user steering instructions were received "
    "before this tool call executed"
)


# ---------------------------------------------------------------------------
# Terminal UI
# ---------------------------------------------------------------------------


def render_markdown(text: str) -> str:
    """
    Minimal terminal Markdown rendering.

    Currently only renders **bold**.
    """
    return re.sub(
        r"\*\*(.+?)\*\*",
        f"{BOLD}\\1{RESET}",
        text,
    )


def tool_name_for_display(
    name: str,
) -> str:
    return name.replace(
        "_",
        " ",
    ).title()


def argument_preview(
    arguments: dict[str, Any],
    limit: int = 50,
) -> str:
    """
    Produce NanoCode-style compact tool-call previews.
    """
    if not arguments:
        return ""

    value = next(
        iter(arguments.values())
    )

    text = str(value).replace(
        "\n",
        " ",
    )

    if len(text) > limit:
        return (
            text[:limit]
            + "..."
        )

    return text


def result_preview(
    result: str,
    line_limit: int = 60,
) -> str:
    """
    Produce a compact tool-result preview.
    """
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


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def merge_consecutive_roles(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge adjacent plain-text messages with the same role.

    Protocol-bearing messages such as assistant tool calls and tool
    results are preserved exactly.
    """

    merged: list[dict[str, Any]] = []

    for message in messages:
        if not merged:
            merged.append(
                dict(message)
            )
            continue

        previous = merged[-1]

        if not _messages_are_mergeable(
            previous,
            message,
        ):
            merged.append(
                dict(message)
            )
            continue

        previous_content = previous.get(
            "content"
        )

        current_content = message.get(
            "content"
        )

        if not previous_content:
            previous[
                "content"
            ] = current_content
            continue

        if not current_content:
            continue

        previous[
            "content"
        ] = (
            f"{previous_content}\n\n"
            f"{current_content}"
        )

    return merged


def _messages_are_mergeable(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    role = first.get(
        "role"
    )

    if role != second.get(
        "role"
    ):
        return False

    if role not in {
        "system",
        "user",
        "assistant",
    }:
        return False

    # Only merge ordinary role/content messages. Anything carrying
    # protocol metadata must remain structurally intact.
    if set(first) - {
        "role",
        "content",
    }:
        return False

    if set(second) - {
        "role",
        "content",
    }:
        return False

    first_content = first.get(
        "content"
    )

    second_content = second.get(
        "content"
    )

    if (
        first_content is not None
        and not isinstance(
            first_content,
            str,
        )
    ):
        return False

    if (
        second_content is not None
        and not isinstance(
            second_content,
            str,
        )
    ):
        return False

    return True

def system_prompt(
    context: ExecutionContext,
) -> str:
    return build_system_prompt(
        context
    )


def _backoff_delay(
    attempt: int,
    initial: float,
    maximum: float,
) -> float:
    """
    Calculate exponential backoff with up to 25% positive jitter.
    """
    base = min(
        initial
        * (
            2
            ** (
                attempt
                - 1
            )
        ),
        maximum,
    )

    jitter = random.uniform(
        0,
        base * 0.25,
    )

    return (
        base
        + jitter
    )


def _retry_after_timeout(
    attempt: int,
    max_attempts: int,
    initial_backoff: float,
    max_backoff: float,
    error: Exception,
) -> None:
    if attempt >= max_attempts:
        raise RuntimeError(
            "Model API timed out after "
            f"{max_attempts} attempts."
        ) from error

    delay = _backoff_delay(
        attempt=attempt,
        initial=initial_backoff,
        maximum=max_backoff,
    )

    print(
        f"{YELLOW}"
        f"⏺ Model request timed out. "
        f"Retrying in {delay:.1f}s "
        f"(next attempt "
        f"{attempt + 1}/{max_attempts})..."
        f"{RESET}"
    )

    time.sleep(
        delay
    )


def call_api(
    context: ExecutionContext,
    messages: list[dict[str, Any]],
    tools: dict[str, Tool],
    *,
    request_timeout: float = 120.0,
    max_attempts: int = 8,
    initial_backoff: float = 1.0,
    max_backoff: float = 30.0,
) -> dict[str, Any]:
    """
    Perform one OpenAI-compatible Chat Completions request.

    Timed-out requests are retried with exponential backoff.
    """
    model = context.model_config

    system_messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt(
                context
            ),
        }
    ]

    memory_context = build_memory_context(
        tools
    )

    if memory_context:
        system_messages.append(
            {
                "role": "system",
                "content": memory_context,
            }
        )

    request_messages = merge_consecutive_roles(
        [
            *system_messages,
            *messages,
        ]
    )

    payload: dict[str, Any] = {
        "model": model.id,
        "max_tokens": model.max_tokens,
        "messages": request_messages,
        "tools": [
            tool.get_as_tool()
            for tool in tools.values()
        ],
    }

    request_data = json.dumps(
        payload
    ).encode(
        "utf-8"
    )

    url = chat_completions_url(
        model.host
    )

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        request = urllib.request.Request(
            url,
            data=request_data,
            headers={
                "Content-Type": (
                    "application/json"
                ),
                "Authorization": (
                    f"Bearer {model.api_key}"
                ),
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=request_timeout,
            ) as response:
                return json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )

        except urllib.error.HTTPError as error:
            try:
                body = error.read().decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                body = ""

            raise RuntimeError(
                f"Model API returned HTTP "
                f"{error.code}: "
                f"{body or error.reason}"
            ) from error

        except (
            TimeoutError,
            socket.timeout,
        ) as error:
            _retry_after_timeout(
                attempt=attempt,
                max_attempts=max_attempts,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                error=error,
            )

            continue

        except urllib.error.URLError as error:
            if isinstance(
                error.reason,
                (
                    TimeoutError,
                    socket.timeout,
                ),
            ):
                _retry_after_timeout(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    initial_backoff=initial_backoff,
                    max_backoff=max_backoff,
                    error=error,
                )

                continue

            raise RuntimeError(
                "Could not connect to model API: "
                f"{error.reason}"
            ) from error

    raise RuntimeError(
        "Model API request failed unexpectedly."
    )

def build_memory_context(
    tools: dict[str, Tool],
) -> str | None:
    """
    Build the current transient session-memory context.

    Memory tools own their state and expose an LLM-readable
    representation through ``format_for_llm``.
    """
    sections: list[str] = []

    for tool in tools.values():
        if not isinstance(tool, MemoryTool):
            continue

        section = tool.format_for_llm().strip()

        if section:
            sections.append(section)

    if not sections:
        return None

    return "\n\n".join(
        (
            "# Session Memory",
            (
                "The following state was retained during the current "
                "agent run. Treat it as active working memory. Update it "
                "through the corresponding memory tools when it becomes "
                "completed, stale, invalid, or otherwise changes."
            ),
            *sections,
        )
    )
# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------


def get_assistant_message(
    response: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract and normalize the assistant message from an OpenAI-compatible
    Chat Completions response.
    """
    try:
        raw = response[
            "choices"
        ][0][
            "message"
        ]

    except (
        KeyError,
        IndexError,
        TypeError,
    ) as error:
        raise RuntimeError(
            "Model returned an invalid "
            "Chat Completions response."
        ) from error

    message: dict[str, Any] = {
        "role": "assistant",
        "content": raw.get(
            "content"
        ),
    }

    tool_calls = raw.get(
        "tool_calls"
    )

    if tool_calls:
        message[
            "tool_calls"
        ] = tool_calls

    return message


def serialize_tool_result(
    result: Any,
) -> str:
    """
    Convert tool results into textual tool-message content.
    """
    if isinstance(
        result,
        str,
    ):
        return result

    try:
        return json.dumps(
            result,
            ensure_ascii=False,
        )

    except (
        TypeError,
        ValueError,
    ):
        return str(
            result
        )


def execute_tool_call(
    tools: dict[str, Tool],
    tool_call: dict[str, Any],
) -> str:
    """
    Parse and execute one OpenAI-compatible tool call.

    Tool failures are returned to the model rather than terminating the
    complete agent loop.
    """
    function = tool_call.get(
        "function",
        {},
    )

    tool_name = function.get(
        "name"
    )

    if not tool_name:
        return (
            "error: tool call does not contain "
            "a function name"
        )

    if tool_name not in tools:
        return (
            f"error: unknown tool "
            f"'{tool_name}'"
        )

    raw_arguments = function.get(
        "arguments",
        "{}",
    )

    try:
        arguments = json.loads(
            raw_arguments
        )

    except json.JSONDecodeError as error:
        return (
            "error: invalid tool arguments JSON: "
            f"{error}"
        )

    if not isinstance(
        arguments,
        dict,
    ):
        return (
            "error: tool arguments must be "
            "a JSON object"
        )

    try:
        result = tools[
            tool_name
        ].execute(
            arguments
        )

        return serialize_tool_result(
            result
        )

    except Exception as error:
        return (
            f"error: {error}"
        )


# ---------------------------------------------------------------------------
# Tool-call UI
# ---------------------------------------------------------------------------


def render_tool_call_start(
    tool_call: dict[str, Any],
) -> dict[str, Any] | None:
    function = tool_call.get(
        "function",
        {},
    )

    tool_name = function.get(
        "name",
        "unknown",
    )

    raw_arguments = function.get(
        "arguments",
        "{}",
    )

    try:
        arguments = json.loads(
            raw_arguments
        )

    except json.JSONDecodeError:
        arguments = None

    preview = (
        argument_preview(
            arguments
        )
        if isinstance(
            arguments,
            dict,
        )
        else raw_arguments[
            :50
        ]
    )

    print(
        f"\n{GREEN}⏺ "
        f"{tool_name_for_display(tool_name)}"
        f"{RESET}("
        f"{DIM}{preview}{RESET}"
        f")"
    )

    return arguments


def render_tool_call_result(
    result: str,
) -> None:
    print(
        f"  {DIM}⎿  "
        f"{result_preview(result)}"
        f"{RESET}"
    )


def get_memory_tool(
    tools: dict[str, Tool],
    tool_call: dict[str, Any],
) -> MemoryTool | None:
    function = tool_call.get(
        "function",
        {},
    )

    tool_name = function.get(
        "name"
    )

    if not tool_name:
        return None

    tool = tools.get(
        tool_name
    )

    if not isinstance(
        tool,
        MemoryTool,
    ):
        return None

    return tool


def render_memory_change(
    tools: dict[str, Tool],
    before: str | None,
) -> None:
    after = build_memory_context(
        tools
    )

    if after == before:
        return

    if not after:
        print(
            f"\n{DIM}"
            f"# Session Memory\n\n"
            f"(empty)"
            f"{RESET}"
        )
        return

    print(
        f"\n{DIM}"
        f"{after}"
        f"{RESET}"
    )
# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------


def run_agent_turn(
    session: AgentSession,
    workspace: WorkspaceContext,
) -> None:
    """
    Continue calling the model until it produces no further tool calls.

    Pending steering messages are injected at the earliest protocol-safe
    boundary.

    If steering arrives before an unexecuted tool call, that call and
    every following call from the same assistant response are cancelled.

    Synthetic tool results are appended so the OpenAI message sequence
    remains valid.
    """

    while True:
        # Safe boundary: no assistant tool-call response is currently
        # awaiting its tool messages.
        session.flush_steering()

        context = ExecutionContext(
            workspace
        )

        tools = TOOL_REGISTRY.instantiate(
            context,
            session,
        )

        response = call_api(
            context=context,
            messages=session.get_last_n_messages(
                context.config.message_context.uncompressed_messages
            ),
            tools=tools,
        )

        assistant_message = get_assistant_message(
            response
        )

        text = assistant_message.get(
            "content"
        )

        if text:
            print(
                f"\n{CYAN}⏺{RESET} "
                f"{render_markdown(text)}"
            )

        tool_calls = assistant_message.get(
            "tool_calls",
            [],
        )

        session.add_assistant_message(
            assistant_message
        )

        if not tool_calls:
            return

        cancel_remaining = False

        for tool_call in tool_calls:
            tool_call_id = tool_call.get(
                "id"
            )

            if not tool_call_id:
                raise RuntimeError(
                    "Model returned a tool call "
                    "without an id."
                )

            if (
                not cancel_remaining
                and session.steering.has_pending()
            ):
                cancel_remaining = True

                print(
                    f"\n{YELLOW}"
                    f"⏺ Steering received. "
                    f"Cancelling remaining tool calls."
                    f"{RESET}"
                )

            render_tool_call_start(
                tool_call
            )

            memory_tool = get_memory_tool(
                tools,
                tool_call,
            )

            memory_before = (
                build_memory_context(
                    tools
                )
                if memory_tool is not None
                else None
            )

            if cancel_remaining:
                result = (
                    _CANCELLED_BY_STEERING
                )
            else:
                result = execute_tool_call(
                    tools,
                    tool_call,
                )

            render_tool_call_result(
                result
            )

            session.add_tool_result(
                tool_call_id,
                result,
            )

            if (
                not cancel_remaining
                and memory_tool is not None
            ):
                render_memory_change(
                    tools,
                    memory_before,
                )
        


# ---------------------------------------------------------------------------
# Command handling
# ---------------------------------------------------------------------------


def is_command(
    user_input: str,
) -> bool:
    """
    Return whether the input is a slash command.
    """
    return user_input.startswith(
        "/"
    )


def handle_command(
    user_input: str,
    session: AgentSession,
    workspace: WorkspaceContext,
) -> bool:
    """
    Execute a slash command.

    Returns ``True`` if the REPL should continue and ``False`` if Citra
    should terminate.
    """
    body = user_input[
        1:
    ]

    parts = body.split(
        None,
        1,
    )

    command_id = (
        parts[0]
        if parts
        else ""
    )

    args = (    
        parts[1].strip()
        if len(parts) > 1
        else ""
    )

    if command_id == "exit":
        command_id = "q"

    context = ExecutionContext(workspace)

    command = COMMAND_REGISTRY.instantiate(
        command_id,
        context,
    )

    if command is None:
        print(
            f"{YELLOW}"
            f"⏺ Unknown command: /{command_id}. "
            f"Type /help for available commands."
            f"{RESET}"
        )

        return True

    result = command.run(
        args
    )

    if result.output:
        print(
            f"\n{result.output}"
        )

    if result.clear_messages:
        session.clear_history()

    return not result.exit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_header(
    context: ExecutionContext,
) -> None:
    print(
        f"{BOLD}citra{RESET} | "
        f"{DIM}"
        f"{context.model_config.id} | "
        f"{context.workspace.workspace}"
        f"{RESET}\n"
    )


def main() -> None:
    session = AgentSession()

    workspace = WorkspaceContext.create(
        os.getcwd()
    )

    try:
        context = ExecutionContext(
            workspace
        )

        print_header(
            context
        )

        while True:
            try:
                print(
                    separator()
                )

                user_input = terminal_input.prompt(
                    f"{BOLD}{BLUE}❯{RESET} "
                ).strip()

                print(
                    separator()
                )

                if not user_input:
                    continue

                if is_command(
                    user_input
                ):
                    should_continue = handle_command(
                        user_input,
                        session,
                        workspace,
                    )

                    if not should_continue:
                        break

                    continue

                session.add_user_message(
                    user_input
                )

                run_agent_turn(
                    session=session,
                    workspace=workspace,
                )

                print()

            except (
                KeyboardInterrupt,
                EOFError,
            ):
                break

            except Exception as error:
                print(
                    f"{RED}"
                    f"⏺ Error: {error}"
                    f"{RESET}"
                )

    finally:
        workspace.cleanup()


if __name__ == "__main__":
    main()