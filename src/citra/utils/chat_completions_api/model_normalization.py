import json
import re
from typing import Any


_KIMI_TOOL_CALL_RE = re.compile(
    r"functions\.([A-Za-z_][A-Za-z0-9_.-]*):(\d+)"
)


def normalize_kimi_tool_calls(
    response: dict[str, Any],
    *,
    tools: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convert leaked Kimi native tool calls such as:

        functions.read:0{"path":"instructions.txt"}
        functions.tree:1{"path":".","max_tokens":4000}

    into OpenAI-compatible message.tool_calls.

    Leaves responses unchanged when structured tool_calls already exist.
    """
    choices = response.get("choices")
    if not isinstance(choices, list):
        return response

    decoder = json.JSONDecoder()

    for choice in choices:
        if not isinstance(choice, dict):
            continue

        message = choice.get("message")
        if not isinstance(message, dict):
            continue

        # Already normalized by provider.
        if message.get("tool_calls"):
            continue

        content = message.get("content")
        if not isinstance(content, str) or not content:
            continue

        matches = list(_KIMI_TOOL_CALL_RE.finditer(content))
        if not matches:
            continue

        parsed_calls: list[dict[str, Any]] = []
        first_call_start: int | None = None
        consumed_until: int | None = None

        for match in matches:
            # Calls must be contiguous after the first parsed call,
            # apart from whitespace.
            if consumed_until is not None:
                between = content[consumed_until:match.start()]
                if between.strip():
                    break

            function_name = match.group(1)
            call_index = match.group(2)

            # Never turn unknown function-looking text into executable calls.
            if tools is not None and function_name not in tools:
                break

            arg_start = match.end()

            # Allow whitespace between call header and JSON.
            while (
                arg_start < len(content)
                and content[arg_start].isspace()
            ):
                arg_start += 1

            try:
                arguments, arg_end = decoder.raw_decode(
                    content,
                    arg_start,
                )
            except json.JSONDecodeError:
                break

            if not isinstance(arguments, dict):
                break

            if first_call_start is None:
                first_call_start = match.start()

            parsed_calls.append(
                {
                    "id": f"functions.{function_name}:{call_index}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(
                            arguments,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ),
                    },
                }
            )

            consumed_until = arg_end

        if not parsed_calls or first_call_start is None:
            continue

        # Don't partially normalize if unparsed non-whitespace remains
        # after the parsed tool-call sequence.
        if (
            consumed_until is not None
            and content[consumed_until:].strip()
        ):
            continue

        leading_content = content[:first_call_start].rstrip()

        message["content"] = leading_content or None
        message["tool_calls"] = parsed_calls

        if choice.get("finish_reason") in {
            None,
            "stop",
        }:
            choice["finish_reason"] = "tool_calls"

    return response