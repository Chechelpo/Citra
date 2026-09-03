"""Provider-response parsing and isolated tool-call execution."""
from __future__ import annotations

import json
import logging
from typing import Any, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
)

from .session import AgentSession
from ..tools.tool import Tool
from citra.logging import Logger


_logger = Logger("response.py")

_CACHE_HIT_TEMPLATE = (
    "unchanged since previous identical {tool_id} call this turn; "
    "reuse the earlier result"
)


def get_assistant_message(response: dict[str, Any]) -> ChatCompletionAssistantMessageParam:
    _logger.debug("Parsing assistant response")

    try:
        raw = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        _logger.error(
            "Model returned invalid Chat Completions response",
            error=str(error),
        )
        raise RuntimeError("Model returned an invalid Chat Completions response.") from error

    if not isinstance(raw, dict):
        _logger.error("Assistant message payload is not an object")
        raise RuntimeError("Model returned an invalid Chat Completions response.")

    content = raw.get("content")

    if content is not None and not isinstance(content, str):
        _logger.error("Assistant content is not a string")
        raise RuntimeError("Model returned invalid assistant content.")

    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }

    tool_calls = raw.get("tool_calls")

    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            _logger.error("Assistant tool_calls field is not a list")
            raise RuntimeError("Model returned invalid tool calls.")

        _logger.debug(
            "Assistant response contains tool calls",
            count=len(tool_calls),
        )
        message["tool_calls"] = tool_calls

    for field in ("reasoning_details", "reasoning", "reasoning_content"):
        if field in raw:
            _logger.trace(
                "Preserving provider reasoning field",
                field=field,
            )
            message[field] = raw[field]

    _logger.info("Assistant message parsed successfully")

    return cast(ChatCompletionAssistantMessageParam, message)


def serialize_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)


def execute_tool_call(
    tools: dict[str, Tool],
    tool_call: ChatCompletionMessageFunctionToolCallParam,
    *,
    session: AgentSession | None = None,
) -> str:
    function = tool_call["function"]

    tool_name = function.get("name")

    _logger.debug(
        "Executing tool call",
        tool=tool_name,
    )

    if not tool_name:
        _logger.warning("Tool call missing function name")
        return "error: tool call does not contain a function name"

    tool = tools.get(tool_name)

    if tool is None:
        _logger.warning(
            "Requested tool does not exist",
            tool=tool_name,
        )
        return f"error: unknown tool '{tool_name}'"

    try:
        arguments = json.loads(function.get("arguments", "{}"))
    except json.JSONDecodeError as error:
        _logger.warning(
            "Tool arguments are invalid JSON",
            tool=tool.id,
            error=str(error),
        )
        return f"error: invalid tool arguments JSON: {error}"

    if not isinstance(arguments, dict):
        _logger.warning(
            "Tool arguments are not an object",
            tool=tool.id,
        )
        return "error: tool arguments must be a JSON object"

    _logger.trace(
        "Tool arguments parsed",
        tool=tool.id,
        arguments=arguments,
    )

    if session is not None and tool.invalidates_tool_cache(arguments):
        _logger.debug(
            "Invalidating tool cache",
            tool=tool.id,
        )
        session.tool_cache.invalidate()

    if session is not None and tool.is_cacheable(arguments):
        cached = session.tool_cache.get(tool.id, arguments)

        if cached is not None:
            _logger.trace(
                "Tool cache hit",
                tool=tool.id,
                generation=cached.generation,
            )

            if session.can_reuse_tool_cache:
                _logger.info(
                    "Reusing tool cache marker",
                    tool=tool.id,
                )
                return _CACHE_HIT_TEMPLATE.format(tool_id=tool.id)

            _logger.info(
                "Returning cached tool result",
                tool=tool.id,
            )
            return cached.result

    _logger.debug(
        "Executing tool",
        tool=tool.id,
    )

    try:
        result = serialize_tool_result(tool.execute(arguments))

        _logger.trace(
            "Tool returned result",
            tool=tool.id,
            size=len(result),
        )

        if tool.MAX_OUTPUT_TOKENS is not None:
            result = tool.context.truncate_output(
                result,
                max_tokens=tool.MAX_OUTPUT_TOKENS,
            )

    except Exception as error:
        _logger.error(
            "Tool execution failed",
            tool=tool.id,
            error=str(error),
        )
        return f"error: {error}"

    if (
        session is not None
        and tool.is_cacheable(arguments)
        and not result.startswith("error:")
    ):
        _logger.debug(
            "Caching tool result",
            tool=tool.id,
        )

        session.tool_cache.put(
            tool.id,
            arguments,
            result,
        )

    _logger.info(
        "Tool execution completed",
        tool=tool.id,
    )

    return result


__all__ = [
    "execute_tool_call",
    "get_assistant_message",
    "serialize_tool_result",
]

