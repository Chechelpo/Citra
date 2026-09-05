"""Provider-response parsing and isolated tool-call execution."""
from __future__ import annotations

import json
import time
import traceback
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


def get_assistant_message(
    response: dict[str, Any],
) -> ChatCompletionAssistantMessageParam:
    """Return assistant message."""

    _logger.debug(
        "Parsing assistant response",
        response=response,
    )

    try:
        raw = response["choices"][0]["message"]

    except (KeyError, IndexError, TypeError) as error:
        _logger.error(
            "Model returned invalid Chat Completions response",
            error=str(error),
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
        )
        raise RuntimeError(
            "Model returned an invalid Chat Completions response."
        ) from error

    if not isinstance(raw, dict):
        _logger.error(
            "Assistant message payload is not an object",
            payload_type=type(raw).__name__,
            payload=raw,
        )
        raise RuntimeError(
            "Model returned an invalid Chat Completions response."
        )

    content = raw.get("content")

    if content is not None and not isinstance(content, str):
        _logger.error(
            "Assistant content is not a string",
            content_type=type(content).__name__,
            content=content,
        )
        raise RuntimeError(
            "Model returned invalid assistant content."
        )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }

    tool_calls = raw.get("tool_calls")

    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            _logger.error(
                "Assistant tool_calls field is not a list",
                value=tool_calls,
            )
            raise RuntimeError(
                "Model returned invalid tool calls."
            )

        _logger.info(
            "Assistant response contains tool calls",
            count=len(tool_calls),
            tool_calls=tool_calls,
        )

        message["tool_calls"] = tool_calls

    for field in (
        "reasoning_details",
        "reasoning",
        "reasoning_content",
    ):
        if field in raw:
            _logger.trace(
                "Preserving provider reasoning field",
                field=field,
                value=raw[field],
            )
            message[field] = raw[field]

    _logger.debug(
        "Assistant message parsed",
        message=message,
    )

    return cast(
        ChatCompletionAssistantMessageParam,
        message,
    )


def serialize_tool_result(result: Any) -> str:
    """Serialize tool result."""

    if isinstance(result, str):
        return result

    try:
        return json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

    except (TypeError, ValueError) as error:
        _logger.warning(
            "Failed to serialize tool result",
            error=str(error),
            error_type=type(error).__name__,
            result=result,
        )
        return str(result)


def execute_tool_call(
    tools: dict[str, Tool],
    tool_call: ChatCompletionMessageFunctionToolCallParam,
    *,
    session: AgentSession | None = None,
) -> str:
    """Execute the execute tool call operation."""

    started = time.monotonic()

    function = tool_call["function"]

    tool_name = function.get("name")
    call_id = tool_call.get("id")

    _logger.info(
        "Starting tool execution",
        tool=tool_name,
        call_id=call_id,
        tool_call=tool_call,
    )

    if not tool_name:
        _logger.warning(
            "Tool call missing function name",
            call_id=call_id,
            tool_call=tool_call,
        )
        return "error: tool call does not contain a function name"

    tool = tools.get(tool_name)

    if tool is None:
        _logger.warning(
            "Requested tool does not exist",
            tool=tool_name,
            available_tools=list(tools.keys()),
        )
        return f"error: unknown tool '{tool_name}'"

    raw_arguments = function.get("arguments", "{}")

    _logger.debug(
        "Raw tool arguments received",
        tool=tool.id,
        call_id=call_id,
        arguments=raw_arguments,
    )

    try:
        arguments = json.loads(raw_arguments)

    except json.JSONDecodeError as error:
        _logger.warning(
            "Tool arguments are invalid JSON",
            tool=tool.id,
            arguments=raw_arguments,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        return f"error: invalid tool arguments JSON: {error}"

    if not isinstance(arguments, dict):
        _logger.warning(
            "Tool arguments are not an object",
            tool=tool.id,
            arguments=arguments,
        )
        return "error: tool arguments must be a JSON object"

    _logger.debug(
        "Parsed tool arguments",
        tool=tool.id,
        arguments=arguments,
    )

    invalidates_cache = tool.invalidates_tool_cache(arguments)
    cacheable = tool.is_cacheable(arguments)

    _logger.trace(
        "Tool cache state evaluated",
        tool=tool.id,
        cacheable=cacheable,
        invalidates_cache=invalidates_cache,
    )

    if session is not None and invalidates_cache:
        _logger.info(
            "Invalidating tool cache",
            tool=tool.id,
        )
        session.tool_cache.invalidate()

    if session is not None and cacheable:
        cached = session.tool_cache.get(
            tool.id,
            arguments,
        )

        if cached is not None:
            _logger.info(
                "Tool cache hit",
                tool=tool.id,
                generation=cached.generation,
                cached_result=cached.result,
            )

            if session.can_reuse_tool_cache:
                _logger.info(
                    "Returning cache reuse marker",
                    tool=tool.id,
                )
                return _CACHE_HIT_TEMPLATE.format(
                    tool_id=tool.id,
                )

            return cached.result

        _logger.debug(
            "Tool cache miss",
            tool=tool.id,
        )

    _logger.info(
        "Calling tool implementation",
        tool=tool.id,
        arguments=arguments,
    )

    try:
        raw_result = tool.execute(arguments)

        _logger.trace(
            "Tool returned raw object",
            tool=tool.id,
            result=raw_result,
        )

        result = serialize_tool_result(raw_result)

        _logger.debug(
            "Tool result serialized",
            tool=tool.id,
            size=len(result),
            result=result,
        )

        if tool.MAX_OUTPUT_TOKENS is not None:
            before = result

            result = tool.context.truncate_output(
                result,
                max_tokens=tool.MAX_OUTPUT_TOKENS,
            )

            if before != result:
                _logger.warning(
                    "Tool output truncated",
                    tool=tool.id,
                    before_size=len(before),
                    after_size=len(result),
                    max_tokens=tool.MAX_OUTPUT_TOKENS,
                )

    except Exception as error:
        _logger.error(
            "Tool execution failed",
            tool=tool.id,
            error=str(error),
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
        )

        return f"error: {error}"

    if (
        session is not None
        and cacheable
        and not result.startswith("error:")
    ):
        _logger.debug(
            "Writing tool result to cache",
            tool=tool.id,
            result=result,
        )

        session.tool_cache.put(
            tool.id,
            arguments,
            result,
        )

    duration = time.monotonic() - started

    _logger.info(
        "Tool execution completed",
        tool=tool.id,
        duration_ms=round(duration * 1000, 3),
        result_size=len(result),
        result=result,
    )

    return result