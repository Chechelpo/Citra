"""Provider-response parsing and isolated tool-call execution."""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
)

from .session import AgentSession, ChatMessage
from ..tools.tool import Tool


logger = logging.getLogger(__name__)

_CACHE_HIT_TEMPLATE = (
    "unchanged since previous identical {tool_id} call this turn; "
    "reuse the earlier result"
)


def get_assistant_message(response: dict[str, Any]) -> ChatCompletionAssistantMessageParam:
    try:
        raw = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Model returned an invalid Chat Completions response.") from error
    if not isinstance(raw, dict):
        raise RuntimeError("Model returned an invalid Chat Completions response.")
    content = raw.get("content")
    if content is not None and not isinstance(content, str):
        raise RuntimeError("Model returned invalid assistant content.")
    message: dict[str, Any] = {"role": "assistant", "content": content}
    tool_calls = raw.get("tool_calls")
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            raise RuntimeError("Model returned invalid tool calls.")
        message["tool_calls"] = tool_calls
    for field in ("reasoning_details", "reasoning", "reasoning_content"):
        if field in raw:
            message[field] = raw[field]
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
    if not tool_name:
        return "error: tool call does not contain a function name"
    tool = tools.get(tool_name)
    if tool is None:
        return f"error: unknown tool '{tool_name}'"
    try:
        arguments = json.loads(function.get("arguments", "{}"))
    except json.JSONDecodeError as error:
        return f"error: invalid tool arguments JSON: {error}"
    if not isinstance(arguments, dict):
        return "error: tool arguments must be a JSON object"

    # Mutating/possibly-mutating tools invalidate before execution. This is
    # deliberately conservative: a tool may partially mutate state before
    # failing, and stale cached inspection results are worse than a miss.
    if session is not None and tool.invalidates_tool_cache(arguments):
        session.tool_cache.invalidate()

    if session is not None and tool.is_cacheable(arguments):
        cached = session.tool_cache.get(tool.id, arguments)
        if cached is not None:
            logger.info(
                "[%s] CACHE HIT generation=%d",
                tool.id,
                cached.generation,
            )
            if session.can_reuse_tool_cache:
                return _CACHE_HIT_TEMPLATE.format(tool_id=tool.id)

            # The earlier result has fallen out of the selected model context.
            # Reuse the cached payload rather than executing the tool again.
            return cached.result

    try:
        result = serialize_tool_result(tool.execute(arguments))
        if tool.MAX_OUTPUT_TOKENS is not None:
            result = tool.context.truncate_output(
                result,
                max_tokens=tool.MAX_OUTPUT_TOKENS,
            )
    except Exception as error:
        return f"error: {error}"

    if (
        session is not None
        and tool.is_cacheable(arguments)
        and not result.startswith("error:")
    ):
        session.tool_cache.put(
            tool.id,
            arguments,
            result,
        )

    return result


__all__ = [
    "execute_tool_call",
    "get_assistant_message",
    "serialize_tool_result",
]

