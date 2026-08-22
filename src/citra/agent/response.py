"""Provider-response parsing and isolated tool-call execution."""

from __future__ import annotations

import json
from typing import Any, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
)

from .session import ChatMessage
from ..tools.tool import Tool


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
    try:
        return serialize_tool_result(tool.execute(arguments))
    except Exception as error:
        return f"error: {error}"


__all__ = [
    "execute_tool_call",
    "get_assistant_message",
    "serialize_tool_result",
]

