"""Parse provider payloads into Citra's typed response contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from citra.agent.chat_message import (
    AssistantMessage,
    ReasoningMetadata,
    ToolCall,
    freeze_json,
)


class ModelResponseParseError(ValueError):
    """A provider response cannot be represented by Citra's response contract."""


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Represent token usage reported by a provider when available."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Represent one parsed model response consumed by the agent runner."""

    assistant: AssistantMessage
    finish_reason: str | None = None
    usage: ModelUsage = ModelUsage()


def parse_model_response(response: object) -> ModelResponse:
    """Validate and parse one normalized Chat Completions response payload."""
    root = _object(response, "response")
    choices = root.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelResponseParseError("Model response contains no choices.")

    choice = _object(choices[0], "response choice")
    message = _object(choice.get("message"), "assistant message")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ModelResponseParseError("Assistant content must be a string or null.")

    return ModelResponse(
        assistant=AssistantMessage(
            content=content,
            tool_calls=_parse_tool_calls(message.get("tool_calls")),
            reasoning=ReasoningMetadata(
                reasoning=freeze_json(message.get("reasoning")),
                content=freeze_json(message.get("reasoning_content")),
                details=freeze_json(message.get("reasoning_details")),
            ),
        ),
        finish_reason=_optional_string(choice.get("finish_reason"), "finish_reason"),
        usage=_parse_usage(root.get("usage")),
    )


def _parse_tool_calls(value: object) -> tuple[ToolCall, ...]:
    """Parse provider tool-call envelopes while leaving arguments untouched."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ModelResponseParseError("Assistant tool_calls must be a list.")

    parsed: list[ToolCall] = []
    seen_ids: set[str] = set()
    for index, raw_call in enumerate(value):
        call = _object(raw_call, f"tool call {index}")
        call_id = _required_string(call.get("id"), f"tool call {index} id")
        if call_id in seen_ids:
            raise ModelResponseParseError(f"Duplicate tool call id {call_id!r}.")
        seen_ids.add(call_id)
        function = _object(call.get("function"), f"tool call {index} function")
        parsed.append(
            ToolCall(
                id=call_id,
                name=_required_string(
                    function.get("name"),
                    f"tool call {index} function name",
                ),
                arguments=_required_string(
                    function.get("arguments"),
                    f"tool call {index} arguments",
                    allow_empty=True,
                ),
            )
        )
    return tuple(parsed)


def _parse_usage(value: object) -> ModelUsage:
    """Parse OpenAI and provider-compatible token usage spellings."""
    if value is None:
        return ModelUsage()
    usage = _object(value, "usage")
    return ModelUsage(
        input_tokens=_optional_int(
            usage.get("prompt_tokens", usage.get("input_tokens")),
            "usage input tokens",
        ),
        output_tokens=_optional_int(
            usage.get("completion_tokens", usage.get("output_tokens")),
            "usage output tokens",
        ),
        total_tokens=_optional_int(usage.get("total_tokens"), "usage total tokens"),
    )


def _object(value: object, label: str) -> dict[str, Any]:
    """Narrow a provider value to a string-keyed object."""
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ModelResponseParseError(f"{label.capitalize()} must be an object.")
    return value


def _required_string(value: object, label: str, *, allow_empty: bool = False) -> str:
    """Return a required provider string with contextual validation."""
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ModelResponseParseError(f"{label.capitalize()} must be a string.")
    return value


def _optional_string(value: object, label: str) -> str | None:
    """Return an optional provider string."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ModelResponseParseError(f"{label.capitalize()} must be a string or null.")
    return value


def _optional_int(value: object, label: str) -> int | None:
    """Return an optional non-negative provider integer."""
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ModelResponseParseError(
            f"{label.capitalize()} must be a non-negative integer or null."
        )
    return value
