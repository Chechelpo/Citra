"""Typed, provider-independent conversation history contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias, cast

if TYPE_CHECKING:
    from citra.tools.tool import Tool, ToolArguments


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


def freeze_json(value: object) -> JsonValue:
    """Validate and freeze a JSON value received from a model provider."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings.")
        return MappingProxyType(
            {str(key): freeze_json(item) for key, item in value.items()}
        )
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ReasoningMetadata:
    """Preserve provider reasoning fields without exposing response dictionaries."""

    reasoning: JsonValue = None
    content: JsonValue = None
    details: JsonValue = None

    def __post_init__(self) -> None:
        """Validate and detach provider metadata supplied by direct callers."""
        object.__setattr__(self, "reasoning", freeze_json(self.reasoning))
        object.__setattr__(self, "content", freeze_json(self.content))
        object.__setattr__(self, "details", freeze_json(self.details))

    @property
    def is_empty(self) -> bool:
        """Return whether the provider supplied no usable reasoning metadata."""
        return self.reasoning is None and self.content is None and self.details is None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Represent one model-requested invocation with parsed tool-owned arguments."""

    id: str
    name: str
    arguments: ToolArguments

    def __post_init__(self) -> None:
        """Keep direct callers compatible while storing only typed arguments."""
        from citra.tools.tool import ToolArguments, UnboundToolArguments

        if isinstance(self.arguments, ToolArguments):
            return
        if not isinstance(self.arguments, str):
            raise TypeError("Tool-call arguments must be ToolArguments or JSON text.")
        decoded = json.loads(self.arguments or "{}")
        if not isinstance(decoded, dict):
            raise TypeError("Tool-call arguments must be a JSON object.")
        object.__setattr__(
            self,
            "arguments",
            UnboundToolArguments.from_dict(decoded),
        )


@dataclass(frozen=True, slots=True)
class UserMessage:
    """Represent user conversation content."""

    content: str


@dataclass(frozen=True, slots=True)
class SystemMessage:
    """Represent system conversation content."""

    content: str


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """Represent parsed assistant content, reasoning, and tool requests."""

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning: ReasoningMetadata = ReasoningMetadata()


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """Represent the model-facing result of one tool invocation."""

    tool_call_id: str
    content: str


ChatMessage: TypeAlias = (
    UserMessage | SystemMessage | AssistantMessage | ToolResultMessage
)


def bind_tool_call_arguments(
    message: AssistantMessage,
    tools: Mapping[str, object],
) -> AssistantMessage:
    """Rebuild calls with their concrete tool argument classes before storage."""
    bound: list[ToolCall] = []
    for call in message.tool_calls:
        tool = tools.get(call.name)
        if tool is not None and hasattr(tool, "arguments_type"):
            concrete_tool = cast("Tool[Any]", tool)
            arguments_type = concrete_tool.arguments_type()
            if isinstance(call.arguments, arguments_type):
                bound.append(call)
                continue
            arguments = arguments_type.from_dict(call.arguments.to_dict())
            call = replace(call, arguments=arguments)
        bound.append(call)
    return replace(message, tool_calls=tuple(bound))
