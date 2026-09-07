"""Typed, provider-independent conversation history contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


def freeze_json(value: object) -> JsonValue:
    """Validate and freeze a JSON value received from a model provider."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, dict):
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

    @property
    def is_empty(self) -> bool:
        """Return whether the provider supplied no usable reasoning metadata."""
        return self.reasoning is None and self.content is None and self.details is None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Represent one model-requested tool invocation with raw tool-owned arguments."""

    id: str
    name: str
    arguments: str


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
