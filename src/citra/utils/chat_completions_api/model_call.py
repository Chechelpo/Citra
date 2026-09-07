"""Typed request boundary between agent orchestration and model providers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citra.agent.chat_message import ChatMessage
    from citra.config import ModelConfig
    from citra.context import ExecutionContext
    from citra.tools.tool import Tool


@dataclass(frozen=True, slots=True)
class ModelCall:
    """Capture every input required for one logical model request."""

    context: ExecutionContext
    messages: tuple[ChatMessage, ...]
    tools: Mapping[str, Tool]
    system_prompt: str = ""
    reasoning_effort: str | None = None
    model_config: ModelConfig | None = None
    request_timeout: float | None = None
    max_attempts: int | None = None
    initial_backoff: float | None = None
    max_backoff: float | None = None
    retry_interrupt: Callable[[], bool] | None = None
    memory_services: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        """Detach mutable request containers from the runner's live state."""
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", MappingProxyType(dict(self.tools)))
        object.__setattr__(self, "memory_services", tuple(self.memory_services))
