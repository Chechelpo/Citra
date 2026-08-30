"""Durable structured memory owned by a conversation, not by tool calls."""

from __future__ import annotations

from threading import RLock
from typing import Callable, TypeVar, cast


T = TypeVar("T")


class ConversationMemory:
    """Own memory services across turns and history compaction.

    Existing memory tools already provide typed validation and formatting, so
    their instances are retained here. The key ownership boundary is the
    :class:`AgentSession`: a registry refresh or a dropped chat message cannot
    silently discard working state.
    """

    def __init__(self) -> None:
        self._services: dict[str, object] = {}
        self._lock = RLock()

    def get_or_create(
        self,
        name: str,
        factory: Callable[[], T],
    ) -> T:
        with self._lock:
            existing = self._services.get(name)
            if existing is None:
                existing = factory()
                self._services[name] = existing
            return cast(T, existing)

    def values(self) -> tuple[object, ...]:
        with self._lock:
            return tuple(self._services.values())

    def get(self, name: str) -> object | None:
        """Return one existing memory service without creating it."""
        with self._lock:
            return self._services.get(name)

    def clear(self) -> None:
        with self._lock:
            self._services.clear()
