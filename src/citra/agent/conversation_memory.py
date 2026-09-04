"""Durable structured memory owned by a conversation, not by tool calls."""

from __future__ import annotations

from threading import RLock
from typing import Callable, TypeVar, cast

from citra.logging import Logger


T = TypeVar("T")
_logger = Logger(__name__)


class ConversationMemory:
    """Own memory services across turns and history compaction.

    Existing memory tools already provide typed validation and formatting, so
    their instances are retained here. The key ownership boundary is the
    :class:`AgentSession`: a registry refresh or a dropped chat message cannot
    silently discard working state.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self._services: dict[str, object] = {}
        self._lock = RLock()
        _logger.trace("Initialized conversation memory")

    def get_or_create(
        self,
        name: str,
        factory: Callable[[], T],
    ) -> T:
        """Return get or create."""
        with self._lock:
            existing = self._services.get(name)
            if existing is None:
                existing = factory()
                self._services[name] = existing
                _logger.debug("Created memory service", service=name)
            else:
                _logger.trace("Reused memory service", service=name)
            return cast(T, existing)

    def values(self) -> tuple[object, ...]:
        """Return retained memory services in insertion order."""
        with self._lock:
            _logger.trace("Listed memory services", count=len(self._services))
            return tuple(self._services.values())

    def get(self, name: str) -> object | None:
        """Return one existing memory service without creating it."""
        with self._lock:
            service = self._services.get(name)
            _logger.trace(
                "Looked up memory service",
                service=name,
                found=service is not None,
            )
            return service

    def clear(self) -> None:
        """Clear every retained service for the conversation."""
        with self._lock:
            count = len(self._services)
            self._services.clear()
            _logger.info("Cleared conversation memory", services=count)
