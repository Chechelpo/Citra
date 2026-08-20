"""
Thread-safe steering inbox for active Citra agent turns.

The terminal/UI thread may enqueue user corrections while the agent
worker is calling the model or executing tools.

The agent worker drains these messages only at valid conversation
boundaries.
"""

from __future__ import annotations

from collections import deque
from threading import Lock


__all__ = [
    "SteeringInbox",
]


class SteeringInbox:
    """
    Thread-safe FIFO queue of user steering messages.

    FIFO ordering is intentional: corrections should reach the model in
    the same order in which the user submitted them.
    """

    def __init__(self) -> None:
        self._messages: deque[str] = deque()
        self._lock = Lock()

    def push(
        self,
        message: str,
    ) -> bool:
        """
        Queue one steering message.

        Returns ``True`` when a non-empty message was queued, otherwise
        ``False``.
        """
        message = message.strip()

        if not message:
            return False

        with self._lock:
            self._messages.append(
                message
            )

        return True

    def drain(self) -> tuple[str, ...]:
        """
        Atomically remove and return all currently queued messages.
        """
        with self._lock:
            messages = tuple(
                self._messages
            )

            self._messages.clear()

        return messages

    def has_pending(self) -> bool:
        """
        Return whether at least one steering message is queued.
        """
        with self._lock:
            return bool(
                self._messages
            )

    def clear(self) -> None:
        """
        Discard every pending steering message.
        """
        with self._lock:
            self._messages.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(
                self._messages
            )
