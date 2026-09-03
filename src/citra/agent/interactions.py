"""Thread-safe handoff of model questions to the terminal/UI thread."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Lock
import time


@dataclass
class UserPromptRequest:
    """Represent UserPromptRequest."""
    id: int
    question: str
    options: tuple[str, ...]
    timeout: float
    created_at: float = field(default_factory=time.monotonic)
    completed: Event = field(default_factory=Event)
    answer: str | None = None


class UserInteractionBroker:
    """Let a background agent synchronously ask the foreground user."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self._pending: Queue[UserPromptRequest] = Queue()
        self._active: dict[int, UserPromptRequest] = {}
        self._lock = Lock()
        self._next_id = 1
        self._closed = False

    def ask(
        self,
        question: str,
        options: tuple[str, ...],
        *,
        timeout: float,
    ) -> str | None:
        """Handle ask."""
        with self._lock:
            if self._closed:
                return None
            request = UserPromptRequest(
                id=self._next_id,
                question=question,
                options=options,
                timeout=timeout,
            )
            self._next_id += 1
            self._active[request.id] = request
            self._pending.put(request)
        if not request.completed.wait(timeout):
            with self._lock:
                self._active.pop(request.id, None)
            return None
        return request.answer

    def take(self) -> UserPromptRequest | None:
        """Handle take."""
        while True:
            try:
                request = self._pending.get_nowait()
            except Empty:
                return None
            with self._lock:
                if request.id in self._active:
                    return request

    def respond(self, request_id: int, answer: str | None) -> bool:
        """Handle respond."""
        with self._lock:
            request = self._active.pop(request_id, None)
            if request is None:
                return False
            request.answer = answer
            request.completed.set()
            return True

    def has_pending(self) -> bool:
        """Return whether has pending."""
        return not self._pending.empty()

    def close(self) -> None:
        """Handle close."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            active = tuple(self._active.values())
            self._active.clear()
        for request in active:
            request.completed.set()
