"""Thread-safe handoff of model questions to the terminal/UI thread."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Lock
import time


@dataclass
class UserPromptRequest:
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
        self._pending: Queue[UserPromptRequest] = Queue()
        self._active: dict[int, UserPromptRequest] = {}
        self._lock = Lock()
        self._next_id = 1

    def ask(
        self,
        question: str,
        options: tuple[str, ...],
        *,
        timeout: float,
    ) -> str | None:
        with self._lock:
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
        while True:
            try:
                request = self._pending.get_nowait()
            except Empty:
                return None
            with self._lock:
                if request.id in self._active:
                    return request

    def respond(self, request_id: int, answer: str | None) -> bool:
        with self._lock:
            request = self._active.pop(request_id, None)
            if request is None:
                return False
            request.answer = answer
            request.completed.set()
            return True

    def has_pending(self) -> bool:
        return not self._pending.empty()

    def close(self) -> None:
        with self._lock:
            active = tuple(self._active.values())
            self._active.clear()
        for request in active:
            request.completed.set()

