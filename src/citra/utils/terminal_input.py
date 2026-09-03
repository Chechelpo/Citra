# src/citra/utils/terminal_input.py

"""
Centralized interactive terminal input for Citra.

This module is the single place where Citra touches ``prompt_toolkit``
for interactive user input. All application-level code (the main REPL,
``PromptUser``, and any other tool that needs to ask the user something)
must go through this utility rather than calling ``input()`` or
implementing its own ``termios``/``select``/``tty`` raw reader.

Design goals:

* Normal line editing is preserved (arrows, backspace, delete,
  Home/End, Ctrl+A/Ctrl+E, paste, etc.) through ``prompt_toolkit``.
* ``Ctrl+C`` raises ``KeyboardInterrupt`` and ``Ctrl+D`` raises
  ``EOFError`` like the stdlib ``input()`` built-in.
* ``prompt_with_idle_timeout()`` implements an inactivity timeout:
  the timer resets on every buffer modification, so a user may take
  arbitrarily long to answer as long as they do not remain idle for
  the full timeout interval.
* Callers may pass ANSI-styled prompt strings. This module converts
  them to ``prompt_toolkit`` formatted text before rendering.
"""

from __future__ import annotations

from asyncio import TimerHandle
from typing import Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import ANSI


__all__ = [
    "TerminalInput",
    "terminal_input",
]


class TerminalInput:
    """
    Reusable ``prompt_toolkit``-based terminal input abstraction.

    A single shared :class:`PromptSession` is kept internally so future
    features such as history, completion, multiline editing, and syntax
    highlighting can be added centrally without changing callers.

    The ``PromptSession`` itself is not exposed.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self._session: PromptSession[str] = PromptSession()

    def prompt(
        self,
        message: str = "",
    ) -> str:
        """
        Read one line of normal, unlimited user input.

        ``message`` may contain ANSI escape sequences, such as Citra's
        styled prompt.

        Raises:
            KeyboardInterrupt:
                When Ctrl+C is pressed.
            EOFError:
                When Ctrl+D is pressed on an empty buffer.
        """
        return self._session.prompt(
            ANSI(message),
            handle_sigint=True,
        )

    def prompt_with_idle_timeout(
        self,
        timeout: float,
        message: str = "",
    ) -> str | None:
        """
        Read one line with an inactivity timeout.

        The timeout restarts whenever the input buffer changes, including
        insertions, deletions, replacements, and pasted text.

        A user may therefore take longer than ``timeout`` overall as long
        as they do not remain inactive for ``timeout`` consecutive
        seconds.

        Args:
            timeout:
                Maximum number of consecutive idle seconds.
            message:
                Optional ANSI-styled prompt text.

        Returns:
            The submitted line, or ``None`` when the inactivity timeout
            expires.

        Raises:
            ValueError:
                If ``timeout`` is not greater than zero.
            KeyboardInterrupt:
                When Ctrl+C is pressed.
            EOFError:
                When Ctrl+D is pressed on an empty buffer.
        """
        if timeout <= 0:
            raise ValueError(
                "'timeout' must be greater than zero."
            )

        watchdog = _IdleWatchdog(
            timeout=timeout,
        )

        try:
            return self._session.prompt(
                ANSI(message),
                pre_run=watchdog.start,
                handle_sigint=True,
            )
        except _IdleTimeout:
            return None
        finally:
            watchdog.stop()

    def prompt_until(
        self,
        predicate: Callable[[], bool],
        message: str = "",
        *,
        poll_interval: float = 0.1,
    ) -> str | None:
        """Read a line, or return ``None`` once ``predicate`` is true.

        This lets the foreground terminal remain available for steering while
        a background agent works, without polling stdin or running two prompt
        sessions concurrently.
        """
        if poll_interval <= 0:
            raise ValueError("'poll_interval' must be greater than zero.")
        watcher = _PredicateWatchdog(
            predicate=predicate,
            interval=poll_interval,
        )
        try:
            return self._session.prompt(
                ANSI(message),
                pre_run=watcher.start,
                handle_sigint=True,
            )
        except _PredicateSatisfied:
            return None
        finally:
            watcher.stop()


class _IdleTimeout(Exception):
    """
    Internal sentinel used to terminate a timed prompt after inactivity.
    """


class _PredicateSatisfied(Exception):
    """Internal sentinel used to wake a prompt for background state."""


class _PredicateWatchdog:
    """Represent PredicateWatchdog."""
    def __init__(self, *, predicate: Callable[[], bool], interval: float) -> None:
        """Initialize the instance."""
        self._predicate = predicate
        self._interval = interval
        self._handle: TimerHandle | None = None

    def start(self) -> None:
        """Handle start."""
        self._check()

    def stop(self) -> None:
        """Handle stop."""
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def _check(self) -> None:
        """Handle check."""
        if self._predicate():
            get_app().exit(exception=_PredicateSatisfied())
            return
        self._handle = get_app().loop.call_later(
            self._interval,
            self._check,
        )


class _IdleWatchdog:
    """
    Tracks user-input inactivity for one prompt invocation.

    ``start()`` attaches to the active prompt buffer and arms the timer.
    Every ``Buffer.on_text_changed`` event resets the timer.

    When the timer expires, the current ``prompt_toolkit`` application
    exits by raising ``_IdleTimeout``.
    """

    def __init__(
        self,
        timeout: float,
    ) -> None:
        """Initialize the instance."""
        self._timeout = timeout

        self._handle: TimerHandle | None = None
        self._buffer: Buffer | None = None

    def start(self) -> None:
        """
        Attach to the active prompt buffer and start the inactivity timer.

        Intended to be passed as ``PromptSession.prompt(pre_run=...)``.
        """
        app = get_app()

        self._buffer = app.layout.current_buffer
        self._buffer.on_text_changed += self._on_text_changed

        self._schedule()

    def stop(self) -> None:
        """
        Cancel the timer and detach the buffer event handler.
        """
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

        if self._buffer is not None:
            try:
                self._buffer.on_text_changed -= self._on_text_changed
            except ValueError:
                # Handler was already removed.
                pass

            self._buffer = None

    def _on_text_changed(
        self,
        _buffer: Buffer,
    ) -> None:
        """
        Reset the inactivity timer whenever the input text changes.
        """
        self._schedule()

    def _schedule(self) -> None:
        """
        Cancel the previous timer and schedule a fresh one.
        """
        if self._handle is not None:
            self._handle.cancel()

        loop = get_app().loop

        self._handle = loop.call_later(
            self._timeout,
            self._fire,
        )

    @staticmethod
    def _fire() -> None:
        """
        Abort the active prompt because the inactivity timeout elapsed.
        """
        get_app().exit(
            exception=_IdleTimeout(),
        )


# Module-level singleton: the single interactive input surface for Citra.
terminal_input = TerminalInput()
