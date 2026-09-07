# src/citra/cli/input.py

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

Expected TUI format:
User prompt area:

<Working animation> Worked for <time> seconds | in : <in_tokens> ; out: <out_tokens>s
---

|
|   <placeholder text that disappears once user starts typing>
|

---
<footer>
"""

from __future__ import annotations

from asyncio import TimerHandle
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from threading import RLock
from time import perf_counter

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app, get_app_or_none
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.styles import Style
from rich.text import Text

from .theme import BACKGROUND, SURFACE, console

__all__ = [
    "BottomStatus",
    "ComposerPrompt",
    "TerminalInput",
    "TerminalUiState",
    "terminal_input",
    "terminal_ui_state",
]

_console = console
_COMPOSER_BACKGROUND = SURFACE
_STATUS_BACKGROUND = BACKGROUND
_PLACEHOLDER_FOREGROUND = "#343434"
_COMPOSER_STYLE = Style.from_dict(
    {
        "": f"bg:{_COMPOSER_BACKGROUND}",
        "bottom-toolbar": (f"bg:{_STATUS_BACKGROUND} #8a8a8a noreverse"),
        "composer.activity": f"bg:{_STATUS_BACKGROUND} bold #7aa2f7",
        "composer.body": f"bg:{_COMPOSER_BACKGROUND}",
        "composer.divider": f"bg:{_STATUS_BACKGROUND} #555555",
        "composer.footer": f"bg:{_STATUS_BACKGROUND} #8a8a8a",
        "composer.placeholder": (
            f"bg:{_COMPOSER_BACKGROUND} {_PLACEHOLDER_FOREGROUND}"
        ),
        "selection": "bg:#3b4261 #ffffff",
        "prompt": f"bg:{_COMPOSER_BACKGROUND} bold #7aa2f7",
    }
)


def _composer_bindings() -> KeyBindings:
    """Return Codex-like multiline bindings: Enter sends, Esc+Enter adds a line."""
    bindings = KeyBindings()

    @bindings.add("enter", eager=True)
    def _submit(event: KeyPressEvent) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    return bindings


_COMPOSER_BINDINGS = _composer_bindings()


@dataclass(frozen=True)
class BottomStatus:
    """Represent stable model activity and token counts below the composer."""

    working_label: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    started_at: float | None = None
    elapsed_seconds: float | None = None


@dataclass(frozen=True)
class ComposerPrompt:
    """Describe one stable prompt composer without mode-specific UI code."""

    placeholder: str
    footer: str
    command_ids: tuple[str, ...] | None = None


class TerminalUiState:
    """Own the prompt-toolkit bottom area while model work runs in another thread."""

    def __init__(self) -> None:
        """Initialize an idle bottom status with no model token accounting."""
        self._lock = RLock()
        self._status = BottomStatus()

    def begin_working(self, label: str) -> None:
        """Show a stable working indicator without a terminal refresh loop."""
        with self._lock:
            self._status = BottomStatus(
                working_label=label,
                input_tokens=self._status.input_tokens,
                output_tokens=self._status.output_tokens,
                started_at=perf_counter(),
            )
        self._invalidate()

    def reset(self) -> None:
        """Reset activity and token accounting for a newly rendered session."""
        with self._lock:
            self._status = BottomStatus()
        self._invalidate()

    def finish_working(self) -> None:
        """Clear the model-working indicator after a request completes."""
        with self._lock:
            elapsed_seconds = (
                perf_counter() - self._status.started_at
                if self._status.started_at is not None
                else self._status.elapsed_seconds
            )
            self._status = BottomStatus(
                input_tokens=self._status.input_tokens,
                output_tokens=self._status.output_tokens,
                elapsed_seconds=elapsed_seconds,
            )
        self._invalidate()

    def record_tokens(self, *, input_tokens: int, output_tokens: int) -> None:
        """Replace the latest model request's visible input/output token counts."""
        with self._lock:
            self._status = BottomStatus(
                working_label=self._status.working_label,
                input_tokens=max(0, input_tokens),
                output_tokens=max(0, output_tokens),
                started_at=self._status.started_at,
                elapsed_seconds=self._status.elapsed_seconds,
            )
        self._invalidate()

    def composer_header(self, *, width: int) -> FormattedText:
        """Render dynamic activity above the fixed-height composer body."""
        with self._lock:
            status = self._status
        activity = self._activity_text(status)
        tokens = f"in: {status.input_tokens:,} · out: {status.output_tokens:,}"
        status_line = _fit_toolbar_line(f"  {activity}  |  {tokens}", width)
        divider = "─" * width
        return FormattedText(
            (
                ("class:composer.activity", status_line + "\n"),
                ("class:composer.divider", divider + "\n"),
                ("class:composer.body", "│" + " " * (width - 1) + "\n"),
                ("class:composer.body", "│   "),
            )
        )

    def composer_footer(self, footer: str, *, width: int) -> FormattedText:
        """Render the bottom margin, divider, and environment footer."""
        return FormattedText(
            (
                ("class:composer.body", "│" + " " * (width - 1) + "\n"),
                ("class:composer.divider", "─" * width + "\n"),
                ("class:composer.footer", _fit_toolbar_line(f"  {footer}", width)),
            )
        )

    @staticmethod
    def _activity_text(status: BottomStatus) -> str:
        """Describe live or most-recent work while keeping one text row."""
        if status.working_label is not None and status.started_at is not None:
            frame = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[
                int(perf_counter() * 8) % 10
            ]
            elapsed = _format_elapsed(perf_counter() - status.started_at)
            return f"{frame} {status.working_label} for {elapsed}"
        if status.elapsed_seconds is not None:
            return f"• Worked for {_format_elapsed(status.elapsed_seconds)}"
        return "• Ready"

    @staticmethod
    def _invalidate() -> None:
        """Request a safe prompt-toolkit redraw when an interactive app exists."""
        application = get_app_or_none()
        if application is not None:
            application.invalidate()


terminal_ui_state = TerminalUiState()


def _fit_toolbar_line(text: str, width: int) -> str:
    """Keep dynamic toolbar text on one row so its compositor height is fixed."""
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def _format_elapsed(seconds: float) -> str:
    """Format a compact duration for the model activity row."""
    total = max(0, round(seconds))
    minutes, remainder = divmod(total, 60)
    return f"{minutes}m {remainder}s" if minutes else f"{remainder}s"


class _CommandCompleter(Completer):
    """Suggest slash commands from their centralized usage declarations."""

    def __init__(self, command_ids: tuple[str, ...] | None = None) -> None:
        """Optionally restrict suggestions to commands valid in this prompt."""
        self._command_ids = frozenset(command_ids) if command_ids is not None else None

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterator[Completion]:
        """Yield command, form, and enumerated argument completions."""
        del complete_event
        text = document.text_before_cursor
        if not text.startswith("/") or "\n" in text:
            return

        from ..commands.default_registry import COMMAND_REGISTRY

        usages = tuple(
            usage
            for usage in COMMAND_REGISTRY.usages
            if self._command_ids is None or usage.command in self._command_ids
        )
        body = text[1:]
        command_text, separator, remainder = body.partition(" ")
        if not separator:
            for usage in usages:
                candidate = f"/{usage.command}"
                if candidate.startswith(text):
                    yield Completion(
                        candidate,
                        start_position=-len(text),
                        display_meta=usage.description,
                    )
            return

        usage = next(
            (
                item
                for item in usages
                if item.command == command_text
            ),
            None,
        )
        if usage is None:
            return

        trailing_space = remainder.endswith(" ")
        tokens = remainder.split()
        prefix = "" if trailing_space else (tokens[-1] if tokens else "")
        completed = tokens if trailing_space else tokens[:-1]
        candidates: dict[str, str] = {}
        for form in usage.forms:
            matched_path = 0
            while (
                matched_path < len(form.path)
                and matched_path < len(completed)
                and completed[matched_path] == form.path[matched_path]
            ):
                matched_path += 1
            if matched_path < min(len(form.path), len(completed)):
                continue
            if matched_path < len(form.path):
                candidate = form.path[matched_path]
                candidates[candidate] = form.description
                continue

            invocation_tokens = completed[len(form.path):]
            options = {option.flag: option for option in form.options}
            used_options: set[str] = set()
            positional_tokens: list[str] = []
            waiting_for_option_value = None
            token_index = 0
            while token_index < len(invocation_tokens):
                token = invocation_tokens[token_index]
                option = options.get(token)
                if option is None:
                    positional_tokens.append(token)
                    token_index += 1
                    continue
                used_options.add(option.flag)
                if option.value is not None:
                    if token_index + 1 >= len(invocation_tokens):
                        waiting_for_option_value = option
                        break
                    token_index += 1
                token_index += 1

            if waiting_for_option_value is not None:
                for candidate in waiting_for_option_value.value_suggestions:
                    candidates[candidate] = waiting_for_option_value.description
                continue

            for option in form.options:
                if option.flag not in used_options:
                    candidates[option.flag] = option.description

            argument_index = len(positional_tokens)
            if argument_index < len(form.arguments):
                argument = form.arguments[argument_index]
                for candidate in argument.suggestions:
                    candidates[candidate] = argument.description

        for candidate, description in sorted(candidates.items()):
            if candidate.startswith(prefix):
                yield Completion(
                    candidate,
                    start_position=-len(prefix),
                    display_meta=description,
                )


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
        self._session: PromptSession[str] = PromptSession(
            history=InMemoryHistory(),
            auto_suggest=AutoSuggestFromHistory(),
        )

    def prompt(
        self,
        message: str = "",
        *,
        boxed: bool = False,
        footer: str = "",
        command_ids: tuple[str, ...] | None = None,
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
        if not boxed:
            return self._session.prompt(
                ANSI(message),
                handle_sigint=True,
            )

        return self._prompt_composer(ComposerPrompt(message, footer, command_ids))

    def _prompt_composer(
        self,
        composer: ComposerPrompt,
        *,
        pre_run: Callable[[], None] | None = None,
    ) -> str:
        """Read from the shared boxed composer with a distinct status toolbar."""
        width = max(24, _console.size.width)
        previous_erase_when_done = self._session.app.erase_when_done
        self._session.app.erase_when_done = True
        try:
            result = self._session.prompt(
                lambda: terminal_ui_state.composer_header(width=width),
                placeholder=FormattedText(
                    (("class:composer.placeholder", composer.placeholder),)
                ),
                bottom_toolbar=lambda: terminal_ui_state.composer_footer(
                    composer.footer,
                    width=width,
                ),
                style=_COMPOSER_STYLE,
                completer=_CommandCompleter(composer.command_ids),
                complete_while_typing=True,
                reserve_space_for_menu=0,
                multiline=True,
                key_bindings=_COMPOSER_BINDINGS,
                prompt_continuation=FormattedText(
                    (("class:composer.body", "│ · "),)
                ),
                refresh_interval=0.125,
                wrap_lines=False,
                pre_run=pre_run,
                handle_sigint=True,
            )
        finally:
            self._session.app.erase_when_done = previous_erase_when_done
        self._render_submitted_section(result, width=width)
        return result

    @staticmethod
    def _render_submitted_section(content: str, *, width: int) -> None:
        """Persist only an accepted prompt's body after its live area is erased."""
        surface_style = f"on {_COMPOSER_BACKGROUND}"
        _console.print(
            Text("│" + " " * (width - 1), style=surface_style),
            soft_wrap=True,
        )
        for line in content.splitlines() or [""]:
            _console.print(
                Text(_fit_toolbar_line(f"│   {line}", width), style=surface_style),
                soft_wrap=True,
            )
        _console.print(Text("│" + " " * (width - 1), style=surface_style))

    def prompt_with_idle_timeout(
        self,
        timeout: float,
        message: str = "",
        *,
        on_activity: Callable[[], None] | None = None,
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
            on_activity:
                Optional callback invoked when the prompt opens and whenever
                its buffer changes.

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
            raise ValueError("'timeout' must be greater than zero.")

        watchdog = _IdleWatchdog(
            timeout=timeout,
            on_activity=on_activity,
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
        boxed: bool = False,
        footer: str = "",
        command_ids: tuple[str, ...] | None = None,
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
            if boxed:
                return self._prompt_composer(
                    ComposerPrompt(message, footer, command_ids),
                    pre_run=watcher.start,
                )
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
        loop = get_app().loop
        if loop is None:
            raise RuntimeError("Prompt event loop is unavailable.")
        self._handle = loop.call_later(
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
        on_activity: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the instance."""
        self._timeout = timeout
        self._on_activity = on_activity

        self._handle: TimerHandle | None = None
        self._buffer: Buffer | None = None

    def start(self) -> None:
        """
        Attach to the active prompt buffer and start the inactivity timer.

        Intended to be passed as ``PromptSession.prompt(pre_run=...)``.
        """
        app = get_app()

        self._buffer = app.layout.current_buffer
        if self._buffer is None:
            raise RuntimeError("Prompt input buffer is unavailable.")
        self._buffer.on_text_changed += self._on_text_changed

        self._record_activity()
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
        self._record_activity()
        self._schedule()

    def _record_activity(self) -> None:
        """Notify the owner that the user is still engaging with the prompt."""
        if self._on_activity is not None:
            self._on_activity()

    def _schedule(self) -> None:
        """
        Cancel the previous timer and schedule a fresh one.
        """
        if self._handle is not None:
            self._handle.cancel()

        loop = get_app().loop
        if loop is None:
            raise RuntimeError("Prompt event loop is unavailable.")

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
