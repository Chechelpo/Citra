# tests/test_terminal_input.py

"""
Tests for the centralized terminal-input utility, focused on the
**inactivity timeout** semantics.

The real ``prompt_toolkit`` event loop is not driven here (that would
require a live terminal).  Instead the internal ``_IdleWatchdog`` is
exercised directly with a fake application/loop so the timeout-reset
behaviour is deterministic.
"""

import os
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest import mock

from prompt_toolkit.keys import Keys
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

_SRC = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
)
os.environ["PYTHONPATH"] = os.path.abspath(_SRC)


from citra.cli.input import (
    _COMPOSER_BACKGROUND,
    _COMPOSER_BINDINGS,
    _COMPOSER_STYLE,
    _CommandCompleter,
    _STATUS_BACKGROUND,
    ComposerPrompt,
    TerminalInput,
    _IdleTimeout,
    _IdleWatchdog,
    _PredicateSatisfied,
    terminal_input,
    terminal_ui_state,
)
from citra.cli.repl import _session_footer


class FakeLoop:
    """
    Minimal asyncio-loop stand-in recording scheduled callbacks.

    ``call_later(delay, callback)`` records the callback; the test can
    then ``advance`` time and fire due callbacks in order, mirroring how
    a real loop would dispatch them.
    """

    def __init__(self):
        self._scheduled: list[tuple[float, int, int, callable]] = []
        self._seq = 0
        self.now = 0.0

    def call_later(self, delay, callback):
        self._seq += 1
        handle = FakeHandle()
        self._scheduled.append((self.now + delay, self._seq, handle.seq, callback))
        handle._cancel = lambda: None
        handle._callback = callback
        return handle

    def fire_next(self):
        """Fire the earliest not-yet-fired callback and return True."""
        due = [s for s in self._scheduled if s[2] is not None]
        if not due:
            return False
        due.sort(key=lambda s: (s[0], s[1]))
        _time, _sequence, hseq, cb = due[0]
        # Mark as fired by setting handle seq to None in the record.
        self._scheduled = [
            (a, b, None if c is hseq else c, d) for (a, b, c, d) in self._scheduled
        ]
        cb()
        return True


class FakeHandle:
    """Stand-in for an asyncio.TimerHandle."""

    def __init__(self):
        self._cancelled = False
        self.seq = id(self)

    def cancel(self):
        self._cancelled = True

    def cancelled(self):
        return self._cancelled


class FakeBuffer:
    def __init__(self):
        self.on_text_changed = _Event()

    def _trigger(self):
        self.on_text_changed._fire(self)


class _Event:
    """Very small callable-list event used by FakeBuffer."""

    def __init__(self):
        self._handlers = []

    def __iadd__(self, other):
        self._handlers.append(other)
        return self

    def __isub__(self, other):
        self._handlers.remove(other)
        return self

    def _fire(self, *args):
        for h in self._handlers:
            h(*args)


class FakeApp:
    def __init__(self):
        self.loop = FakeLoop()
        self.layout = mock.Mock()
        self.layout.current_buffer = FakeBuffer()

    def exit(self, exception=None):
        self._exit_exception = exception


def _make_watchdog(timeout):
    app = FakeApp()
    wd = _IdleWatchdog(timeout=timeout)
    patcher = mock.patch(
        "citra.cli.input.get_app",
        return_value=app,
    )
    patcher.start()
    wd.start()
    # Stop the patcher at end-of-test via addCleanup-like behaviour:
    # we attach it to the watchdog for callers to stop.
    wd._test_patcher = patcher
    return wd, app


class IdleWatchdogTests(unittest.TestCase):
    def test_activity_callback_runs_on_open_and_buffer_changes(self):
        activity = mock.Mock()
        app = FakeApp()
        watchdog = _IdleWatchdog(timeout=1.0, on_activity=activity)
        with mock.patch(
            "citra.cli.input.get_app",
            return_value=app,
        ):
            watchdog.start()
            app.layout.current_buffer._trigger()
            watchdog.stop()

        self.assertEqual(activity.call_count, 2)

    def test_timeout_fires_when_idle(self):
        wd, app = _make_watchdog(1.0)

        # One callback is scheduled at start; firing it triggers exit.
        fired = app.loop.fire_next()
        self.assertTrue(fired)
        self.assertIsInstance(app._exit_exception, _IdleTimeout)

        wd._test_patcher.stop()

    def test_activity_resets_timer_no_timeout(self):
        wd, app = _make_watchdog(1.0)

        # Simulate repeated activity: each keystroke reschedules the
        # timer, so the original deadline never fires.
        for _ in range(5):
            app.layout.current_buffer._trigger()

        # No timeout should have fired yet (we never let a deadline run).
        self.assertFalse(hasattr(app, "_exit_exception"))

        wd._test_patcher.stop()

    def test_schedule_cancels_previous_handle(self):
        wd, app = _make_watchdog(1.0)
        first = wd._handle

        app.layout.current_buffer._trigger()

        # The previous handle must have been cancelled.
        self.assertIsNotNone(first)
        self.assertTrue(first.cancelled())
        # A new handle is now active.
        self.assertIsNotNone(wd._handle)
        self.assertIsNot(wd._handle, first)

        wd._test_patcher.stop()


class TerminalInputApiTests(unittest.TestCase):
    def test_composer_footer_identifies_model_source_and_workspace(self):
        application = SimpleNamespace(
            config=SimpleNamespace(
                model=lambda: SimpleNamespace(name="fast", id="gpt-test")
            ),
            workspace=SimpleNamespace(
                source_workspace=Path("/work/project"),
                workspace=Path("/work/runtime"),
                runtime_id="citra-process-12-abcd",
            ),
        )

        footer = _session_footer(application).render()

        self.assertIn("model: fast (gpt-test)", footer)
        self.assertIn("source: /work/project", footer)
        self.assertIn("workspace: /work/runtime", footer)

    def test_enter_submits_the_composer_buffer(self):
        buffer = mock.Mock()
        enter = next(
            binding
            for binding in _COMPOSER_BINDINGS.bindings
            if binding.keys == (Keys.ControlM,)
        )

        enter.handler(SimpleNamespace(current_buffer=buffer))

        buffer.validate_and_handle.assert_called_once_with()
        buffer.insert_text.assert_not_called()

    def test_escape_enter_inserts_a_newline(self):
        buffer = mock.Mock()
        escape_enter = next(
            binding
            for binding in _COMPOSER_BINDINGS.bindings
            if binding.keys == (Keys.Escape, Keys.ControlM)
        )

        escape_enter.handler(SimpleNamespace(current_buffer=buffer))

        buffer.insert_text.assert_called_once_with("\n")
        buffer.validate_and_handle.assert_not_called()

    def test_boxed_prompt_draws_a_composer_border(self):
        ti = TerminalInput()
        terminal_ui_state.reset()
        erase_values = []

        def submit(*_arguments, **_keywords):
            erase_values.append(ti._session.app.erase_when_done)
            return "/help"

        with (
            mock.patch.object(ti._session, "prompt", side_effect=submit) as prompt,
            mock.patch("citra.cli.input._console.print") as output,
        ):
            result = ti.prompt(
                "› ",
                boxed=True,
                footer=(
                    "model: default (gpt-test)  ·  source: ~/Code/Citra  ·  "
                    "process: citra-process-12-abcd"
                ),
        )

        self.assertEqual(result, "/help")
        header = prompt.call_args.args[0]()
        header_text = "".join(fragment[1] for fragment in header)
        self.assertIn("• Ready", header_text)
        self.assertIn("─" * 24, header_text)
        placeholder = prompt.call_args.kwargs["placeholder"]
        placeholder_text = "".join(fragment[1] for fragment in placeholder)
        self.assertEqual(placeholder_text, "› ")
        toolbar = prompt.call_args.kwargs["bottom_toolbar"]()
        toolbar_text = "".join(fragment[1] for fragment in toolbar)
        self.assertTrue(toolbar_text.startswith("│"))
        self.assertIn("\n" + "─" * 24, toolbar_text)
        self.assertIn("model: default (gpt-test)", toolbar_text)
        self.assertTrue(toolbar_text.endswith("…"))
        self.assertNotIn("rprompt", prompt.call_args.kwargs)
        self.assertIn("style", prompt.call_args.kwargs)
        self.assertTrue(prompt.call_args.kwargs["multiline"])
        self.assertEqual(prompt.call_args.kwargs["reserve_space_for_menu"], 0)
        self.assertEqual(erase_values, [True])
        self.assertFalse(ti._session.app.erase_when_done)
        self.assertIn("key_bindings", prompt.call_args.kwargs)
        continuation = prompt.call_args.kwargs["prompt_continuation"]
        self.assertEqual("".join(fragment[1] for fragment in continuation), "│ · ")
        self.assertEqual(output.call_count, 4)
        self.assertTrue(output.call_args_list[0].args[0].plain.startswith("│"))
        self.assertEqual(output.call_args_list[1].args[0].plain.strip(), "│   /help")
        persisted_width = len(output.call_args_list[0].args[0].plain)
        self.assertEqual(len(output.call_args_list[1].args[0].plain), persisted_width)
        self.assertTrue(output.call_args_list[2].args[0].plain.startswith("│"))
        self.assertEqual(
            output.call_args_list[3].args[0].plain,
            "─" * persisted_width,
        )
        submitted = "\n".join(call.args[0].plain for call in output.call_args_list)
        self.assertNotIn("model: default", submitted)
        self.assertTrue(submitted.endswith("─" * persisted_width))

    def test_prompt_and_status_toolbar_have_distinct_backgrounds(self):
        self.assertNotEqual(_COMPOSER_BACKGROUND, _STATUS_BACKGROUND)
        toolbar = _COMPOSER_STYLE.get_attrs_for_style_str(
            "class:bottom-toolbar"
        )
        composer = _COMPOSER_STYLE.get_attrs_for_style_str("")
        margin = _COMPOSER_STYLE.get_attrs_for_style_str(
            "class:composer-margin"
        )
        self.assertNotEqual(toolbar.bgcolor, composer.bgcolor)
        self.assertEqual(margin.bgcolor, composer.bgcolor)

    def test_prompt_with_idle_timeout_returns_none_on_idle_timeout(self):
        """
        Drive a PromptSession prompt where the watchdog fires an
        _IdleTimeout; the public API must convert that into None.
        """
        ti = TerminalInput()

        def fake_prompt(*args, **kwargs):
            raise _IdleTimeout()

        with mock.patch.object(ti._session, "prompt", side_effect=fake_prompt):
            result = ti.prompt_with_idle_timeout(1.0, "q> ")

        self.assertIsNone(result)

    def test_module_singleton_exists(self):
        self.assertIsInstance(terminal_input, TerminalInput)

    def test_command_completions_come_from_declared_usage(self):
        completer = _CommandCompleter()

        def suggestions(text: str) -> set[str]:
            return {
                completion.text
                for completion in completer.get_completions(
                    Document(text, cursor_position=len(text)),
                    CompleteEvent(text_inserted=True),
                )
            }

        self.assertIn("/model", suggestions("/mod"))
        self.assertEqual(suggestions("/model sh"), {"show"})
        self.assertEqual(suggestions("/debug o"), {"off", "on"})
        self.assertEqual(
            suggestions("/apply --f"),
            {"--force", "--force-conflicts"},
        )
        self.assertEqual(suggestions("/model set --profile p"), set())
        self.assertIn("host", suggestions("/model set --profile primary h"))

    def test_steering_completions_only_include_available_commands(self):
        completer = _CommandCompleter(("agent", "memory", "workflow"))
        suggestions = {
            completion.text
            for completion in completer.get_completions(
                Document("/", cursor_position=1),
                CompleteEvent(text_inserted=True),
            )
        }

        self.assertEqual(suggestions, {"/agent", "/memory", "/workflow"})

    def test_prompt_until_returns_none_when_background_state_changes(self):
        ti = TerminalInput()

        with mock.patch.object(
            ti._session,
            "prompt",
            side_effect=_PredicateSatisfied(),
        ):
            result = ti.prompt_until(lambda: True, "steer> ")

        self.assertIsNone(result)

    def test_prompt_until_uses_the_same_boxed_composer(self):
        ti = TerminalInput()

        with (
            mock.patch.object(
                ti._session,
                "prompt",
                side_effect=_PredicateSatisfied(),
            ) as prompt,
            mock.patch("citra.cli.input._console.print"),
        ):
            result = ti.prompt_until(
                lambda: True,
                "› ",
                boxed=True,
                footer="model: test · source: /project · process: citra-process-1",
            )

        self.assertIsNone(result)
        self.assertTrue(prompt.call_args.kwargs["multiline"])
        toolbar = prompt.call_args.kwargs["bottom_toolbar"]()
        toolbar_text = "".join(fragment[1] for fragment in toolbar)
        self.assertIn("model: test", toolbar_text)

    def test_dynamic_status_keeps_the_composer_bottom_height_fixed(self):
        footer = "model: test · source: /project · workspace: /runtime"
        terminal_ui_state.finish_working()
        terminal_ui_state.record_tokens(
            input_tokens=12,
            cached_tokens=9,
            output_tokens=4,
        )
        idle = "".join(
            fragment[1]
            for fragment in terminal_ui_state.composer_header(width=80)
        )

        terminal_ui_state.begin_working("Working")
        active = "".join(
            fragment[1]
            for fragment in terminal_ui_state.composer_header(width=80)
        )
        terminal_ui_state.finish_working()

        self.assertEqual(idle.count("\n"), active.count("\n"))
        self.assertIn("in: 12 (hit 75%) · out: 4", idle)
        self.assertIn("Working for", active)

    def test_model_debug_keeps_only_latest_request(self):
        terminal_ui_state.reset()
        terminal_ui_state.record_model_debug("Starting model request one")
        terminal_ui_state.record_model_debug("Model HTTP 200 received")
        terminal_ui_state.record_model_debug("Model finish_reason(s): stop")
        first = "".join(
            fragment[1]
            for fragment in terminal_ui_state.composer_header(width=100)
        )
        self.assertIn("Starting model request one", first)

        terminal_ui_state.record_model_debug("Starting model request two")
        terminal_ui_state.record_model_debug("Model HTTP 429 received")
        second = "".join(
            fragment[1]
            for fragment in terminal_ui_state.composer_header(width=100)
        )

        self.assertNotIn("request one", second)
        self.assertNotIn("finish_reason(s): stop", second)
        self.assertIn("Starting model request two", second)
        self.assertIn("Model HTTP 429 received", second)
        self.assertEqual(first.count("\n"), second.count("\n"))
        terminal_ui_state.reset()

    def test_initial_and_steering_use_the_same_composer_object(self):
        initial = ComposerPrompt("/help or type your first message", "footer")
        steering = ComposerPrompt("enter steering", "footer")

        self.assertEqual(initial.footer, steering.footer)
        self.assertEqual(initial.__class__, steering.__class__)

    def test_mock_full_session_preserves_composer_shape_across_input_and_work(self):
        ti = TerminalInput()
        footer = "model: fast · source: /project · workspace: /runtime"
        terminal_ui_state.finish_working()
        terminal_ui_state.record_tokens(input_tokens=0, output_tokens=0)

        with (
            mock.patch.object(
                ti._session,
                "prompt",
                side_effect=["Implement the feature", _PredicateSatisfied()],
            ) as prompt,
            mock.patch("citra.cli.input._console.print"),
        ):
            first_message = ti.prompt(
                "/help or type your first message",
                boxed=True,
                footer=footer,
            )
            initial_header = prompt.call_args_list[0].args[0]()

            terminal_ui_state.begin_working("Working")
            terminal_ui_state.record_tokens(input_tokens=123, output_tokens=45)
            working_header = prompt.call_args_list[0].args[0]()
            terminal_ui_state.finish_working()

            steering = ti.prompt_until(
                lambda: True,
                "enter steering",
                boxed=True,
                footer=footer,
            )

        initial = "".join(fragment[1] for fragment in initial_header)
        active = "".join(fragment[1] for fragment in working_header)
        self.assertEqual(first_message, "Implement the feature")
        self.assertIsNone(steering)
        first_placeholder = prompt.call_args_list[0].kwargs["placeholder"]
        steering_placeholder = prompt.call_args_list[1].kwargs["placeholder"]
        self.assertIn(
            "/help or type your first message",
            "".join(fragment[1] for fragment in first_placeholder),
        )
        self.assertIn(
            "enter steering",
            "".join(fragment[1] for fragment in steering_placeholder),
        )
        self.assertEqual(initial.count("\n"), active.count("\n"))
        self.assertIn("Working for", active)
        self.assertIn("in: 123 (hit 0%) · out: 45", active)


if __name__ == "__main__":
    unittest.main()
