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
import unittest
from unittest import mock


_SRC = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
)
os.environ["PYTHONPATH"] = os.path.abspath(_SRC)


from citra.utils.terminal_input import (  # noqa: E402
    _IdleTimeout,
    _IdleWatchdog,
    terminal_input,
    TerminalInput,
)


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
        self._scheduled.append(
            (self.now + delay, self._seq, handle.seq, callback)
        )
        handle._cancel = lambda: None
        handle._callback = callback
        return handle

    def fire_next(self):
        """Fire the earliest not-yet-fired callback and return True."""
        due = [s for s in self._scheduled if s[2] is not None]
        if not due:
            return False
        due.sort(key=lambda s: (s[0], s[1]))
        t, seq, hseq, cb = due[0]
        # Mark as fired by setting handle seq to None in the record.
        self._scheduled = [
            (a, b, None if c is hseq else c, d)
            for (a, b, c, d) in self._scheduled
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
        "citra.utils.terminal_input.get_app",
        return_value=app,
    )
    patcher.start()
    wd.start()
    # Stop the patcher at end-of-test via addCleanup-like behaviour:
    # we attach it to the watchdog for callers to stop.
    wd._test_patcher = patcher
    return wd, app


class IdleWatchdogTests(unittest.TestCase):
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
    def test_prompt_with_idle_timeout_returns_none_on_idle_timeout(self):
        """
        Drive a PromptSession prompt where the watchdog fires an
        _IdleTimeout; the public API must convert that into None.
        """
        ti = TerminalInput()

        def fake_prompt(*args, **kwargs):
            raise _IdleTimeout()

        with mock.patch.object(
            ti._session, "prompt", side_effect=fake_prompt
        ):
            result = ti.prompt_with_idle_timeout(1.0, "q> ")

        self.assertIsNone(result)

    def test_module_singleton_exists(self):
        self.assertIsInstance(terminal_input, TerminalInput)


if __name__ == "__main__":
    unittest.main()
