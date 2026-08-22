# tests/test_prompt_user.py

"""
Unit tests for the ``PromptUser`` tool.

The centralized terminal-input utility (``citra.utils.terminal_input``)
is mocked so these tests never touch a real terminal.
"""

import os
from types import SimpleNamespace
import unittest
from unittest import mock

# Make the Citra package importable regardless of cwd, mirroring start.sh.
_SRC = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
)
os.environ.setdefault("CITRA_CONFIG_PATH", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".config", "config.toml")
))
os.environ["PYTHONPATH"] = os.path.abspath(_SRC)


from citra.tools.transient.prompt_user import (  # noqa: E402
    PromptUser,
    USER_UNAVAILABLE_MESSAGE,
)


def _make_prompt_user(answer) -> PromptUser:
    context = SimpleNamespace(user_interactions=None)
    tool = PromptUser(context)
    if isinstance(answer, BaseException):
        tool._execute  # ensure bound
    return tool


def _run(arguments, answer):
    """
    Instantiate PromptUser and run _execute with the terminal-input
    utility mocked to return *answer* (or raise it if it is an
    exception).
    """
    context = SimpleNamespace(user_interactions=None)
    tool = PromptUser(context)

    with mock.patch(
        "citra.tools.transient.prompt_user.terminal_input"
    ) as fake_ti:
        if isinstance(answer, BaseException):
            fake_ti.prompt_with_idle_timeout.side_effect = answer
        else:
            fake_ti.prompt_with_idle_timeout.return_value = answer

        return tool._execute(arguments)


class PlainTextTests(unittest.TestCase):
    def _args(self, question="Which database?", **extra):
        args = {"question": question}
        args.update(extra)
        return args

    def test_plain_text_answer(self):
        result = _run(self._args(), "postgres")
        self.assertEqual(result, "postgres")

    def test_empty_response(self):
        result = _run(self._args(), "   ")
        self.assertEqual(result, "(empty response)")

    def test_strips_whitespace(self):
        result = _run(self._args(), "  redis  ")
        self.assertEqual(result, "redis")


class OptionListTests(unittest.TestCase):
    def _args(self, options, question="Which implementation?"):
        return {"question": question, "options": options}

    def test_numeric_selection(self):
        result = _run(self._args(["A", "B", "C"]), "2")
        self.assertEqual(result, "B")

    def test_numeric_first(self):
        result = _run(self._args(["foo", "bar"]), "1")
        self.assertEqual(result, "foo")

    def test_free_form_override(self):
        result = _run(self._args(["A", "B"]), "custom answer")
        self.assertEqual(result, "custom answer")

    def test_out_of_range_int_is_free_form(self):
        # An out-of-range integer stays a free-form answer.
        result = _run(self._args(["foo", "bar"]), "7")
        self.assertEqual(result, "7")

    def test_exact_text_match_returns_text(self):
        # "foo" is not a number, so it is returned as-is.
        result = _run(self._args(["foo", "bar"]), "foo")
        self.assertEqual(result, "foo")

    def test_options_are_stripped(self):
        result = _run(self._args(["  spaced  "]), "1")
        self.assertEqual(result, "spaced")


class TimeoutTests(unittest.TestCase):
    def _args(self, **extra):
        args = {"question": "Are you there?"}
        args.update(extra)
        return args

    def test_inactivity_timeout_returns_user_unavailable(self):
        result = _run(self._args(), None)
        self.assertEqual(result, USER_UNAVAILABLE_MESSAGE)

    def test_invalid_timeout_zero(self):
        with self.assertRaises(ValueError):
            _run(self._args(timeout=0), "x")

    def test_invalid_timeout_negative(self):
        with self.assertRaises(ValueError):
            _run(self._args(timeout=-5), "x")


class ValidationTests(unittest.TestCase):
    def test_blank_question_rejected(self):
        with self.assertRaises(ValueError):
            _run({"question": "   "}, "x")

    def test_empty_options_rejected(self):
        with self.assertRaises(ValueError):
            _run({"question": "q?", "options": ["   "]}, "x")

    def test_whitespace_only_options_rejected(self):
        with self.assertRaises(ValueError):
            _run({"question": "q?", "options": ["  ", "\t"]}, "x")


class InterruptTests(unittest.TestCase):
    def test_ctrl_c_propagates(self):
        with self.assertRaises(KeyboardInterrupt):
            _run({"question": "q?"}, KeyboardInterrupt())

    def test_ctrl_d_propagates(self):
        with self.assertRaises(EOFError):
            _run({"question": "q?"}, EOFError())


if __name__ == "__main__":
    unittest.main()
