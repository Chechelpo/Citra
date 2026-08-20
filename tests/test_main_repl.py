# tests/test_main_repl.py

"""
Integration tests for the main REPL after migrating from ``input()`` to
the centralized ``terminal_input`` utility.

The model API and ``terminal_input.prompt`` are mocked so no network or
real terminal is involved.  We verify that slash commands (``/q``,
``/c``, ``/test``, ``exit``) and ordinary text still flow correctly.
"""

import os
import unittest
from unittest import mock


_SRC = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
)
os.environ.setdefault("CITRA_CONFIG_PATH", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".config", "config.toml")
))
os.environ["PYTHONPATH"] = os.path.abspath(_SRC)


from citra import main as citra_main  # noqa: E402
from citra.commands import COMMAND_REGISTRY  # noqa: E402


def run_repl(inputs):
    """
    Run the REPL main loop feeding *inputs* (in order) to
    ``terminal_input.prompt`` and capturing stdout.
    """
    outputs: list[str] = []

    def fake_print(*args, **kwargs):
        outputs.append(" ".join(str(a) for a in args))

    prompt_responses = list(inputs)
    call_api_calls: list = []

    def fake_prompt(*args, **kwargs):
        if not prompt_responses:
            raise EOFError
        return prompt_responses.pop(0)

    def fake_call_api(*args, **kwargs):
        call_api_calls.append(args)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                    }
                }
            ]
        }

    with mock.patch("citra.main.terminal_input") as fake_ti, \
            mock.patch("builtins.print", side_effect=fake_print), \
            mock.patch("citra.main.call_api", side_effect=fake_call_api):
        fake_ti.prompt.side_effect = fake_prompt

        try:
            citra_main.main()
        except EOFError:
            pass

    return outputs, call_api_calls


class CommandTests(unittest.TestCase):
    def test_exit_command_terminates_repl(self):
        outputs, calls = run_repl(["/q"])
        # No model calls should happen for /q.
        self.assertEqual(calls, [])

    def test_exit_alias_terminates_repl(self):
        outputs, calls = run_repl(["/exit"])
        self.assertEqual(calls, [])

    def test_clear_command_does_not_call_model(self):
        outputs, calls = run_repl(["/c", "/q"])
        self.assertEqual(calls, [])

    def test_ordinary_text_enters_agent_loop(self):
        outputs, calls = run_repl(["hello", "/q"])
        # Exactly one model call for the ordinary message.
        self.assertEqual(len(calls), 1)

    def test_empty_input_is_skipped(self):
        outputs, calls = run_repl(["", "   ", "/q"])
        self.assertEqual(calls, [])

    def test_known_commands_are_registered(self):
        for cid in ("q", "c", "help", "test"):
            self.assertTrue(
                COMMAND_REGISTRY.contains(cid),
                f"missing command /{cid}",
            )


if __name__ == "__main__":
    unittest.main()
