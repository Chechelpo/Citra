# tests/test_debug_command.py

from __future__ import annotations

import unittest

from citra.commands.default_registry import COMMAND_REGISTRY
from citra.commands.debug import DebugCommand
from citra.utils import chat_completions_api


class DebugCommandTests(unittest.TestCase):
    def setUp(self):
        self.command = DebugCommand(None)
        chat_completions_api.DEBUG_PRINTING = True

    def tearDown(self):
        chat_completions_api.DEBUG_PRINTING = True

    def set_flag(self, value: bool) -> None:
        chat_completions_api.DEBUG_PRINTING = value

    def test_is_registered(self):
        self.assertTrue(COMMAND_REGISTRY.contains("debug"))
        instance = COMMAND_REGISTRY.instantiate("debug", None)
        self.assertIsInstance(instance, DebugCommand)

    def test_no_argument_toggles_state(self):
        self.set_flag(True)
        result = self.command.run("")
        self.assertFalse(chat_completions_api.DEBUG_PRINTING)
        self.assertIn("disabled", result.output)

        result = self.command.run("")
        self.assertTrue(chat_completions_api.DEBUG_PRINTING)
        self.assertIn("enabled", result.output)

    def test_on_sets_enabled(self):
        self.set_flag(False)
        result = self.command.run("on")
        self.assertTrue(chat_completions_api.DEBUG_PRINTING)
        self.assertIn("enabled", result.output)

    def test_off_sets_disabled(self):
        result = self.command.run("off")
        self.assertFalse(chat_completions_api.DEBUG_PRINTING)
        self.assertIn("disabled", result.output)

    def test_unknown_action_reports_usage(self):
        result = self.command.run("maybe")
        self.assertEqual(chat_completions_api.DEBUG_PRINTING, True)
        self.assertIn("Usage: /debug [on|off]", result.output)


if __name__ == "__main__":
    unittest.main()
