# src/citra/commands/debug.py

from __future__ import annotations

from typing import override

from ..utils import chat_completions_api
from .command import Command, CommandResult


class DebugCommand(Command):
    """
    Toggle the grey diagnostic lines printed around model requests.

    With no argument the current state is flipped. ``on`` and ``off``
    set the state explicitly.
    """

    id = "debug"
    description = "toggle grey model-request debug output"

    @override
    def _run(self, args: str) -> CommandResult:
        """Execute the run operation."""
        action = args.strip().lower()

        if action == "on":
            chat_completions_api.set_debug_printing(True)
        elif action == "off":
            chat_completions_api.set_debug_printing(False)
        elif action:
            return CommandResult(
                f"Unknown debug action: {action}\n\n"
                f"{self._usage()}"
            )
        else:
            chat_completions_api.set_debug_printing(
                not chat_completions_api.debug_printing_enabled()
            )

        state = (
            "enabled"
            if chat_completions_api.debug_printing_enabled()
            else "disabled"
        )
        return CommandResult(output=f"Debug printing {state}.")

    @staticmethod
    def _usage() -> str:
        """Handle usage."""
        return "Usage: /debug [on|off]"
