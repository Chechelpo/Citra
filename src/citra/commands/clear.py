# src/citra/commands/clear.py

from __future__ import annotations

from ..utils.terminal import GREEN, RESET
from .command import Command, CommandResult


class ClearCommand(Command):
    """Clear the conversation history."""

    id = "c"
    description = "Clear the conversation history."

    def _run(self, args: str) -> CommandResult:
        return CommandResult(
            output=f"{GREEN}⏺ Cleared conversation{RESET}",
            clear_messages=True,
        )
