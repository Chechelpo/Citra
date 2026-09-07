# src/citra/commands/clear.py

from __future__ import annotations

from ..utils.terminal import GREEN, RESET
from .command import Command, CommandResult, CommandUsage


class ClearCommand(Command):
    """Clear the conversation history."""

    id = "c"
    usage = CommandUsage(command="c", description="Clear the conversation history.")

    def _run(self, args: str) -> CommandResult:
        """Execute the run operation."""
        return CommandResult(
            output=f"{GREEN}⏺ Cleared conversation{RESET}",
            clear_messages=True,
        )
