# src/citra/commands/clear.py

from __future__ import annotations

from .command import Command, CommandResult, CommandUsage


class ClearCommand(Command):
    """Clear the conversation history."""

    id = "c"
    usage = CommandUsage(command="c", description="Clear the conversation history.")

    def _run(self, args: str) -> CommandResult:
        """Execute the run operation."""
        if args.strip():
            return self.usage_result("Clear takes no arguments.")
        return CommandResult(
            output="⏺ Cleared conversation",
            clear_messages=True,
        )
