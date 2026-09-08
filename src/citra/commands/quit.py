# src/citra/commands/quit.py

from __future__ import annotations

from .command import Command, CommandResult, CommandUsage


class QuitCommand(Command):
    """Exit the Citra REPL."""

    id = "q"
    usage = CommandUsage(command="q", description="Exit Citra.")

    def _run(self, args: str) -> CommandResult:
        """Execute the run operation."""
        if args.strip():
            return self.usage_result("Quit takes no arguments.")
        return CommandResult(
            output="Bye.",
            exit=True,
        )
