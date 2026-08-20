# src/citra/commands/quit.py

from __future__ import annotations

from .command import Command, CommandResult


class QuitCommand(Command):
    """Exit the Citra REPL."""

    id = "q"
    description = "Exit Citra."

    def _run(self, args: str) -> CommandResult:
        return CommandResult(
            output="Bye.",
            exit=True,
        )
