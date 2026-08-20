# src/citra/commands/help.py

from __future__ import annotations

from .command import Command, CommandResult


class HelpCommand(Command):
    """List available commands."""

    id = "help"
    description = "Show available commands."

    def _run(self, args: str) -> CommandResult:
        # Access the registry through the module-level singleton.
        from .default_registry import COMMAND_REGISTRY

        lines = ["Available commands:"]
        lines.extend(COMMAND_REGISTRY.help_lines())

        return CommandResult(output="\n".join(lines))
