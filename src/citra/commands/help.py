# src/citra/commands/help.py

from __future__ import annotations

from .command import Command, CommandResult, CommandUsage


class HelpCommand(Command):
    """List available commands."""

    id = "help"
    usage = CommandUsage(command="help", description="Show available commands.")

    def _run(self, args: str) -> CommandResult:
        # Access the registry through the module-level singleton.
        """Execute the run operation."""
        from .default_registry import COMMAND_REGISTRY

        return CommandResult(usage=COMMAND_REGISTRY.usages)
