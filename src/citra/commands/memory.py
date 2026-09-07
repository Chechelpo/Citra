"""CLI inspection of the current conversation's structured memory."""

from citra.utils.chat_completions_api import build_memory_context

from .command import Command, CommandForm, CommandResult, CommandUsage


class MemoryCommand(Command):
    """Print structured session memory on demand."""

    id = "memory"
    usage = CommandUsage(
        command="memory",
        description="Show the current structured session memory.",
        forms=(CommandForm(path=("show",), description="Display all memory records."),),
    )

    def _run(self, args: str) -> CommandResult:
        """Render the active session's complete structured memory."""
        if args.strip() not in {"", "show", "status"}:
            return self.usage_result("Unknown memory action.")
        session = self.context.session
        if session is None:
            return CommandResult(output="Session memory is unavailable.")
        rendered = build_memory_context(session.memory.values())
        return CommandResult(
            output=rendered or "# Conversation Memory\n\n(empty)"
        )


__all__ = ["MemoryCommand"]
