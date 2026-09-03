# src/citra/commands/command.py

"""
Command framework for Citra's REPL.

A *command* is a slash-prefixed action typed by the user at the
interactive prompt, for example ``/test`` or ``/clear``.

This module defines the abstract :class:`Command` base class, the
:class:`CommandResult` dataclass returned by every command, and the
:class:`CommandRegistry` that maps command ids to command classes.

The design intentionally mirrors the tool framework:

- Registry stores *classes*, not instances.
- Instances are created for a single invocation and discarded.
- Each command receives the :class:`ExecutionContext`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

from ..context import ExecutionContext


@dataclass(frozen=True)
class CommandResult:
    """
    Outcome of running a command.

    Attributes
    ----------
    output:
        Text to print to the terminal. May be empty.
    clear_messages:
        If ``True``, the conversation history is cleared after the
        command runs. Used by ``/clear`` and similar maintenance
        commands.
    exit:
        If ``True``, the REPL terminates after the command runs.
        Used by ``/quit``.
    """

    output: str = ""
    clear_messages: bool = False
    exit: bool = False


class Command(ABC):
    """
    Abstract base for all REPL commands.

    Subclasses must set :attr:`id` (the string the user types without
    the leading ``/``, e.g. ``"test"``) and :attr:`description`, and
    implement :meth:`_run`.
    """

    id: str = ""
    description: str = ""

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize the instance."""
        self._context = context

    @property
    def context(self) -> ExecutionContext:
        """Handle context."""
        return self._context

    @final
    def run(self, args: str) -> CommandResult:
        """
        Execute the command.

        ``args`` is the raw text typed *after* the command name, with
        leading/trailing whitespace already stripped. For most commands
        it is empty.

        Errors raised by :meth:`_run` are caught here and converted into
        error :class:`CommandResult` so the REPL never crashes.
        """
        try:
            return self._run(args)
        except Exception as error:  # noqa: BLE001
            from ..utils.terminal import RED, RESET

            return CommandResult(
                output=f"{RED}⏺ Command error: {error}{RESET}",
            )

    @abstractmethod
    def _run(self, args: str) -> CommandResult:
        """Command-specific logic. Override in subclasses."""
        ...


class CommandRegistry:
    """
    Permanent registry of command implementations.

    The registry stores :class:`Command` *classes*, not instances.
    Instances are created for a single invocation via
    :meth:`instantiate`.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.__commands: dict[str, type[Command]] = {}

    def register(
        self,
        command_id: str,
        command_type: type[Command],
    ) -> None:
        """Handle register."""
        if command_id in self.__commands:
            raise ValueError(
                f"Command '{command_id}' is already registered."
            )

        if not command_id:
            raise ValueError("Command id cannot be empty.")

        self.__commands[command_id] = command_type

    def instantiate(
        self,
        command_id: str,
        context: ExecutionContext,
    ) -> Command | None:
        """
        Create one command instance, or ``None`` if the id is unknown.
        """
        command_type = self.__commands.get(command_id)

        if command_type is None:
            return None

        return command_type(context)

    def contains(self, command_id: str) -> bool:
        """Handle contains."""
        return command_id in self.__commands

    @property
    def command_ids(self) -> tuple[str, ...]:
        """Handle command ids."""
        return tuple(self.__commands)

    def help_lines(self) -> list[str]:
        """
        Return ``(id, description)`` pairs sorted alphabetically,
        formatted for display.
        """
        lines: list[str] = []

        for command_id in sorted(self.__commands):
            description = self.__commands[command_id].description
            lines.append(f"  /{command_id:<10} {description}")

        return lines
