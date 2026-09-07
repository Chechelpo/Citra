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
from ..logging import Logger


_registry_logger = Logger(__name__)


@dataclass(frozen=True)
class CommandArgument:
    """Describe one positional argument or option in a command form."""

    syntax: str
    description: str
    suggestions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.syntax.strip():
            raise ValueError("Command argument syntax cannot be empty.")
        if not self.description.strip():
            raise ValueError("Command argument description cannot be empty.")
        if any(not suggestion.strip() for suggestion in self.suggestions):
            raise ValueError("Command argument suggestions cannot be empty.")


@dataclass(frozen=True)
class CommandOption:
    """Describe a named option and its optional value placeholder."""

    flag: str
    description: str
    value: str | None = None
    value_suggestions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.flag.startswith("--") or any(
            character.isspace() for character in self.flag
        ):
            raise ValueError("Command option flags must be one --long token.")
        if not self.description.strip():
            raise ValueError("Command option description cannot be empty.")

    @property
    def syntax(self) -> str:
        """Return optional command-line syntax for this option."""
        value = f" {self.value}" if self.value else ""
        return f"[{self.flag}{value}]"


@dataclass(frozen=True)
class CommandForm:
    """Describe one supported invocation beneath a slash command."""

    path: tuple[str, ...] = ()
    arguments: tuple[CommandArgument, ...] = ()
    options: tuple[CommandOption, ...] = ()
    description: str = ""

    @property
    def suffix(self) -> str:
        """Return the form syntax following the slash-command name."""
        return " ".join(
            (
                *self.path,
                *(item.syntax for item in self.options),
                *(item.syntax for item in self.arguments),
            )
        )


@dataclass(frozen=True)
class CommandUsage:
    """Declare the complete user-facing usage tree for one command."""

    command: str
    description: str
    forms: tuple[CommandForm, ...] = ()

    def __post_init__(self) -> None:
        if not self.command.strip() or any(character.isspace() for character in self.command):
            raise ValueError("Command usage name must be one non-empty token.")
        if not self.description.strip():
            raise ValueError("Command usage description cannot be empty.")


@dataclass(frozen=True)
class CommandResult:
    """
    Outcome of running a command.

    Attributes
    ----------
    output:
        Text to print to the terminal. May be empty.
    usage:
        Command usage trees to render after the output.
    clear_messages:
        If ``True``, the conversation history is cleared after the
        command runs. Used by ``/clear`` and similar maintenance
        commands.
    exit:
        If ``True``, the REPL terminates after the command runs.
        Used by ``/quit``.
    """

    output: str = ""
    usage: tuple[CommandUsage, ...] = ()
    clear_messages: bool = False
    exit: bool = False


class Command(ABC):
    """
    Abstract base for all REPL commands.

    Subclasses must set :attr:`id` and declare a matching
    :attr:`usage` tree, then implement :meth:`_run`.
    """

    id: str = ""
    usage = CommandUsage(command="command", description="Undocumented command.")

    def __init__(self, context: ExecutionContext) -> None:
        """Bind one execution context and a source-labelled diagnostic logger."""
        self._context = context
        self._logger = Logger(type(self).__module__)
        self._logger.trace("Created command instance", command=self.id)

    @property
    def context(self) -> ExecutionContext:
        """Return the execution context bound to this invocation."""
        return self._context

    def usage_result(self, output: str = "") -> CommandResult:
        """Return an optional message followed by this command's usage tree."""
        return CommandResult(output=output, usage=(self.usage,))

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
        self._logger.debug("Executing command", command=self.id)
        try:
            result = self._run(args)
            self._logger.info(
                "Command completed",
                command=self.id,
                exit=result.exit,
                clears_messages=result.clear_messages,
            )
            return result
        except Exception as error:  # noqa: BLE001
            from ..utils.terminal import RED, RESET

            self._logger.error(
                "Command failed",
                command=self.id,
                error_type=type(error).__name__,
                error=str(error),
            )
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
        """Create an empty registry of command implementation classes."""
        self.__commands: dict[str, type[Command]] = {}
        _registry_logger.trace("Created command registry")

    def register(
        self,
        command_id: str,
        command_type: type[Command],
    ) -> None:
        """Register one command class under a unique non-empty id."""
        if command_id in self.__commands:
            _registry_logger.error(
                "Rejected duplicate command registration",
                command=command_id,
            )
            raise ValueError(
                f"Command '{command_id}' is already registered."
            )

        if not command_id:
            _registry_logger.error("Rejected empty command registration")
            raise ValueError("Command id cannot be empty.")

        if command_type.usage.command != command_id:
            raise ValueError(
                f"Command usage name {command_type.usage.command!r} does not match "
                f"registered id {command_id!r}."
            )

        self.__commands[command_id] = command_type
        _registry_logger.debug(
            "Registered command",
            command=command_id,
            implementation=command_type.__module__,
        )

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
            _registry_logger.warning(
                "Requested command is not registered",
                command=command_id,
            )
            return None

        _registry_logger.trace("Instantiating command", command=command_id)
        return command_type(context)

    def contains(self, command_id: str) -> bool:
        """Return whether a command id is registered."""
        present = command_id in self.__commands
        _registry_logger.trace(
            "Checked command registration",
            command=command_id,
            present=present,
        )
        return present

    @property
    def command_ids(self) -> tuple[str, ...]:
        """Return command ids in registration order."""
        return tuple(self.__commands)

    @property
    def usages(self) -> tuple[CommandUsage, ...]:
        """Return declared command usage trees sorted by command name."""
        return tuple(
            self.__commands[command_id].usage
            for command_id in sorted(self.__commands)
        )
