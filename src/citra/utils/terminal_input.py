"""Compatibility exports for the CLI-owned input composer."""

from ..cli import input as _input

TerminalInput = _input.TerminalInput
terminal_input = _input.terminal_input
_IdleTimeout = _input._IdleTimeout
_IdleWatchdog = _input._IdleWatchdog
_PredicateSatisfied = _input._PredicateSatisfied

__all__ = ["TerminalInput", "terminal_input"]
