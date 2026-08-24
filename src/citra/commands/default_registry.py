# src/citra/commands/default_registry.py

"""
Build the default :class:`CommandRegistry` singleton.

Add new commands here by importing their class and registering it,
exactly like ``citra.tools.default_registry``.
"""

from .model import ModelCommand
from .lsp import LspCommand
from .clear import ClearCommand
from .command import CommandRegistry
from .help import HelpCommand
from .quit import QuitCommand
from .test import TestCommand


COMMAND_REGISTRY = CommandRegistry()

COMMAND_REGISTRY.register("q", QuitCommand)
COMMAND_REGISTRY.register("c", ClearCommand)
COMMAND_REGISTRY.register("help", HelpCommand)
COMMAND_REGISTRY.register("test", TestCommand)
COMMAND_REGISTRY.register("model", ModelCommand)
COMMAND_REGISTRY.register("lsp", LspCommand)