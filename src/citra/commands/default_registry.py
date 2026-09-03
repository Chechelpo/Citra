# src/citra/commands/default_registry.py

"""
Build the default :class:`CommandRegistry` singleton.

Add new commands here by importing their class and registering it,
exactly like ``citra.tools.default_registry``.
"""

from .model import ModelCommand
from .lsp import LspCommand
from .agent import AgentCommand
from .apply import ApplyCommand
from .clear import ClearCommand
from .command import CommandRegistry
from .debug import DebugCommand
from .help import HelpCommand
from .quit import QuitCommand
from .test import TestCommand
from .workflow import WorkflowCommand
from .workspace import WorkspaceCommand


COMMAND_REGISTRY = CommandRegistry()

COMMAND_REGISTRY.register("q", QuitCommand)
COMMAND_REGISTRY.register("c", ClearCommand)
COMMAND_REGISTRY.register("help", HelpCommand)
COMMAND_REGISTRY.register("test", TestCommand)
COMMAND_REGISTRY.register("model", ModelCommand)
COMMAND_REGISTRY.register("lsp", LspCommand)
COMMAND_REGISTRY.register("debug", DebugCommand)
COMMAND_REGISTRY.register("agent", AgentCommand)
COMMAND_REGISTRY.register("workflow", WorkflowCommand)
COMMAND_REGISTRY.register("workspace", WorkspaceCommand)
COMMAND_REGISTRY.register("apply", ApplyCommand)
