from ..tool_group import ToolGroup
from .git import Git
from .lsp import Lsp


class DeveloperToolGroup(ToolGroup):
    GROUP_NAME = "Inspected code"
    GROUP_TOOLS = (Git, Lsp)
