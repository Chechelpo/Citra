from ..tool_group import ToolGroup
from .workspace import Workspace


class WorkspaceToolGroup(ToolGroup):
    GROUP_NAME = "Managed workspace"
    GROUP_TOOLS = (Workspace,)
