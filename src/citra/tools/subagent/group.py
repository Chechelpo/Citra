from ..tool_group import ToolGroup
from .guidance import RequestGuidanceTool
from .tool import SubagentTool


class SubagentToolGroup(ToolGroup):
    GROUP_NAME = "Delegated"
    GROUP_TOOLS = (SubagentTool, RequestGuidanceTool)
