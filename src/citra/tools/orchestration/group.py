from ..tool_group import ToolGroup
from .enable_tools import EnableTools
from .workflow_handoff import WorkflowHandoffTool


class OrchestrationToolGroup(ToolGroup):
    GROUP_NAME = "Coordinated"
    GROUP_TOOLS = (EnableTools, WorkflowHandoffTool)
