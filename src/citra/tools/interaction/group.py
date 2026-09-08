from ..tool_group import ToolGroup
from .prompt_user import PromptUser


class InteractionToolGroup(ToolGroup):
    GROUP_NAME = "Asked"
    GROUP_TOOLS = (PromptUser,)
