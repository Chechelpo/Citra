from ..tool_group import ToolGroup
from .tool import SkillTool


class SkillToolGroup(ToolGroup):
    GROUP_NAME = "Used skills"
    GROUP_TOOLS = (SkillTool,)
