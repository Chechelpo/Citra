from ..tool_group import ToolGroup
from .browser import Browser
from .web_search import WebSearch


class WebToolGroup(ToolGroup):
    GROUP_NAME = "Explored the web"
    GROUP_TOOLS = (Browser, WebSearch)
