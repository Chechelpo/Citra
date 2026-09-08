from ..tool_group import ToolGroup
from .diagram import Diagram
from .document import Document


class DocumentToolGroup(ToolGroup):
    GROUP_NAME = "Updated documents"
    GROUP_TOOLS = (Diagram, Document)
