from collections.abc import Mapping
from typing import Any

from ..tool import Tool
from ..tool_group import ToolGroup
from .find import Find
from .glob import Glob
from .grep import Grep
from .read import Read
from .read_image import ReadImage
from .tree import Tree


class ExplorerToolGroup(ToolGroup):
    GROUP_NAME = "Explored"
    GROUP_TOOLS = (Find, Glob, Grep, Read, ReadImage, Tree)

    @classmethod
    def format_result(cls, tool: Tool, result: str, arguments: Mapping[str, Any] | None = None) -> str:
        del arguments
        if not result or result in {"none", "no matches", "(empty)"}:
            return "No results"
        lines = len(result.splitlines())
        if tool.id == "read":
            return f"Read {lines} line(s) · {len(result)} chars"
        if tool.id in {"find", "glob", "grep"}:
            return f"Found {lines} match(es)"
        return result
