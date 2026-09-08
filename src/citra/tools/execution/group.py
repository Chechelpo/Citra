from collections.abc import Mapping
from typing import Any

from ..tool import Tool
from ..tool_group import ToolGroup
from .bash import Bash
from .python import Python
from .subprocess import Subprocess


class ExecutionToolGroup(ToolGroup):
    GROUP_NAME = "Ran commands"
    GROUP_TOOLS = (Bash, Python, Subprocess)

    @classmethod
    def format_result(cls, tool: Tool, result: str, arguments: Mapping[str, Any] | None = None) -> str:
        del tool, arguments
        if not result or result == "(empty)":
            return "Completed · no output"
        if result.startswith(("error:", "permission-denied:", "cancelled:")):
            return result
        lines = result.splitlines()
        return f"Completed · {len(lines)} output line(s)"
