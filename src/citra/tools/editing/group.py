from collections.abc import Mapping
from typing import Any

from ..tool import Tool
from ..tool_group import ToolGroup
from .edit import Edit
from .write import Write


class EditingToolGroup(ToolGroup):
    GROUP_NAME = "Changed"
    GROUP_TOOLS = (Edit, Write)

    @classmethod
    def format_result(cls, tool: Tool, result: str, arguments: Mapping[str, Any] | None = None) -> str:
        if not result.startswith("ok"):
            return result
        arguments = arguments or {}
        path = next(
            (str(arguments[name]) for name in ("path", "file_path") if arguments.get(name)),
            "",
        )
        if tool.id == "write":
            summary = f"Wrote {path}" if path else "Wrote"
        else:
            old = str(arguments.get("old", ""))
            new = str(arguments.get("new", ""))
            summary = f"ok (+{len(new.splitlines())}, -{len(old.splitlines())})"
        diagnostics = result.removeprefix("ok").strip()
        return summary if not diagnostics else f"{summary}\n{diagnostics}"
