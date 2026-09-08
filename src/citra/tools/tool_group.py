"""Semantic tool groups used by the registry and CLI."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import Any, ClassVar, final

from .tool import Tool, ToolArguments


class ToolGroup(ABC):
    """Own a coherent tool package and its user-facing rendering policy."""

    GROUP_TOOLS: ClassVar[tuple[type[Tool], ...]] = ()
    GROUP_NAME: ClassVar[str] = "Used tools"
    _GROUPS: ClassVar[list[type[ToolGroup]]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.GROUP_TOOLS:
            cls._GROUPS.append(cls)

    @classmethod
    @final
    def owns(cls, tool_type: type[Tool]) -> bool:
        return tool_type in cls.GROUP_TOOLS

    @classmethod
    def for_tool(cls, tool: Tool | None) -> type[ToolGroup] | None:
        if tool is None:
            return None
        matches = [group for group in cls._GROUPS if group.owns(type(tool))]
        if len(matches) > 1:
            raise ValueError(f"Tool {tool.id!r} belongs to multiple tool groups.")
        return matches[0] if matches else None

    @classmethod
    def for_name(cls, tool_name: str) -> type[ToolGroup] | None:
        """Resolve a group for an unavailable call from registered stable IDs."""
        normalized = tool_name.casefold()
        for group in cls._GROUPS:
            for tool_type in group.GROUP_TOOLS:
                definition = getattr(
                    tool_type,
                    "DEFINITION",
                    getattr(tool_type, "CITRA_DEFINITION", None),
                )
                if tool_type.TOOL_ID.casefold() == normalized:
                    return group
                if definition is not None and definition.function.name.casefold() == normalized:
                    return group
        return None

    @classmethod
    def format_call(cls, tool: Tool, arguments: ToolArguments) -> str:
        """Delegate call presentation to the tool implementation."""
        return str(tool.format_call_log(arguments.to_dict()))

    @classmethod
    def format_result(
        cls,
        tool: Tool,
        result: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> str:
        """Return the group's result receipt text."""
        del tool, arguments
        return result.strip() or "(empty)"


class GenericToolGroup(ToolGroup):
    """Fallback for tools that are intentionally not package-grouped."""

    GROUP_NAME = "Used tools"
