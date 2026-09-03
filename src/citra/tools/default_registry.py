
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

from .session_memory import *
from .subagent import SubagentTool
from .tool import Tool
from .transient import (
    Bash,
    Browser,
    Diagram,
    Document,
    Edit,
    Git,
    Glob,
    Lsp,
    PromptUser,
    Read,
    ReadImage,
    SkillTool,
    Subprocess,
    Tree,
    WebSearch,
    Workspace,
    Write,
)

__all__ = ["ToolSet", "all_tools", "memory_tools"]

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class ToolSet:
    """Tool implementations made available by a workflow.

    Entries are identified by their stable ``TOOL_ID``. The public function
    name exposed to a model is resolved later, after an execution context is
    available, and must never be used to select a registered implementation.
    """

    core_tools: tuple[type[Tool], ...]
    deferred_tools: tuple[type[Tool], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.core_tools, tuple):
            raise TypeError("core_tools must be a tuple")
        if not isinstance(self.deferred_tools, tuple):
            raise TypeError("deferred_tools must be a tuple")
        registered = self.core_tools + self.deferred_tools

        for tool_type in registered:
            if not isinstance(tool_type, type) or not issubclass(
                tool_type,
                Tool,
            ):
                raise TypeError("ToolSet entries must be Tool subclasses")

        duplicate_types = _duplicates(registered)
        if duplicate_types:
            names = ", ".join(
                tool_type.__name__ for tool_type in duplicate_types
            )
            raise ValueError(f"Duplicate tool implementations: {names}")

        ids = tuple(tool_type.TOOL_ID for tool_type in registered)
        duplicate_ids = _duplicates(ids)
        if duplicate_ids:
            raise ValueError(
                "Duplicate internal tool IDs: " + ", ".join(duplicate_ids)
            )

    @property
    def core_tool_ids(self) -> frozenset[str]:
        return frozenset(tool_type.TOOL_ID for tool_type in self.core_tools)

    @property
    def deferred_tool_ids(self) -> frozenset[str]:
        return frozenset(tool_type.TOOL_ID for tool_type in self.deferred_tools)

    def get_tool_with_id(self, tool_id: str) -> type[Tool] | None:
        for tool_type in self.allowed_tools():
            if tool_type.TOOL_ID == tool_id:
                return tool_type
        return None

    def allowed_tools(self) -> tuple[type[Tool], ...]:
        return self.core_tools + self.deferred_tools

    def is_core_tool(self, tool_type: type[Tool]) -> bool:
        return tool_type in self.core_tools

    def is_deferred_tool(self, tool_type: type[Tool]) -> bool:
        return tool_type in self.deferred_tools

    # Compatibility for callers using the historical misspelling.
    def is_deffered_tool(self, tool_type: type[Tool]) -> bool:
        return self.is_deferred_tool(tool_type)

    def is_allowed_tool(self, tool_type: type[Tool]) -> bool:
        return self.is_core_tool(tool_type) or self.is_deferred_tool(tool_type)


def _duplicates(values: tuple[T, ...]) -> tuple[T, ...]:
    seen: set[T] = set()
    duplicates: list[T] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def memory_tools(
    *,
    exclude: tuple[type[Tool], ...] = (),
) -> tuple[type[Tool], ...]:
    """Return concrete conversation-memory tools excluding selected tools."""

    tools = (
        TodoTool,
        DecisionTool,
        ConstraintTool,
        FactTool,
        CheckpointTool,
        RequirementTool,
        ScopeTool,
        AcceptanceCriteriaTool,
        WorkingStateTool,
    )

    excluded = set(exclude)
    return tuple(tool for tool in tools if tool not in excluded)


_CORE_TOOL_TYPES: tuple[type[Tool], ...] = (
    Read,
    Write,
    Edit,
    Glob,
    Tree,
    Bash,
    Workspace,
    PromptUser,
    Lsp,
    SkillTool,
    *memory_tools(),
)

_DEFERRED_TOOL_TYPES: tuple[type[Tool], ...] = (
    Git,
    Subprocess,
    Browser,
    WebSearch,
    Document,
    Diagram,
    ReadImage,
    SubagentTool,
)


def all_tools(
    are_deferred: bool | None = None,
    excluded: set[type[Tool]] | frozenset[type[Tool]] = frozenset(),
) -> tuple[type[Tool], ...]:
    """Return the default core, deferred, or complete tool collection."""

    if are_deferred is True:
        registry = _DEFERRED_TOOL_TYPES
    elif are_deferred is False:
        registry = _CORE_TOOL_TYPES
    else:
        registry = _CORE_TOOL_TYPES + _DEFERRED_TOOL_TYPES

    return tuple(tool_type for tool_type in registry if tool_type not in excluded)
