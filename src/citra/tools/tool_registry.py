# src/citra/tools/registry.py

import tomllib
from citra.tools.default_registry import ToolSet
from dataclasses import dataclass
from typing import Protocol, cast

from citra.agent import AgentSession

from ..context import ExecutionContext
from .tool import Tool
from .session_tool import SessionTool
from .session_memory import MemoryTool


@dataclass(frozen=True)
class ToolRegistration:
    tool_type: type[Tool]
    deferred: bool = False
    summary: str = ""


class _ToolFactory(Protocol):
    def __call__(self, *, context: ExecutionContext) -> Tool: ...


class _SessionToolFactory(Protocol):
    def __call__(
        self,
        *,
        context: ExecutionContext,
        session: AgentSession,
    ) -> SessionTool: ...


class _MemoryToolFactory(Protocol):
    def __call__(
        self,
        *,
        context: ExecutionContext,
        session: AgentSession,
    ) -> MemoryTool: ...


class ToolRegistry:
    """
    Permanent registry of tool implementations.

    Tool lifetimes:

    - Tool:
        Instantiated fresh for each model call.

    - SessionTool:
        Instantiated fresh for each model call and receives the
        current AgentSession.

    - MemoryTool:
        Owned by AgentSession and reused across model calls and turns, even
        when older conversation messages are omitted from model context.

    Deferred tools remain registered but are omitted from model calls until
    explicitly enabled for the current agent turn.
    """

    def __init__(self, toolset: ToolSet):
        self.__tools = toolset

    @property
    def tools(self) -> ToolSet:
        return self.__tools
    
    def instantiate(
        self,
        context: ExecutionContext,
        session: AgentSession,
        *,
        tool_ids: set[str] | None = None,
    ) -> dict[str, Tool]:
        """
        Instantiate selected registered tools for a model call.

        If tool_ids is omitted, all registered tools are instantiated.
        MemoryTool instances are reused for the lifetime of the AgentSession.
        Other tools are created fresh.
        """
        result: dict[str, Tool] = {}

        for tool_type in self.__tools.allowed_tools():
            if tool_ids is not None and tool_type not in tool_ids:
                continue

            if issubclass(tool_type, MemoryTool):
                configured_memory = getattr(
                    getattr(context, "config", None),
                    "memory",
                    None,
                )
                if not session.memory_enabled or not bool(
                    getattr(configured_memory, "enabled", True)
                ):
                    continue
                tool = self._get_memory_tool(
                    tool_id=tool_type.TOOL_ID,
                    tool_type=tool_type,
                    context=context,
                    session=session,
                )

            elif issubclass(tool_type, SessionTool):
                tool = cast(_SessionToolFactory, tool_type)(
                    context=context,
                    session=session,
                )

            else:
                tool = cast(_ToolFactory, tool_type)(
                    context=context,
                )

            result[tool.TOOL_ID] = tool

        return result

    def _get_memory_tool(
        self,
        *,
        tool_id: str,
        tool_type: type[MemoryTool],
        context: ExecutionContext,
        session: AgentSession,
    ) -> MemoryTool:
        tool = session.memory.get_or_create(
            tool_id,
            lambda: cast(_MemoryToolFactory, tool_type)(
                context=context,
                session=session,
            ),
        )
        tool.rebind_context(
            context
        )
        return tool

    def release_session(
        self,
        session: AgentSession,
    ) -> None:
        """
        Compatibility no-op; AgentSession owns and clears its own memory.
        """
        # Memory lifetime is explicitly controlled by AgentSession now.
        # Retain this method for callers written against the old registry API.
        del session

    
    def contains(
        self,
        tool_id: str,
    ) -> bool:
        return self.tools.get_tool_with_id(tool_id) is not None

    @property
    def deferred_catalog(self) -> dict[str, str]:
        return {
            tool_id: registration.summary
            for tool_id, registration in self.__tools.items()
            if registration.deferred
        }
