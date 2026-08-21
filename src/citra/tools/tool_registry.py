# src/citra/tools/registry.py

from citra.agent import AgentSession

from ..context import ExecutionContext
from .tool import Tool
from .session_tool import SessionTool
from .session_memory import MemoryTool


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
        Instantiated once per AgentSession and reused across model
        calls so its in-memory extracts survive for the duration
        of the agent run.
    """

    def __init__(self):
        self.__tools: dict[str, type[Tool]] = {}

        self.__memory_tools: dict[
            int,
            dict[str, MemoryTool],
        ] = {}

    def register(
        self,
        tool_id: str,
        tool_type: type[Tool],
    ) -> None:
        if tool_id in self.__tools:
            raise ValueError(
                f"Tool '{tool_id}' is already registered."
            )

        self.__tools[tool_id] = tool_type

    def instantiate(
        self,
        context: ExecutionContext,
        session: AgentSession,
    ) -> dict[str, Tool]:
        """
        Instantiate all registered tools for a model call.

        MemoryTool instances are reused for the lifetime of the
        AgentSession. Other tools are created fresh.
        """
        result: dict[str, Tool] = {}

        for tool_id, tool_type in self.__tools.items():
            if issubclass(tool_type, MemoryTool):
                tool = self._get_memory_tool(
                    tool_id=tool_id,
                    tool_type=tool_type,
                    context=context,
                    session=session,
                )

            elif issubclass(tool_type, SessionTool):
                tool = tool_type(
                    context=context,
                    session=session,
                )

            else:
                tool = tool_type(
                    context=context,
                )

            result[tool_id] = tool

        return result

    def _get_memory_tool(
        self,
        *,
        tool_id: str,
        tool_type: type[MemoryTool],
        context: ExecutionContext,
        session: AgentSession,
    ) -> MemoryTool:
        session_key = id(session)

        session_tools = self.__memory_tools.setdefault(
            session_key,
            {},
        )

        existing = session_tools.get(tool_id)

        if existing is not None:
            existing.rebind_context(
                context
            )
            return existing

        tool = tool_type(
            context=context,
            session=session,
        )

        session_tools[tool_id] = tool

        return tool

    def release_session(
        self,
        session: AgentSession,
    ) -> None:
        """
        Release MemoryTool instances associated with a completed
        AgentSession.
        """
        self.__memory_tools.pop(
            id(session),
            None,
        )

    def contains(
        self,
        tool_id: str,
    ) -> bool:
        return tool_id in self.__tools

    @property
    def tool_ids(self) -> tuple[str, ...]:
        return tuple(self.__tools)
