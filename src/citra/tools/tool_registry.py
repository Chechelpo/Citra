from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, cast

from citra.agent import AgentSession

from ..context import ExecutionContext
from .default_registry import ToolSet
from .session_memory import MemoryTool
from .session_tool import SessionTool
from .tool import Tool


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
    """Instantiate a workflow's tools while preserving internal/public identity.

    Registry selection always uses stable Citra ``TOOL_ID`` values. Once the
    active context has selected each tool's model-facing definition,
    :meth:`index_by_model_name` produces the map used for API dispatch.
    """

    def __init__(self, toolset: ToolSet):
        self.__tools = toolset

    @property
    def tools(self) -> ToolSet:
        return self.__tools

    @property
    def core_tool_ids(self) -> frozenset[str]:
        return self.__tools.core_tool_ids

    @property
    def deferred_tool_ids(self) -> frozenset[str]:
        return self.__tools.deferred_tool_ids

    def deferred_catalog(self, context: ExecutionContext) -> dict[str, str]:
        """Return stable enablement IDs and context-selected descriptions."""

        catalog: dict[str, str] = {}
        for tool_type in self.__tools.deferred_tools:
            definition = tool_type.resolve_definition_for_context(context)
            catalog[tool_type.TOOL_ID] = definition.function.description
        return catalog

    def instantiate(
        self,
        context: ExecutionContext,
        session: AgentSession,
        *,
        tool_ids: set[str] | frozenset[str] | None = None,
    ) -> dict[str, Tool]:
        """Instantiate selected tools, keyed by stable internal ID."""

        result: dict[str, Tool] = {}

        for tool_type in self.__tools.allowed_tools():
            tool_id = tool_type.TOOL_ID
            if tool_ids is not None and tool_id not in tool_ids:
                continue

            if issubclass(tool_type, MemoryTool):
                workflow_requires_memory = (
                    context.workflow_runtime.workflow.requires_memory
                )
                memory_configured = context.config.memory.enabled
                if not session.memory_enabled or not (
                    memory_configured or workflow_requires_memory
                ):
                    continue
                tool = self._get_memory_tool(
                    tool_id=tool_id,
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
                tool = cast(_ToolFactory, tool_type)(context=context)

            result[tool_id] = tool

        return result

    @staticmethod
    def index_by_model_name(tools: Iterable[Tool]) -> dict[str, Tool]:
        """Index tools by the exact function name exposed to the model."""

        result: dict[str, Tool] = {}
        for tool in tools:
            previous = result.get(tool.model_name)
            if previous is not None:
                raise ValueError(
                    "Model-facing tool name collision "
                    f"{tool.model_name!r}: internal IDs "
                    f"{previous.id!r} and {tool.id!r}"
                )
            result[tool.model_name] = tool
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
        tool.rebind_context(context)
        tool.rebind_session(session)
        return tool

    def release_session(self, session: AgentSession) -> None:
        """Compatibility no-op; ``AgentSession`` owns durable memory."""

        del session

    def contains(self, tool_id: str) -> bool:
        return self.__tools.get_tool_with_id(tool_id) is not None
