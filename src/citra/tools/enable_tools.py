"""Runner-owned tool for enabling deferred model-facing tools."""

from __future__ import annotations

from typing import Any, override

from ..context import ExecutionContext
from ..utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from .tool import Tool, ToolDefinition


class EnableTools(Tool):
    """Enable deferred tools for the remainder of the current agent turn."""

    TOOL_ID = "enable_tools"

    INVALIDATES_TOOL_CACHE = False
    MAX_OUTPUT_TOKENS = 500

    def __init__(
        self,
        context: ExecutionContext,
        *,
        available_tools: dict[str, str],
        enabled_tool_ids: set[str],
    ) -> None:
        self.__available_tools = dict(
            available_tools
        )

        self.__enabled_tool_ids = (
            enabled_tool_ids
        )

        super().__init__(
            context=context,
        )

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        # EnableTools requires runtime instance state, so its actual
        # definition is supplied by definitions_for_instance().
        del context

        return ()

    @override
    def definitions_for_instance(
        self,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        del context

        catalog = "\n".join(
            f"- {tool_id}: {summary}"
            for tool_id, summary
            in self.__available_tools.items()
        )

        definition = ChatCompletionTool(
            function=FunctionDefinition(
                name="enable_tools",
                description=(
                    "Enable one or more specialized tools for the remainder "
                    "of the current agent turn. Enable a tool only when its "
                    "capability is actually needed. Available deferred tools:\n"
                    f"{catalog}"
                ),
                parameters=JsonSchema.object(
                    properties=(
                        JsonProperty(
                            name="tools",
                            schema=JsonSchema.array(
                                JsonSchema.string(
                                    enum=tuple(
                                        self.__available_tools
                                    ),
                                ),
                                description=(
                                    "Deferred tool IDs to enable. Multiple "
                                    "tools may be enabled in one call."
                                ),
                            ),
                        ),
                    ),
                    additional_properties=False,
                ),
            ),
        )

        return (
            ToolDefinition(
                definition=definition,
            ),
        )

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        requested = tuple(
            dict.fromkeys(
                arguments["tools"]
            )
        )

        # Schema validation already constrains these values, but retain
        # runtime validation because this tool's catalog is lifecycle state.
        unknown = [
            tool_id
            for tool_id in requested
            if tool_id not in self.__available_tools
        ]

        if unknown:
            raise ValueError(
                "Unknown deferred tool ID(s): "
                + ", ".join(
                    unknown
                )
            )

        newly_enabled = [
            tool_id
            for tool_id in requested
            if tool_id not in self.__enabled_tool_ids
        ]

        self.__enabled_tool_ids.update(
            requested
        )

        if not newly_enabled:
            return (
                "ok: requested tools were already enabled"
            )

        return (
            "enabled: "
            + ", ".join(
                newly_enabled
            )
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        return (
            ", ".join(
                arguments.get(
                    "tools",
                    (),
                )
            )
            or "none"
        )