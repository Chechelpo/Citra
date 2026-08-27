from typing import Any, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..tool import Tool, ToolDefinition


class Materialize(Tool):
    """Hidden migration compatibility tool for older embedders."""

    TOOL_ID = "materialize"

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="materialize",
            description=(
                "Compatibility endpoint for clients predating the Agent "
                "Runtime. The source workspace is already completely copied "
                "into @workspace at startup, so this endpoint performs no copy."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description=(
                                "Legacy compatibility value. Both 'copy' and "
                                "'preview' are no-ops. Defaults to copy."
                            ),
                            enum=(
                                "copy",
                                "preview",
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="paths",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Ignored legacy source-relative paths."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="include_ignored",
                        schema=JsonSchema.boolean(
                            description=(
                                "Ignored legacy compatibility flag."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="allow_large",
                        schema=JsonSchema.boolean(
                            description=(
                                "Ignored legacy compatibility flag."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        del context

        return (
            ToolDefinition(
                definition=cls.DEFINITION,
            ),
        )

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        super().__init__(
            context=context,
        )

    def is_cacheable(
        self,
        arguments: dict[str, Any],
    ) -> bool:
        return (
            arguments.get(
                "action",
                "copy",
            )
            == "preview"
        )

    def invalidates_tool_cache(
        self,
        arguments: dict[str, Any],
    ) -> bool:
        return (
            arguments.get(
                "action",
                "copy",
            )
            != "preview"
        )

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        del arguments

        return (
            "No action required: @workspace is the complete disposable "
            "source snapshot created at Agent Runtime startup."
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        action = arguments.get(
            "action",
            "copy",
        )

        paths = arguments.get(
            "paths",
            [],
        )

        preview = ", ".join(
            str(path)
            for path in paths[:3]
        )

        if len(paths) > 3:
            preview += (
                f", +{len(paths) - 3} more"
            )

        parts = [
            f"action={action}",
            f"paths=[{preview}]",
        ]

        if arguments.get(
            "include_ignored"
        ):
            parts.append(
                "include_ignored=true"
            )

        if arguments.get(
            "allow_large"
        ):
            parts.append(
                "allow_large=true"
            )

        return " | ".join(
            parts
        )

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        text = str(
            result
        )

        lines = text.splitlines()

        return (
            f"{len(lines)} lines"
        )