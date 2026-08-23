from typing import Any, override

from ...context import ExecutionContext
from ..tool import Tool
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)


class Write(Tool):
    """
    Writes or replaces a complete text file.

    Relative paths are resolved against the active workspace.
    """

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="write",
            description=(
                "Write complete text content to a file. "
                "Creates the file if it does not exist and completely "
                "overwrites it if it does exist. Writes are restricted "
                "to the isolated agent workspace and lifecycle agent "
                "filesystem. @source is read-only. "
                "Use edit instead when only a specific existing fragment "
                "should be changed."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "Destination file path. Relative paths are "
                                "resolved against the active workspace."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description=(
                                "Complete content that the file should contain."
                            ),
                        ),
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        super().__init__(
            context=context,
            definition=self.DEFINITION,
        )

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        return self.context.filesystem.execute(
            "write",
            arguments,
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        path = arguments.get("path", "")
        content = arguments.get("content", "")
        return f"path={path} | {len(content)} chars"

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        return str(result)
