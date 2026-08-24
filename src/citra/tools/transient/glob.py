from typing import Any, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..tool import Tool


class Glob(Tool):
    CACHEABLE = True
    INVALIDATES_TOOL_CACHE = False

    """
    Finds filesystem entries matching a glob pattern.

    Results are sorted by modification time, newest first.
    """

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="glob",
            description=(
                "Find files and directories using a glob pattern. "
                "Supports recursive patterns such as '**/*.py'. "
                "Results are sorted by modification "
                "time with the most recently modified files first."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="pat",
                        schema=JsonSchema.string(
                            description=(
                                "Glob pattern, for example '*.py', "
                                "'src/**/*.py', or '**/config.toml'."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "Directory from which to apply the pattern. "
                                "Relative paths resolve from the active "
                                "workspace. Paths may also refer to the "
                                "temporary agent filesystem."
                            ),
                        ),
                        required=False,
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
            "glob",
            arguments,
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        pat = arguments.get("pat", "")
        path = arguments.get("path")

        if path is not None:
            return f"pat={pat} | path={path}"

        return f"pat={pat}"

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        text = str(result)

        if not text or text == "none":
            return "no matches"

        lines = text.splitlines()
        return f"{len(lines)} match(es)"
