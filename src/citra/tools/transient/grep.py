from typing import Any, override

from ...context import ExecutionContext
from ..tool import Tool
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)


class Grep(Tool):
    """
    Searches text files using a regular expression.

    If path refers to a file, only that file is searched.
    If path refers to a directory, it is searched recursively.
    Binary/unreadable files are silently skipped.
    """

    MAX_RESULTS = 50

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="grep",
            description=(
                "Search files for lines matching a Python regular expression. "
                "If path is a file, searches only that file. If path is a "
                "directory, searches it recursively. Searches within the active "
                "workspace or temporary agent filesystem. Returns matching file "
                "paths, line numbers, and line contents. At most 50 matches are "
                "returned."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="pat",
                        schema=JsonSchema.string(
                            description=(
                                "Python regular expression used to match "
                                "individual lines."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "File or directory to search. If a file is "
                                "provided, only that file is searched. If a "
                                "directory is provided, it is searched "
                                "recursively. Relative paths resolve from the "
                                "active workspace. Defaults to the current "
                                "workspace."
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
            "grep",
            arguments,
        )