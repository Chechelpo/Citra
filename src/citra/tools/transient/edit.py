from typing import Any, override

from ...context import ExecutionContext
from ..tool import Tool
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)


class Edit(Tool):
    """
    Performs exact text replacement in an existing file.

    Unless all=true, the text being replaced must occur exactly once.
    """

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="edit",
            description=(
                "Replace an exact text fragment in an existing file. "
                "Files may only be modified inside the active workspace "
                "or temporary agent filesystem. By default the old text "
                "must occur exactly once, preventing ambiguous edits. "
                "Set all=true only when every occurrence should be replaced."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "Path of the file to modify. Relative paths "
                                "resolve from the active workspace."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="line",
                        schema=JsonSchema.integer(
                            description=(
                                "1-based line number before which 'new' should be inserted "
                                "without deleting existing text. Use len(file)+1 to append. "
                                "Cannot be used together with 'old'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="old",
                        schema=JsonSchema.string(
                            description=(
                                "Exact existing text to replace. Whitespace and line breaks "
                                "must match. Omit when using 'line' to insert text."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="new",
                        schema=JsonSchema.string(
                            description=(
                                "Replacement text. May be empty to delete "
                                "the matched text."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="all",
                        schema=JsonSchema.boolean(
                            description=(
                                "Replace every occurrence instead of requiring "
                                "the old text to be unique. Defaults to false."
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
            "edit",
            arguments,
        )
