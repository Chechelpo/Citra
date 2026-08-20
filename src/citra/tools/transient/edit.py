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
                        name="old",
                        schema=JsonSchema.string(
                            description=(
                                "Exact existing text that should be replaced. "
                                "Whitespace and line breaks must match."
                            ),
                        ),
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
        path = self.context.workspace.require_writable_path(
            arguments["path"]
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        old: str = arguments["old"]
        new: str = arguments["new"]
        replace_all: bool = arguments.get(
            "all",
            False,
        )

        if not old:
            return "error: old string cannot be empty"

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:
            text = file.read()

        if old not in text:
            return "error: old_string not found"

        count = text.count(
            old
        )

        if (
            not replace_all
            and count > 1
        ):
            return (
                f"error: old_string appears {count} times, "
                "must be unique (use all=true)"
            )

        replacement = text.replace(
            old,
            new,
            -1 if replace_all else 1,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                replacement
            )

        return "ok"