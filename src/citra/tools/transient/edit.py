from typing import Any, override

from ...context import ExecutionContext
from ..tool import Tool
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ._post_edit import post_edit_result

_TRUNCATE_LENGTH = 120


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
                "Set all=true only when every occurrence should be replaced. "
                "After a successful edit, Citra automatically runs available "
                "LSP diagnostics and configured project lint checks."
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
                    JsonProperty(
                        name="diagnostics",
                        schema=JsonSchema.boolean(
                            description=(
                                "Deprecated compatibility flag. LSP diagnostics "
                                "and configured lint checks run automatically "
                                "after every successful edit."
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
        filesystem_arguments = {
            key: value
            for key, value in arguments.items()
            if key != "diagnostics"
        }
        result = self.context.filesystem.execute(
            "edit",
            filesystem_arguments,
        )

        if result != "ok":
            return result

        return post_edit_result(
            self.context,
            arguments["path"],
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        path = arguments.get("path", "")
        parts = [f"path={path}"]

        old = arguments.get("old")
        new = arguments.get("new")
        line = arguments.get("line")
        replace_all = arguments.get("all", False)

        if line is not None:
            parts.append(f"insert@line={line}")
        elif old is not None:
            parts.append(
                f"old={self._truncate(old)!r}"
            )
            if new is not None:
                parts.append(
                    f"new={self._truncate(new)!r}"
                )
            if replace_all:
                parts.append("all=true")

        return " | ".join(parts)

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        text = str(result)

        if text == "ok":
            return "ok"

        if text.startswith("error:"):
            return self._truncate(text)

        return self._truncate(text)

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= _TRUNCATE_LENGTH:
            return value
        return value[:_TRUNCATE_LENGTH] + "..."
