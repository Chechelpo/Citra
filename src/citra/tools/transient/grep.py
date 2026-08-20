import glob as globlib
import os
from pathlib import Path
import re
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
    Searches text files recursively using a regular expression.

    Binary/unreadable files are silently skipped.
    """

    MAX_RESULTS = 50

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="grep",
            description=(
                "Recursively search files for lines matching a Python regular "
                "expression. Searches within the active workspace or temporary "
                "agent filesystem. Returns matching file paths, line numbers, "
                "and line contents. At most 50 matches are returned."
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
                                "Directory to search recursively. Relative "
                                "paths resolve from the active workspace. "
                                "Defaults to the current workspace."
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
        try:
            pattern = re.compile(
                arguments["pat"]
            )
        except re.error as error:
            return (
                f"error: invalid regex: {error}"
            )

        base_path = self.context.workspace.resolve_path(
            arguments.get(
                "path",
                ".",
            )
        )

        hits: list[str] = []

        for filepath in globlib.glob(
            str(base_path / "**"),
            recursive=True,
        ):
            if not os.path.isfile(
                filepath
            ):
                continue

            try:
                with open(
                    filepath,
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as file:
                    for line_num, line in enumerate(
                        file,
                        1,
                    ):
                        if not pattern.search(
                            line
                        ):
                            continue

                        shown_path = self.context.workspace.display_path(
                            Path(filepath)
                        )

                        hits.append(
                            f"{shown_path}:"
                            f"{line_num}:"
                            f"{line.rstrip()}"
                        )

                        if len(hits) >= self.MAX_RESULTS:
                            return "\n".join(
                                hits
                            )

            except (
                OSError,
                UnicodeError,
            ):
                continue

        return (
            "\n".join(hits)
            or "none"
        )