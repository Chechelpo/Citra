import glob as globlib
import os
from pathlib import Path
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
    """
    Finds filesystem entries matching a glob pattern.

    Results are restricted to the active workspace and temporary
    agent filesystem.

    Results are sorted by modification time, newest first.
    """

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="glob",
            description=(
                "Find files and directories using a glob pattern. "
                "Supports recursive patterns such as '**/*.py'. "
                "Searches within the active workspace or temporary "
                "agent filesystem. Results are sorted by modification "
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
        base_path = self.context.workspace.resolve_path(
            arguments.get(
                "path",
                ".",
            )
        )

        pattern = str(
            base_path / arguments["pat"]
        )

        entries = globlib.glob(
            pattern,
            recursive=True,
        )

        entries = sorted(
            entries,
            key=self._modified_time,
            reverse=True,
        )

        if not entries:
            return "none"

        return "\n".join(
            self.context.workspace.display_path(
                Path(entry)
            )
            for entry in entries
        )

    @staticmethod
    def _modified_time(
        path: str,
    ) -> float:
        try:
            return os.path.getmtime(
                path
            )
        except OSError:
            return 0