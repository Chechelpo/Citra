from typing import Any, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..tool import Tool


class Materialize(Tool):
    """Copy selected source files into the isolated agent workspace."""

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="materialize",
            description=(
                "Preview or copy selected files from read-only @source into "
                "the turn-scoped agent workspace. Git is not required: "
                "tracked, untracked, and non-repository files are eligible. "
                "Calls are additive and never overwrite files already copied "
                "or edited by the agent. Directory and glob expansion obeys "
                "ignore files and built-in cache/build exclusions by default; "
                "an exact file path overrides soft ignores. VCS internals and "
                "special files are always excluded. Use preview before a "
                "large expansion and ['.'] when the complete project is "
                "needed for testing."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description=(
                                "'copy' materializes files; 'preview' only "
                                "reports the eligible scope. Defaults to copy."
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
                                "Source-relative files, directories, or glob "
                                "patterns, such as 'src/app.py', 'tests', "
                                "'**/*.py', or '.'."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="include_ignored",
                        schema=JsonSchema.boolean(
                            description=(
                                "Include soft-ignored files during directory "
                                "and glob expansion. Defaults to false. Exact "
                                "file paths already override soft ignores."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="allow_large",
                        schema=JsonSchema.boolean(
                            description=(
                                "Permit a copy above the normal file-count or "
                                "byte limit. Defaults to false."
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
        action = arguments.get(
            "action",
            "copy",
        )
        return self.context.workspace.changes.materialize(
            arguments["paths"],
            preview=action == "preview",
            include_ignored=arguments.get(
                "include_ignored",
                False,
            ),
            allow_large=arguments.get(
                "allow_large",
                False,
            ),
        ).format()
