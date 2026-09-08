"""Model-facing literal file reader backed by the sandbox worker."""

from typing import Any, override

from citra.sandbox.filesystem_ops import ReadInput

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..capabilities import ToolCapabilities
from ..tool import Tool

_TRUNCATE_LENGTH = 120


class Read(Tool):
    """Read one or more literal file paths."""

    TOOL_ID = "read"
    CAPABILITIES = ToolCapabilities()
    CACHEABLE = True
    INVALIDATES_TOOL_CACHE = False

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="read",
            description=(
                "Read the complete contents of one or more files by literal path. "
                "Use glob or find to discover files first; glob patterns are not "
                "accepted."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "One literal project-relative, absolute, or @tmp "
                                "file path. Mutually exclusive with paths."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="paths",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Literal file paths to read as a batch. Mutually "
                                "exclusive with path."
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
    def definition_for_context(
        cls,
        context: ExecutionContext,
    ) -> ChatCompletionTool:
        """Return the model-independent read definition."""
        del context
        return cls.DEFINITION

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        """Read the requested literal paths through the sandbox."""
        return self.context.filesystem.execute(
            ReadInput.parse(dict(arguments))
        ).to_budgeted(
            model_id=self.context.model_config().id,
            token_count=4_000,
        )

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        """Render the requested paths without dumping file contents."""
        path = arguments.get("path")
        if isinstance(path, str):
            return self._truncate(path)

        paths = arguments.get("paths")
        if isinstance(paths, list):
            return "\n".join(f"- {self._truncate(path)}" for path in paths)

        return "(no files)"

    @override
    def format_result_log(self, result: Any) -> str:
        """Summarize the returned content."""
        text = str(result)
        if not text:
            return "empty result"

        file_count = text.count("===== ")
        parts = [f"{len(text.splitlines())} lines", f"{len(text)} chars"]
        if file_count:
            parts.insert(0, f"{file_count} file(s)")
        return " | ".join(parts)

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= _TRUNCATE_LENGTH:
            return value
        return value[:_TRUNCATE_LENGTH] + "..."
