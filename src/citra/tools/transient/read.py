"""Model-facing read tool backed by the sandbox filesystem worker."""

from typing import Any, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..tool import Tool

_TRUNCATE_LENGTH = 120


class Read(Tool):
    """Read literal paths or globs without host-process filesystem I/O."""

    MAX_REQUESTS = 20
    MAX_FILES_PER_CALL = 20

    READ_REQUEST_SCHEMA = JsonSchema.object(
        properties=(
            JsonProperty(
                name="path",
                schema=JsonSchema.string(
                    description=(
                        "File path or glob pattern. Examples: 'main.py', "
                        "'src/**/*.py', or '@source/pyproject.toml'."
                    ),
                ),
            ),
            JsonProperty(
                name="offset",
                schema=JsonSchema.integer(
                    description="Zero-based line offset. Defaults to 0.",
                ),
                required=False,
            ),
            JsonProperty(
                name="limit",
                schema=JsonSchema.integer(
                    description="Maximum number of lines to return.",
                ),
                required=False,
            ),
        ),
        additional_properties=False,
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="read",
            description=(
                "Read one or more files visible inside Citra's filesystem sandbox and "
                "return line-numbered text. Literal paths and recursive glob "
                "patterns are supported. Relative paths resolve from the "
                "persistent agent workspace; @source addresses the original "
                "read-only project and @tmp addresses disposable storage. "
                "Use requests for a batch with independent offsets/limits. "
                "At most 20 concrete files are returned per call. PDF and "
                "notebook conversion also occurs inside the sandbox."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description="Single file path or glob pattern.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="offset",
                        schema=JsonSchema.integer(
                            description="Zero-based offset for top-level path.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="limit",
                        schema=JsonSchema.integer(
                            description="Line limit for top-level path.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="requests",
                        schema=JsonSchema.array(
                            READ_REQUEST_SCHEMA,
                            description="Batch of independent read requests.",
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    def __init__(self, context: ExecutionContext) -> None:
        super().__init__(context=context, definition=self.DEFINITION)

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        return self.context.filesystem.execute(
            "read",
            arguments,
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        path = arguments.get("path")
        requests = arguments.get("requests")

        if path is not None:
            parts = [f"path={self._truncate(path)}"]

            offset = arguments.get("offset")
            if offset:
                parts.append(f"offset={offset}")

            limit = arguments.get("limit")
            if limit is not None:
                parts.append(f"limit={limit}")

            return " | ".join(parts)

        if requests:
            return f"batch={len(requests)} request(s)"

        return "no path"

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        text = str(result)

        if not text:
            return "empty result"

        file_count = text.count("===== ")

        lines = text.splitlines()

        parts = [f"{len(lines)} lines", f"{len(text)} chars"]

        if file_count:
            parts.insert(0, f"{file_count} file(s)")

        return " | ".join(parts)

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= _TRUNCATE_LENGTH:
            return value
        return value[:_TRUNCATE_LENGTH] + "..."

