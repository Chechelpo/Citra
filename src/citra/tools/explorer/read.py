"""Model-facing literal file reader backed by the sandbox worker."""

from typing import Any, override

from citra.sandbox.filesystem_ops import ReadInput
from citra.sandbox.filesystem_ops.read import ReadOutput, ReadSymbol

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..capabilities import ToolCapabilities
from ..tool import Tool

_DEFAULT_MAX_TOKENS = 4_000
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
                "Read one or more files by literal path. "
                "For source files, a compact function/class index with line "
                "spans is returned before the requested content. "
                "Optional line ranges can be used to read a section of a file. "
                "If no line range is provided, the complete document is read. "
                "Large results may be limited by max_tokens. "
                "Use glob or find to discover files first; glob patterns are "
                "not accepted."
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
                    JsonProperty(
                        name="from_line",
                        schema=JsonSchema.integer(
                            description=(
                                "First 1-based line to include. Defaults to the "
                                "first line of the document."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="to_line",
                        schema=JsonSchema.integer(
                            description=(
                                "Last 1-based line to include. Defaults to the "
                                "last line of the document."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="max_tokens",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum number of output tokens. Defaults to "
                                "the tool's standard read budget."
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
        request = ReadInput.parse(dict(arguments))

        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else _DEFAULT_MAX_TOKENS
        )

        output = self.context.filesystem.execute(request)

        if isinstance(output, ReadOutput) and hasattr(self.context, "repo_map"):
            try:
                definitions = self.context.repo_map.definitions_for_paths(
                    entry.path for entry in output.entries
                )
            except RuntimeError:
                # Symbol indexing is an enhancement; an unavailable parser must
                # not prevent the requested file contents from being returned.
                definitions = {}

            output = output.with_symbols(
                {
                    path: tuple(
                        ReadSymbol(
                            name=definition.name,
                            from_line=definition.from_line + 1,
                            to_line=definition.to_line + 1,
                        )
                        for definition in path_definitions
                    )
                    for path, path_definitions in definitions.items()
                }
            )

        return output.to_budgeted(
            model_id=self.context.model_config().id,
            token_count=max_tokens,
        )

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        """Render the requested paths without dumping file contents."""
        path = arguments.get("path")
        if isinstance(path, str):
            return self._format_path_options(
                self._truncate(path),
                arguments,
            )

        paths = arguments.get("paths")
        if isinstance(paths, list):
            return "\n".join(
                self._format_path_options(
                    self._truncate(path),
                    arguments,
                )
                for path in paths
            )

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
    def _format_path_options(
        path: str,
        arguments: dict[str, Any],
    ) -> str:
        """Include optional read bounds in call logging."""
        options: list[str] = [path]

        from_line = arguments.get("from_line")
        to_line = arguments.get("to_line")
        max_tokens = arguments.get("max_tokens")

        if from_line is not None or to_line is not None:
            options.append(
                f"lines={from_line or 1}-{to_line or 'EOF'}"
            )

        if max_tokens is not None:
            options.append(
                f"max_tokens={max_tokens}"
            )

        return " ".join(options)

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= _TRUNCATE_LENGTH:
            return value

        return value[:_TRUNCATE_LENGTH] + "..."
