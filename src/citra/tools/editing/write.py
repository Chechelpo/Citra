from typing import Any, override

from citra.sandbox.filesystem_ops import WriteInput

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..capabilities import ToolCapabilities
from ..tool import Tool
from ._post_edit import post_edit_result

_LOG_CONTENT_LIMIT = 2_000


def _write_definition(
    *,
    name: str,
    path_name: str,
    content_name: str = "content",
    description: str,
    path_description: str,
    content_description: str = (
        "Complete content that the file should contain."
    ),
    include_diagnostics: bool = False,
    include_line_count: bool = False,
) -> ChatCompletionTool:
    """Handle write definition."""
    properties: list[JsonProperty] = [
        JsonProperty(
            name=path_name,
            schema=JsonSchema.string(
                description=path_description,
            ),
        ),
        JsonProperty(
            name=content_name,
            schema=JsonSchema.string(
                description=content_description,
            ),
        ),
    ]

    if include_line_count:
        properties.append(
            JsonProperty(
                name="line_count",
                schema=JsonSchema.integer(
                    description=(
                        "Number of lines in the complete file content."
                    ),
                ),
            )
        )

    if include_diagnostics:
        properties.append(
            JsonProperty(
                name="diagnostics",
                schema=JsonSchema.boolean(
                    description=(
                        "Deprecated compatibility flag. LSP diagnostics "
                        "and configured lint fixes and checks run automatically "
                        "after every successful write."
                    ),
                ),
                required=False,
            )
        )

    properties.append(
        JsonProperty(
            name="auto_fix",
            schema=JsonSchema.boolean(
                description=(
                    "Whether configured lint fixers may rewrite the file after "
                    "this write. Defaults to the project setting. Set false for "
                    "an intermediate write; lint checks and LSP diagnostics still run."
                ),
            ),
            required=False,
        )
    )

    return ChatCompletionTool(
        function=FunctionDefinition(
            name=name,
            description=description,
            parameters=JsonSchema.object(
                properties=tuple(properties),
                additional_properties=False,
            ),
        ),
    )


class Write(Tool):
    """
    Writes or replaces a complete text file.

    Relative paths are resolved against the current project.
    """

    TOOL_ID = "write"
    CAPABILITIES = ToolCapabilities()

    # ------------------------------------------------------------------
    # Citra-native fallback
    # ------------------------------------------------------------------

    CITRA_DEFINITION = _write_definition(
        name="write",
        path_name="path",
        description=(
            "Write complete text content to a file. "
            "Creates the file if it does not exist and completely "
            "overwrites it if it does exist. Writes are restricted "
            "to the current project and lifecycle scratch directories. "
            "Use edit instead when only a specific existing fragment "
            "should be changed. After a successful write, Citra "
            "automatically runs configured project lint fixes and checks, "
            "then available LSP diagnostics."
        ),
        path_description=(
            "Destination file path. Relative paths are resolved "
            "against the current project."
        ),
        include_diagnostics=True,
    )

    # ------------------------------------------------------------------
    # Model-family/native-harness definitions
    # ------------------------------------------------------------------

    # Claude Code style:
    #
    #   Write(
    #       file_path: str,
    #       content: str,
    #   )

    # Gemini CLI:
    #
    #   write_file(
    #       file_path: str,
    #       content: str,
    #   )

    # Qwen Code currently has the same external shape as Gemini CLI.

    # Current Kimi Code:
    #
    #   Write(
    #       path: str,
    #       content: str,
    #       mode?: "overwrite" | "append",
    #   )
    #
    # Citra's Write primitive is intentionally whole-file/create-or-
    # overwrite only, so the optional append mode is not advertised here.
    # Add it once the filesystem worker has a real append primitive.

    # ZCode exposes the same important tool-call shape as Claude Code:
    #
    #   Write(
    #       file_path: str,
    #       content: str,
    #   )

    # ------------------------------------------------------------------
    # Harness-specific definitions
    #
    # These are recorded separately because they are genuinely different,
    # but cannot be selected correctly from model_id alone.
    # ------------------------------------------------------------------



    # Recent Cline generations use write_to_file as their whole-file
    # operation. Keep this as a harness profile rather than assigning
    # arbitrary model families to it.

    # Roo/Kilo lineage historically adds a required line_count.

    # Copilot CLI's patch surface is deliberately not exposed by this tool.
    # Its `create(path, file_text)` operation is create-only and therefore
    # does not have the same semantics as this class.

    # ------------------------------------------------------------------
    # Definition resolution
    # ------------------------------------------------------------------

    @classmethod
    @override
    def definition_for_context(
        cls,
        context: ExecutionContext,
    ) -> ChatCompletionTool:
        """Return the tool's model-independent definition."""
        del context
        return cls.CITRA_DEFINITION

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        """Initialize the instance."""
        super().__init__(
            context=context,
        )

    # ------------------------------------------------------------------
    # Argument normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _path_from_arguments(
        arguments: dict[str, Any],
    ) -> str:
        """Handle path from arguments."""
        for name in (
            "path",
            "file_path",
            "filePath",
        ):
            if name in arguments:
                return arguments[name]

        raise ValueError(
            "Write arguments contained no recognized path parameter."
        )

    @staticmethod
    def _content_from_arguments(
        arguments: dict[str, Any],
    ) -> str:
        """Handle content from arguments."""
        for name in (
            "content",
            "file_text",
        ):
            if name in arguments:
                return arguments[name]

        raise ValueError(
            "Write arguments contained no recognized content parameter."
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Execute the execute operation."""
        path = self._path_from_arguments(
            arguments
        )

        content = self._content_from_arguments(
            arguments
        )

        result: str = self.context.filesystem.execute(
            WriteInput(path, content)
        ).to_budgeted(model_id=self.context.model_config().id,token_count=4_000)

        if result != "ok":
            return result

        return post_edit_result(
            self.context,
            path,
            auto_fix=arguments.get("auto_fix"),
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Handle format call log."""
        path = self._path_from_arguments(
            arguments
        )

        content = self._content_from_arguments(
            arguments
        )

        shown_content = content
        omitted = len(content) - _LOG_CONTENT_LIMIT
        if omitted > 0:
            shown_content = (
                content[:_LOG_CONTENT_LIMIT]
                + f"\n… <truncated {omitted} chars>"
            )

        return f"path={path} | {len(content)} chars\n{shown_content}"

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        """Handle format result log."""
        return str(result)
