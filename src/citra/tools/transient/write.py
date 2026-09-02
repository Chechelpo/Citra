from citra.sandbox.filesystem_ops import WriteInput
from typing import Any, override

from ...context import ExecutionContext
from ..tool import Tool, ToolDefinition
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ._post_edit import post_edit_result


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
                        "and configured lint checks run automatically "
                        "after every successful write."
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
            "automatically runs available LSP diagnostics and configured "
            "project lint checks."
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
    CLAUDE_CODE_DEFINITION = _write_definition(
        name="Write",
        path_name="file_path",
        description=(
            "Write complete content to a file. "
            "Creates the file if it does not exist and overwrites "
            "the existing file if it does."
        ),
        path_description="Path of the file to write.",
        content_description="Complete contents of the file.",
    )

    # Gemini CLI:
    #
    #   write_file(
    #       file_path: str,
    #       content: str,
    #   )
    GEMINI_CLI_DEFINITION = _write_definition(
        name="write_file",
        path_name="file_path",
        description=(
            "Writes content to a specified file, overwriting it if "
            "it exists or creating it if it does not."
        ),
        path_description="Path to the file.",
        content_description="Complete content to write to the file.",
    )

    # Qwen Code currently has the same external shape as Gemini CLI.
    QWEN_CODE_DEFINITION = _write_definition(
        name="write_file",
        path_name="file_path",
        description=(
            "Write content to a specified file. If the file exists, "
            "it is overwritten; otherwise it is created."
        ),
        path_description="Absolute path to the file to write.",
        content_description="Complete content to write into the file.",
    )

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
    KIMI_CODE_DEFINITION = _write_definition(
        name="Write",
        path_name="path",
        description=(
            "Create or overwrite a file with complete text content. "
            "Missing parent directories may be created automatically."
        ),
        path_description="Path to the file to write.",
        content_description="Complete content to write to the file.",
    )

    # ZCode exposes the same important tool-call shape as Claude Code:
    #
    #   Write(
    #       file_path: str,
    #       content: str,
    #   )
    ZCODE_DEFINITION = _write_definition(
        name="Write",
        path_name="file_path",
        description=(
            "Write complete content to a file, creating or "
            "overwriting the target file."
        ),
        path_description="Path of the file to write.",
        content_description="Complete file contents.",
    )

    # ------------------------------------------------------------------
    # Harness-specific definitions
    #
    # These are recorded separately because they are genuinely different,
    # but cannot be selected correctly from model_id alone.
    # ------------------------------------------------------------------

    OPENCODE_DEFINITION = _write_definition(
        name="write",
        path_name="filePath",
        description=(
            "Write a file to the local filesystem. "
            "Creates new files and overwrites existing files."
        ),
        path_description=(
            "The absolute path to the file to write."
        ),
        content_description="The content to write to the file.",
    )

    CRUSH_DEFINITION = _write_definition(
        name="write",
        path_name="file_path",
        description=(
            "Write complete content to a file, creating or "
            "overwriting it."
        ),
        path_description="Path to the file to write.",
        content_description="Complete file contents.",
    )

    # Recent Cline generations use write_to_file as their whole-file
    # operation. Keep this as a harness profile rather than assigning
    # arbitrary model families to it.
    CLINE_DEFINITION = _write_definition(
        name="write_to_file",
        path_name="path",
        description=(
            "Write complete content to a file. Creates a new file "
            "or completely replaces the existing file."
        ),
        path_description="Path of the file to write.",
        content_description="Complete content of the file.",
    )

    # Roo/Kilo lineage historically adds a required line_count.
    ROO_KILO_DEFINITION = _write_definition(
        name="write_to_file",
        path_name="path",
        description=(
            "Write complete content to a file, creating or "
            "overwriting the target."
        ),
        path_description="Path of the file to write.",
        content_description="Complete content of the file.",
        include_line_count=True,
    )

    # Copilot CLI is deliberately NOT placed in definitions_for_context().
    # Its `create(path, file_text)` operation is create-only and therefore
    # does not have the same semantics as this class.
    COPILOT_CREATE_DEFINITION = _write_definition(
        name="create",
        path_name="path",
        content_name="file_text",
        description=(
            "Create a new file with the specified content. "
            "The target file must not already exist."
        ),
        path_description=(
            "Absolute path of the new file."
        ),
        content_description="Content of the file to create.",
    )

    # ------------------------------------------------------------------
    # Definition resolution
    # ------------------------------------------------------------------

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        del context

        return (
            # Anthropic / Claude Code.
            ToolDefinition(
                definition=cls.CLAUDE_CODE_DEFINITION,
                model_family_matchers=(
                    "claude",
                ),
            ),

            # Google Gemini CLI.
            ToolDefinition(
                definition=cls.GEMINI_CLI_DEFINITION,
                model_family_matchers=(
                    "gemini",
                ),
            ),

            # Qwen Code.
            ToolDefinition(
                definition=cls.QWEN_CODE_DEFINITION,
                model_family_matchers=(
                    "qwen",
                ),
            ),

            # Current Kimi Code.
            ToolDefinition(
                definition=cls.KIMI_CODE_DEFINITION,
                model_family_matchers=(
                    "kimi",
                    "moonshot",
                ),
            ),

            # Z.ai ZCode / GLM.
            ToolDefinition(
                definition=cls.ZCODE_DEFINITION,
                model_family_matchers=(
                    "glm",
                ),
            ),

            # Everything without a sufficiently well-established native
            # whole-file tool stays on Citra's own schema.
            ToolDefinition(
                definition=cls.CITRA_DEFINITION,
            ),
        )

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
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
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        path = self._path_from_arguments(
            arguments
        )

        content = self._content_from_arguments(
            arguments
        )

        return (
            f"path={path} | "
            f"{len(content)} chars"
        )

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        return str(result)
