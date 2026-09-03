from citra.sandbox.filesystem_ops import GlobInput
from typing import Any, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..tool import Tool, ToolDefinition


def _glob_definition(
    *,
    name: str,
    pattern_name: str,
    path_name: str,
    description: str,
) -> ChatCompletionTool:
    """Handle glob definition."""
    return ChatCompletionTool(
        function=FunctionDefinition(
            name=name,
            description=description,
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name=pattern_name,
                        schema=JsonSchema.string(
                            description=(
                                "Glob pattern to match files against, "
                                "for example '*.py', 'src/**/*.py', "
                                "or '**/config.toml'."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name=path_name,
                        schema=JsonSchema.string(
                            description=(
                                "Directory to search in. "
                                "Defaults to the current project."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )


class Glob(Tool):
    """Represent Glob."""
    CACHEABLE = True
    INVALIDATES_TOOL_CACHE = False

    """
    Finds filesystem entries matching a glob pattern.

    Results are sorted by modification time, newest first.
    """

    TOOL_ID = "glob"

    # ------------------------------------------------------------------
    # Citra-native fallback
    #
    # glob(
    #     pattern,
    #     path?,
    # )
    # ------------------------------------------------------------------

    CITRA_DEFINITION = _glob_definition(
        name="glob",
        pattern_name="pattern",
        path_name="path",
        description=(
            "Find files and directories using a glob pattern. "
            "Supports recursive patterns such as '**/*.py'. "
            "Results are sorted by modification time with the most "
            "recently modified files first."
        ),
    )

    # ------------------------------------------------------------------
    # Claude Code
    #
    # Glob(
    #     pattern,
    #     path?,
    # )
    # ------------------------------------------------------------------

    CLAUDE_CODE_DEFINITION = _glob_definition(
        name="Glob",
        pattern_name="pattern",
        path_name="path",
        description=(
            "Find files matching a glob pattern. "
            "Returns matching paths sorted by modification time."
        ),
    )

    # ------------------------------------------------------------------
    # Gemini CLI
    #
    # Current Gemini tool declarations use:
    #
    # glob(
    #     pattern,
    #     dir_path?,
    #     case_sensitive?,
    #     respect_git_ignore?,
    #     respect_gemini_ignore?,
    # )
    #
    # Citra's filesystem glob does not currently expose those behavioral
    # flags, so only advertise the compatible pattern + directory subset.
    # ------------------------------------------------------------------

    GEMINI_CLI_DEFINITION = _glob_definition(
        name="glob",
        pattern_name="pattern",
        path_name="dir_path",
        description=(
            "Find files matching a glob pattern across the current project. "
            "Results are sorted by modification time with newest files first."
        ),
    )

    # ------------------------------------------------------------------
    # Qwen Code
    #
    # glob(
    #     pattern,
    #     path?,
    # )
    # ------------------------------------------------------------------

    QWEN_CODE_DEFINITION = _glob_definition(
        name="glob",
        pattern_name="pattern",
        path_name="path",
        description=(
            "Find files matching a glob pattern. "
            "Returns absolute paths sorted by modification time, newest first."
        ),
    )

    # ------------------------------------------------------------------
    # Kimi Code
    #
    # Glob(
    #     pattern,
    #     path?,
    #     include_ignored?,
    # )
    #
    # include_ignored is deliberately omitted until Citra's filesystem
    # worker supports that semantic explicitly.
    # ------------------------------------------------------------------

    KIMI_CODE_DEFINITION = _glob_definition(
        name="Glob",
        pattern_name="pattern",
        path_name="path",
        description=(
            "Find files by glob pattern within a directory. "
            "The search directory defaults to the working directory."
        ),
    )

    # ------------------------------------------------------------------
    # ZCode / GLM
    #
    # ZCode exposes Claude-style capitalized filesystem tools. This class
    # supplies its Glob-compatible schema.
    #
    # Its complete Glob JSON schema is not publicly documented as
    # clearly as the others, so preserve the well-established
    # pattern/path shape.
    # ------------------------------------------------------------------

    ZCODE_DEFINITION = _glob_definition(
        name="Glob",
        pattern_name="pattern",
        path_name="path",
        description=(
            "Find files matching a glob pattern in a directory."
        ),
    )

    # ------------------------------------------------------------------
    # Reference harness definitions
    #
    # OpenCode and Crush happen to share the same actual function shape.
    # Keep these constants around for future harness-aware resolution.
    # ------------------------------------------------------------------

    OPENCODE_DEFINITION = _glob_definition(
        name="glob",
        pattern_name="pattern",
        path_name="path",
        description=(
            "Find files matching a glob pattern. "
            "The current working directory is used when path is omitted."
        ),
    )

    CRUSH_DEFINITION = _glob_definition(
        name="glob",
        pattern_name="pattern",
        path_name="path",
        description=(
            "Find files matching a glob pattern. "
            "The current working directory is used when path is omitted."
        ),
    )

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        """Handle definitions for context."""
        del context

        return (
            # Claude Code.
            ToolDefinition(
                definition=cls.CLAUDE_CODE_DEFINITION,
                model_family_matchers=(
                    "claude",
                ),
            ),

            # Gemini CLI.
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

            # Kimi Code.
            ToolDefinition(
                definition=cls.KIMI_CODE_DEFINITION,
                model_family_matchers=(
                    "kimi",
                    "moonshot",
                ),
            ),

            # ZCode / GLM.
            ToolDefinition(
                definition=cls.ZCODE_DEFINITION,
                model_family_matchers=(
                    "glm",
                ),
            ),

            # Universal Citra fallback.
            ToolDefinition(
                definition=cls.CITRA_DEFINITION,
            ),
        )

    # ------------------------------------------------------------------
    # Argument normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_arguments(
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle normalize arguments."""
        pattern = arguments.get(
            "pat",
            arguments.get("pattern"),
        )

        path = arguments.get(
            "path",
            arguments.get("dir_path"),
        )

        normalized: dict[str, Any] = {
            "pat": pattern,
        }

        if path is not None:
            normalized["path"] = path

        return normalized

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Execute the execute operation."""
        return self.context.filesystem.execute(
            GlobInput.parse(arguments)
        ).to_budgeted(model_id=self.context.model_config().id, token_count=4_000)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Handle format call log."""
        normalized = self._normalize_arguments(
            arguments
        )

        pattern = normalized.get(
            "pat",
            "",
        )

        path = normalized.get(
            "path",
        )

        if path is not None:
            return (
                f"pat={pattern} | "
                f"path={path}"
            )

        return f"pat={pattern}"

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        """Handle format result log."""
        text = str(result)

        if not text or text == "none":
            return "no matches"

        lines = text.splitlines()

        return (
            f"{len(lines)} match(es)"
        )
