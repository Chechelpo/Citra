from citra.sandbox.filesystem_ops import GlobInput
from typing import Any, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..capabilities import ToolCapabilities
from ..tool import Tool


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
    CAPABILITIES = ToolCapabilities()

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


    # ------------------------------------------------------------------
    # Qwen Code
    #
    # glob(
    #     pattern,
    #     path?,
    # )
    # ------------------------------------------------------------------


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


    # ------------------------------------------------------------------
    # Reference harness definitions
    #
    # OpenCode and Crush happen to share the same actual function shape.
    # Keep these constants around for future harness-aware resolution.
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
            GlobInput.parse(dict(arguments))
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
