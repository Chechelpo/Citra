"""Model-facing content search backed by ripgrep."""
from citra.sandbox.filesystem_ops import GrepInput
from typing import Any, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..capabilities import ToolCapabilities
from ..tool import Tool, ToolDefinition

_OUTPUT_MODES = ("content", "files_with_matches", "count")


def _grep_definition(
    *,
    name: str,
    pattern_name: str,
    path_name: str,
    glob_name: str,
    description: str,
) -> ChatCompletionTool:
    """Build one harness-shaped grep tool definition."""
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
                                "Regular expression to search for "
                                "(use literal=true for fixed strings)."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name=path_name,
                        schema=JsonSchema.string(
                            description=(
                                "File or directory to search in. "
                                "Defaults to the current project."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name=glob_name,
                        schema=JsonSchema.string(
                            description=(
                                "Glob filter for searched files, "
                                "for example '*.py' or 'src/**/*.ts'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="output_mode",
                        schema=JsonSchema.string(
                            description=(
                                "content: path:line:text; "
                                "files_with_matches: matching paths; "
                                "count: per-file match counts."
                            ),
                            enum=_OUTPUT_MODES,
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="case_insensitive",
                        schema=JsonSchema.boolean(
                            description=(
                                "Match without regard to case."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="max_results",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum matches, files, or counts "
                                "to return (1-1000, default 100)."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="literal",
                        schema=JsonSchema.boolean(
                            description=(
                                "Treat the pattern as a fixed string "
                                "instead of a regular expression."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )


class Grep(Tool):
    """
    Search file contents with ripgrep semantics.

    The worker prefers the ``rg`` binary when the sandbox exposes it and
    falls back to a Python regex walk otherwise. Results are sorted by
    path, then line number, so output is deterministic across backends.
    """

    TOOL_ID = "grep"
    CAPABILITIES = ToolCapabilities()
    CACHEABLE = True
    INVALIDATES_TOOL_CACHE = False

    # ------------------------------------------------------------------
    # Citra-native fallback
    #
    # grep(
    #     pattern,
    #     path?,
    #     glob?,
    #     output_mode?,
    #     case_insensitive?,
    #     max_results?,
    #     literal?,
    # )
    # ------------------------------------------------------------------

    CITRA_DEFINITION = _grep_definition(
        name="grep",
        pattern_name="pattern",
        path_name="path",
        glob_name="glob",
        description=(
            "Search file contents with a regular expression. "
            "Uses ripgrep when available with a Python fallback. "
            "Results are sorted by path, then line number."
        ),
    )

    # ------------------------------------------------------------------
    # Claude Code
    #
    # Grep(
    #     pattern,
    #     path?,
    #     glob?,
    #     output_mode?,
    #     head_limit?,
    # )
    #
    # Claude's native declaration also carries head_limit/-n style aliases;
    # they are accepted by argument normalization but only the canonical
    # max_results spelling is advertised here.
    # ------------------------------------------------------------------

    CLAUDE_CODE_DEFINITION = _grep_definition(
        name="Grep",
        pattern_name="pattern",
        path_name="path",
        glob_name="glob",
        description=(
            "Search file contents with a regular expression. "
            "Returns matching lines sorted by path and line number."
        ),
    )

    # ------------------------------------------------------------------
    # Gemini CLI
    #
    # search_file_content(
    #     pattern,
    #     dir_path?,
    #     glob?,
    #     output_mode?,
    # )
    #
    # Only the compatible pattern + directory + filter subset is
    # advertised here.
    # ------------------------------------------------------------------

    GEMINI_CLI_DEFINITION = _grep_definition(
        name="search_file_content",
        pattern_name="pattern",
        path_name="dir_path",
        glob_name="glob",
        description=(
            "Search file contents with a regular expression across "
            "the current project."
        ),
    )

    # ------------------------------------------------------------------
    # Qwen Code
    #
    # grep(
    #     pattern,
    #     path?,
    #     glob?,
    #     output_mode?,
    # )
    # ------------------------------------------------------------------

    QWEN_CODE_DEFINITION = _grep_definition(
        name="grep",
        pattern_name="pattern",
        path_name="path",
        glob_name="glob",
        description=(
            "Search file contents with a regular expression. "
            "Returns matches sorted by path and line number."
        ),
    )

    # ------------------------------------------------------------------
    # Kimi Code
    #
    # Grep(
    #     pattern,
    #     path?,
    #     glob?,
    #     output_mode?,
    # )
    # ------------------------------------------------------------------

    KIMI_CODE_DEFINITION = _grep_definition(
        name="Grep",
        pattern_name="pattern",
        path_name="path",
        glob_name="glob",
        description=(
            "Search file contents by pattern within a directory. "
            "The search directory defaults to the working directory."
        ),
    )

    # ------------------------------------------------------------------
    # ZCode / GLM
    #
    # ZCode exposes Claude-style capitalized filesystem tools. Preserve
    # the well-established pattern/path/glob shape.
    # ------------------------------------------------------------------

    ZCODE_DEFINITION = _grep_definition(
        name="Grep",
        pattern_name="pattern",
        path_name="path",
        glob_name="glob",
        description=(
            "Search file contents with a regular expression."
        ),
    )

    # ------------------------------------------------------------------
    # Reference harness definitions
    # ------------------------------------------------------------------

    OPENCODE_DEFINITION = _grep_definition(
        name="grep",
        pattern_name="pattern",
        path_name="path",
        glob_name="glob",
        description=(
            "Search file contents with a regular expression. "
            "The current working directory is used when path is omitted."
        ),
    )

    CRUSH_DEFINITION = _grep_definition(
        name="grep",
        pattern_name="pattern",
        path_name="path",
        glob_name="glob",
        description=(
            "Search file contents with a regular expression. "
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
    def _normalize_arguments(
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Map native harness spellings onto canonical grep arguments."""
        normalized: dict[str, Any] = {}

        if "pattern" in arguments:
            normalized["pattern"] = arguments["pattern"]
        elif "regex" in arguments:
            normalized["pattern"] = arguments["regex"]
        elif "query" in arguments:
            normalized["pattern"] = arguments["query"]

        for name in ("path", "dir_path", "directory", "dir"):
            if name in arguments:
                normalized["path"] = arguments[name]
                break

        for name in ("glob", "include", "file_pattern"):
            if name in arguments:
                normalized["glob"] = arguments[name]
                break

        if "output_mode" in arguments:
            normalized["output_mode"] = arguments["output_mode"]
        elif "mode" in arguments:
            normalized["output_mode"] = arguments["mode"]

        for name in ("case_insensitive", "caseInsensitive", "-i"):
            if name in arguments:
                normalized["case_insensitive"] = arguments[name]
                break

        for name in ("max_results", "head_limit", "limit", "maxResults"):
            if name in arguments:
                normalized["max_results"] = arguments[name]
                break

        for name in ("literal", "fixed_strings"):
            if name in arguments:
                normalized["literal"] = arguments[name]
                break

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
            GrepInput.parse(self._normalize_arguments(arguments))
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
            "pattern",
            "",
        )

        path = normalized.get(
            "path",
            ".",
        )

        parts = [
            f"pattern={pattern}",
            f"path={path}",
        ]

        glob = normalized.get("glob")
        if glob is not None:
            parts.append(f"glob={glob}")

        mode = normalized.get("output_mode")
        if mode is not None:
            parts.append(f"mode={mode}")

        return " | ".join(parts)

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        """Handle format result log."""
        text = str(result)

        if not text or text == "no matches":
            return "no matches"

        lines = [
            line
            for line in text.splitlines()
            if line and not line.startswith("... <truncated")
        ]

        return f"{len(lines)} match(es)"
