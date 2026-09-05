"""Model-facing combined filesystem search (filename + extension + content)."""

from typing import Any, override

from citra.sandbox.filesystem_ops import FindInput

from ...context import ExecutionContext
from ...sandbox.filesystem_ops.find import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    OUTPUT_MODES,
)
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..capabilities import ToolCapabilities
from ..tool import Tool, ToolDefinition


# Single canonical schema across every model family. The harness-targeted
# definitions below are intentionally identical to the Citra fallback: the
# spec signature already uses the snake-free ``camelCase`` form, and there is
# no truthful Claude/Gemini/Qwen/Kimi/GLM callable-tool shape to imitate
# yet. New families can subclass the helper below without breaking compat.
def _find_definition(
    *,
    name: str,
    description: str,
) -> ChatCompletionTool:
    """Build one harness-shaped ``find`` tool definition."""
    return ChatCompletionTool(
        function=FunctionDefinition(
            name=name,
            description=description,
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="paths",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Root paths to search. Each entry may be a "
                                "file or a directory; directories are walked "
                                "recursively. Missing paths are skipped "
                                "without raising."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="name",
                        schema=JsonSchema.string(
                            description=(
                                "Glob pattern matched against the filename. "
                                "Use an array of patterns for OR matching. "
                                "Examples: '*.ts', 'src/**/*.tsx', "
                                "'test_*'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="extensions",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Whitelist of file extensions. Leading dots "
                                "are optional ('ts' and '.ts' are equivalent)."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description=(
                                "Substring (or regex) to search for inside "
                                "each candidate file. Combined with "
                                "``regex=true`` for full regular expression "
                                "matching."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="regex",
                        schema=JsonSchema.boolean(
                            description=(
                                "When true, treat ``content`` as a regular "
                                "expression. Default: false."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="caseSensitive",
                        schema=JsonSchema.boolean(
                            description=(
                                "When false, ``content`` matches without "
                                "regard to case. Default: true."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="exclude",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Glob patterns for directories that should "
                                "not be traversed. Patterns are matched "
                                "against the directory basename and the "
                                "relative POSIX path; 'node_modules/**' "
                                "prunes the matching subtree entirely."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="maxDepth",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum directory depth to descend. 0 keeps "
                                "the root and its direct files; 1 adds one "
                                "level of subdirectories, and so on. Files "
                                "under a directory at ``maxDepth`` are not "
                                "visited."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="context",
                        schema=JsonSchema.integer(
                            description=(
                                "Number of lines to include before and after "
                                "each content match. Only meaningful when "
                                "``content`` is provided. Default: 0."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="limit",
                        schema=JsonSchema.integer(
                            description=(
                                f"Maximum number of results to return "
                                f"(1..{MAX_LIMIT}, default {DEFAULT_LIMIT}). "
                                "Excess results are dropped and ``truncated`` "
                                "is set when ``mode='matches'``."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="mode",
                        schema=JsonSchema.string(
                            description=(
                                "``files`` (default) returns matching file "
                                "paths in a flat array; ``matches`` returns "
                                "per-file structured hits with line numbers "
                                "and context."
                            ),
                            enum=OUTPUT_MODES,
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )


class Find(Tool):
    """
    Combined filesystem search (find + glob + grep).

    The agent supplies one or more root ``paths`` and any combination of
    filename globs (``name``), extension whitelists (``extensions``),
    content expressions (``content`` with optional ``regex`` /
    ``caseSensitive``), directory pruning (``exclude``), traversal depth
    (``maxDepth``), per-match context (``context``), and a result cap
    (``limit``). The single output mode (``files`` for a flat list of
    paths, ``matches`` for structured per-file hits) is selected with
    ``mode``.

    Implementation delegates to the sandboxed ``FindInput`` worker; the
    transient tool is responsible for schema exposure, argument
    normalization across harness spellings, and log rendering.
    """

    TOOL_ID = "find"
    CAPABILITIES = ToolCapabilities()
    CACHEABLE = True
    INVALIDATES_TOOL_CACHE = False

    # ------------------------------------------------------------------
    # Citra-native fallback
    #
    # find(
    #     paths,
    #     name?,
    #     extensions?,
    #     content?,
    #     regex?,
    #     caseSensitive?,
    #     exclude?,
    #     maxDepth?,
    #     context?,
    #     limit?,
    #     mode?,
    # )
    # ------------------------------------------------------------------

    CITRA_DEFINITION = _find_definition(
        name="find",
        description=(
            "Combined filesystem search (find + glob + grep). Walks the "
            "given root paths and returns either matching file paths or "
            "structured per-file hits. Supports filename globs, extension "
            "filters, content search (literal or regex), case sensitivity, "
            "directory pruning, traversal depth, per-match context lines, "
            "and a result limit. Use this when you need a single primitive "
            "that covers the find/glob/grep triangle; reach for Grep, Glob, "
            "or Tree directly only when their specialized output is more "
            "useful."
        ),
    )

    # ------------------------------------------------------------------
    # Model-family profiles
    #
    # The Citra schema is reused for every family. There is no established
    # Claude / Gemini / Qwen / Kimi / GLM ``find`` callable-tool surface to
    # mimic; new families can subclass ``_find_definition`` and register an
    # additional ``ToolDefinition`` here without breaking compatibility.
    # ------------------------------------------------------------------

    CLAUDE_CODE_DEFINITION = CITRA_DEFINITION
    GEMINI_CLI_DEFINITION = CITRA_DEFINITION
    QWEN_CODE_DEFINITION = CITRA_DEFINITION
    KIMI_CODE_DEFINITION = CITRA_DEFINITION
    ZCODE_DEFINITION = CITRA_DEFINITION

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        """Return the model-facing definitions for the active context."""
        del context

        return (
            ToolDefinition(
                definition=cls.CLAUDE_CODE_DEFINITION,
                model_family_matchers=("claude",),
            ),
            ToolDefinition(
                definition=cls.GEMINI_CLI_DEFINITION,
                model_family_matchers=("gemini",),
            ),
            ToolDefinition(
                definition=cls.QWEN_CODE_DEFINITION,
                model_family_matchers=("qwen",),
            ),
            ToolDefinition(
                definition=cls.KIMI_CODE_DEFINITION,
                model_family_matchers=("kimi", "moonshot"),
            ),
            ToolDefinition(
                definition=cls.ZCODE_DEFINITION,
                model_family_matchers=("glm",),
            ),
            ToolDefinition(
                definition=cls.CITRA_DEFINITION,
            ),
        )

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        """Initialize the instance."""
        super().__init__(context=context)

    # ------------------------------------------------------------------
    # Argument normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_arguments(
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Map native harness spellings onto canonical ``find`` arguments.

        The model-facing schema already uses the spec's ``camelCase`` keys,
        but harness aliases (``roots``, ``pattern``, ``include``, ``maxDepth``
        vs ``max_depth``) are accepted here so existing call sites keep
        working.
        """
        normalized: dict[str, Any] = {}

        paths = arguments.get("paths")
        if paths is None:
            paths = arguments.get("roots")
        if paths is None:
            root = arguments.get("path")
            if root is not None:
                paths = [root]
        if paths is not None:
            normalized["paths"] = paths

        if "name" in arguments:
            normalized["name"] = arguments["name"]
        elif "pattern" in arguments:
            normalized["name"] = arguments["pattern"]
        elif "include" in arguments:
            normalized["name"] = arguments["include"]

        if "extensions" in arguments:
            normalized["extensions"] = arguments["extensions"]
        elif "ext" in arguments:
            normalized["ext"] = arguments["ext"]

        for key in (
            "content",
            "regex",
            "caseSensitive",
            "case_sensitive",
            "caseInsensitive",
            "exclude",
            "context",
        ):
            if key in arguments:
                normalized[key] = arguments[key]

        for key in ("maxDepth", "max_depth"):
            if key in arguments:
                normalized["maxDepth"] = arguments[key]
                break

        for key in ("limit", "max_results", "maxResults"):
            if key in arguments:
                normalized["limit"] = arguments[key]
                break

        for key in ("mode", "output_mode"):
            if key in arguments:
                normalized["mode"] = arguments[key]
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
        """Execute the find operation through the sandboxed filesystem."""
        return self.context.filesystem.execute(
            FindInput.parse(self._normalize_arguments(arguments))
        ).to_budgeted(
            model_id=self.context.model_config().id,
            token_count=4_000,
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Render a compact, single-line call log."""
        normalized = self._normalize_arguments(arguments)

        parts: list[str] = []

        paths = normalized.get("paths")
        if paths is not None:
            parts.append(f"paths={self._format_paths(paths)}")

        name = normalized.get("name")
        if name is not None:
            parts.append(f"name={self._format_name(name)}")

        extensions = normalized.get("extensions")
        if extensions is not None:
            parts.append(f"ext={self._format_extensions(extensions)}")

        content = normalized.get("content")
        if content is not None:
            parts.append(f"content={self._truncate(content)}")

        if normalized.get("regex"):
            parts.append("regex=true")

        if "caseSensitive" in normalized:
            case_sensitive = bool(normalized["caseSensitive"])
            if not case_sensitive:
                parts.append("case-insensitive=true")

        exclude = normalized.get("exclude")
        if exclude is not None:
            parts.append(f"exclude={len(exclude)}")

        if "maxDepth" in normalized:
            parts.append(f"maxDepth={normalized['maxDepth']}")

        if normalized.get("context"):
            parts.append(f"context={normalized['context']}")

        if "limit" in normalized:
            parts.append(f"limit={normalized['limit']}")

        if "mode" in normalized:
            parts.append(f"mode={normalized['mode']}")

        return " | ".join(parts) if parts else "find()"

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        """Render a compact, single-line result log."""
        text = str(result)
        if not text or text == "no matches":
            return "no matches"
        if text.endswith("... <truncated: showing first results>"):
            head = text.rsplit("\n... <truncated: showing first results>", 1)[0]
            text = head
        lines = [line for line in text.splitlines() if line]
        return f"{len(lines)} match(es)"

    @staticmethod
    def _format_paths(paths: Any) -> str:
        """Render the ``paths`` argument for the call log."""
        if isinstance(paths, list):
            if len(paths) == 1:
                return str(paths[0])
            return f"{len(paths)}"
        return str(paths)

    @staticmethod
    def _format_name(name: Any) -> str:
        """Render the ``name`` argument for the call log."""
        if isinstance(name, list):
            if len(name) == 1:
                return str(name[0])
            return f"{len(name)}"
        return str(name)

    @staticmethod
    def _format_extensions(extensions: Any) -> str:
        """Render the ``extensions`` argument for the call log."""
        if isinstance(extensions, list):
            return ",".join(str(value) for value in extensions)
        return str(extensions)

    @staticmethod
    def _truncate(value: Any, *, limit: int = 80) -> str:
        """Truncate a long string for the call log."""
        text = str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."
