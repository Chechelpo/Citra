from citra.sandbox.filesystem_ops import TreeInput
from typing import Any, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ...utils.repo_map import (
    DEFAULT_MAP_TOKENS,
    MAX_MAP_TOKENS,
)
from ..capabilities import ToolCapabilities
from ..tool import Tool, ToolDefinition


class Tree(Tool):
    """
    Show an Aider-style structural map of the repository.

    Unlike a filesystem tree/listing, this returns a ranked semantic map
    containing important definitions, signatures, and code locations.
    """

    TOOL_ID = "tree"
    CAPABILITIES = ToolCapabilities()

    CACHEABLE = True
    INVALIDATES_TOOL_CACHE = False

    # ------------------------------------------------------------------
    # Semantic repo-map definition
    #
    # None of the major coding harnesses exposes an equivalent callable
    # tool. Aider provides the closest semantics, but injects its repo map
    # into model context instead of exposing it as a function.
    # ------------------------------------------------------------------

    REPO_MAP_DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="tree",
            description=(
                "Show a semantic map of the repository containing important "
                "files, definitions, signatures, and relevant code locations. "
                "This is not a plain directory listing. The map is ranked to "
                "fit a token budget and is useful for understanding repository "
                "structure before reading implementations. Use read when you "
                "need exact source code."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "Project-relative subtree or @tmp path to map. "
                                "Defaults to the entire project."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="focus",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Identifiers, filenames, or project-relative "
                                "path fragments whose definitions and related "
                                "code should receive higher ranking."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="max_tokens",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum semantic repository-map size in model "
                                f"tokens. Defaults to {DEFAULT_MAP_TOKENS} and "
                                f"cannot exceed {MAX_MAP_TOKENS}."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    # Keep an explicit native definition name for consistency with the
    # other tools, even though all model families currently share it.
    CITRA_DEFINITION = REPO_MAP_DEFINITION

    # ------------------------------------------------------------------
    # Model-family profiles
    #
    # These intentionally share the same schema. There is no truthful
    # Claude/Gemini/Qwen/Kimi/GLM callable-tool schema to imitate here.
    # ------------------------------------------------------------------

    CLAUDE_CODE_DEFINITION = REPO_MAP_DEFINITION
    GEMINI_CLI_DEFINITION = REPO_MAP_DEFINITION
    QWEN_CODE_DEFINITION = REPO_MAP_DEFINITION
    KIMI_CODE_DEFINITION = REPO_MAP_DEFINITION
    ZCODE_DEFINITION = REPO_MAP_DEFINITION

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        """Handle definitions for context."""
        del context

        return (
            ToolDefinition(
                definition=cls.CLAUDE_CODE_DEFINITION,
                model_family_matchers=(
                    "claude",
                ),
            ),
            ToolDefinition(
                definition=cls.GEMINI_CLI_DEFINITION,
                model_family_matchers=(
                    "gemini",
                ),
            ),
            ToolDefinition(
                definition=cls.QWEN_CODE_DEFINITION,
                model_family_matchers=(
                    "qwen",
                ),
            ),
            ToolDefinition(
                definition=cls.KIMI_CODE_DEFINITION,
                model_family_matchers=(
                    "kimi",
                    "moonshot",
                ),
            ),
            ToolDefinition(
                definition=cls.ZCODE_DEFINITION,
                model_family_matchers=(
                    "glm",
                ),
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
        super().__init__(
            context=context,
        )

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Execute the execute operation."""
        if (
            not hasattr(self.context, "repo_map")
            or not hasattr(self.context, "config")
        ):
            # Compatibility for lightweight embedded contexts.
            # Production ExecutionContext uses the semantic repo map.
            return self.context.filesystem.execute(
                TreeInput.parse(arguments)
            ).to_budgeted(model_id=self.context.model_config().id,token_count=4_000)

        model_id = self.context.config.model().id

        focus = arguments.get(
            "focus",
        ) or ()

        if not isinstance(
            focus,
            list,
        ):
            focus = ()

        return self.context.repo_map.render(
            model_id=model_id,
            path=arguments.get(
                "path",
                ".",
            ),
            focus=focus,
            max_tokens=arguments.get(
                "max_tokens",
                DEFAULT_MAP_TOKENS,
            ),
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Handle format call log."""
        parts = [
            f"path={arguments.get('path', '.')}",
        ]

        focus = arguments.get(
            "focus",
        )

        if focus:
            parts.append(
                f"focus={len(focus)}"
            )

        max_tokens = arguments.get(
            "max_tokens",
        )

        if max_tokens is not None:
            parts.append(
                f"max_tokens={max_tokens}"
            )

        return " | ".join(
            parts
        )

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        """Handle format result log."""
        text = str(result)

        if not text:
            return "empty map"

        lines = text.splitlines()

        files = sum(
            1
            for line in lines
            if (
                line
                and not line.startswith(
                    (
                        " ",
                        "\t",
                    )
                )
                and line.endswith(":")
            )
        )

        return (
            f"{files} file(s) | "
            f"{len(lines)} lines | "
            f"{len(text)} chars"
        )
