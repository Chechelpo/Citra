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
from ..tool import Tool


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


    # Keep an explicit native definition name for consistency with the
    # other tools, even though all model families currently share it.
    CITRA_DEFINITION = ChatCompletionTool(
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

    # ------------------------------------------------------------------
    # Model-family profiles
    #
    # These intentionally share the same schema. There is no truthful
    # Claude/Gemini/Qwen/Kimi/GLM callable-tool schema to imitate here.
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
                TreeInput.parse(dict(arguments))
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
