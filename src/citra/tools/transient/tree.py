from typing import Any, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ...utils.repo_map import DEFAULT_MAP_TOKENS, MAX_MAP_TOKENS
from ..tool import Tool


class Tree(Tool):
    """Show an Aider-style structural map of the repository."""

    CACHEABLE = True
    INVALIDATES_TOOL_CACHE = False
    MAX_OUTPUT_TOKENS = MAX_MAP_TOKENS

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="tree",
            description=(
                "Discover repository structure, definitions, signatures, and relevant code locations. "
                "Use read when you need the implementation."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "Project-relative subtree or @tmp path to map. Defaults "
                                "to the entire project. '@source' is accepted as a "
                                "project-relative source alias."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="focus",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Optional identifiers, filenames, or project-relative path "
                                "fragments to boost in the repository ranking."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="max_tokens",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum repository-map size in model tokens. Defaults to "
                                f"{DEFAULT_MAP_TOKENS} and cannot exceed {MAX_MAP_TOKENS}."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        super().__init__(
            context=context,
            definition=self.DEFINITION,
        )

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        model_id = self.context.config.model().id
        focus = arguments.get("focus") or ()

        if not isinstance(focus, list):
            focus = ()

        return self.context.repo_map.render(
            model_id=model_id,
            path=arguments.get("path", "."),
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
        parts = [f"path={arguments.get('path', '.')}"]
        focus = arguments.get("focus")
        if focus:
            parts.append(f"focus={len(focus)}")
        max_tokens = arguments.get("max_tokens")
        if max_tokens is not None:
            parts.append(f"max_tokens={max_tokens}")
        return " | ".join(parts)

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        text = str(result)
        if not text:
            return "empty map"
        files = sum(
            1
            for line in text.splitlines()
            if line and not line.startswith((" ", "\t")) and line.endswith(":")
        )
        return f"{files} file(s) | {len(text.splitlines())} lines | {len(text)} chars"
