from citra.sandbox.filesystem_ops import TreeInput
from typing import Any, override

from ...context import ExecutionContext
from ...sandbox.filesystem_ops.tree import (
    DEFAULT_TREE_DEPTH,
    MAX_TREE_DEPTH,
)
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
    Show a structural view of source code or directories.

    Semantic maps are provided by RepoMap; directory listings are executed by
    the sandboxed tree filesystem operation.
    """

    TOOL_ID = "tree"
    CAPABILITIES = ToolCapabilities()

    CACHEABLE = True
    INVALIDATES_TOOL_CACHE = False

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="tree",
            description=(
                "Show a structural view of the workspace. By default, return "
                "an Aider-style semantic map containing important files, definitions, "
                "signatures, and code locations. Pass kind=\"directory\" to "
                "render a bounded directory tree. Directory-only options also "
                "select directory mode when kind is omitted."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="kind",
                        schema=JsonSchema.string(
                            description=(
                                "Rendering mode: \"aider\" (the default) for "
                                "a semantic source map, or \"directory\" for a "
                                "directory tree."
                            ),
                            enum=("aider", "directory"),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "Project-relative subtree, directory, or @tmp "
                                "path to inspect. Defaults to the entire project."
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
                    JsonProperty(
                        name="max_depth",
                        schema=JsonSchema.integer(
                            description=(
                                "Directory mode only. Maximum depth below the "
                                f"root. Defaults to {DEFAULT_TREE_DEPTH} and "
                                f"cannot exceed {MAX_TREE_DEPTH}."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="directories_only",
                        schema=JsonSchema.boolean(
                            description=(
                                "Directory mode only. Omit files when true."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="skip",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Directory mode only. Basenames, relative paths, "
                                "or glob patterns to skip."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="hidden",
                        schema=JsonSchema.boolean(
                            description=(
                                "Directory mode only. Include hidden entries "
                                "when true; explicit skip rules still apply."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="use_default_skips",
                        schema=JsonSchema.boolean(
                            description=(
                                "Directory mode only. Apply common VCS, cache, "
                                "dependency, and build-directory skips. Defaults "
                                "to true."
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
        if self._kind(arguments) == "directory":
            return self.context.filesystem.execute(
                TreeInput.parse(dict(arguments))
            ).to_budgeted(
                model_id=self.context.model_config().id,
                token_count=4_000,
            )

        if (
            not hasattr(self.context, "repo_map")
            or not hasattr(self.context, "config")
        ):
            return self.context.filesystem.execute(
                TreeInput.parse(dict(arguments))
            ).to_budgeted(
                model_id=self.context.model_config().id,
                token_count=4_000,
            )

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
        if self._kind(arguments) == "directory":
            parts = [
                "kind=directory",
                f"path={arguments.get('path', '.')}",
            ]
            if "max_depth" in arguments:
                parts.append(f"depth={arguments['max_depth']}")
            if arguments.get("directories_only"):
                parts.append("dirs-only=true")
            if arguments.get("skip"):
                parts.append(f"skip={len(arguments['skip'])}")
            if arguments.get("hidden"):
                parts.append("hidden=true")
            return " | ".join(parts)

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

    @staticmethod
    def _kind(arguments: dict[str, Any]) -> str:
        """Resolve the explicit mode or infer it from directory options."""
        kind = arguments.get("kind")
        if kind is not None:
            return str(kind)
        directory_fields = {
            "max_depth",
            "directories_only",
            "skip",
            "hidden",
            "use_default_skips",
        }
        return "directory" if directory_fields.intersection(arguments) else "aider"

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
