from typing import Any, override

from ...context import ExecutionContext
from ...utils.directory_tree import (
    DEFAULT_LIMIT,
    DEFAULT_MAX_DEPTH,
    MAX_LIMIT,
    MAX_MAX_DEPTH,
    render_tree,
)
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..tool import Tool


class Tree(Tool):
    """
    Show the structure of a directory.

    Traversal and rendering are implemented by utils.directory_tree.
    """

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="tree",
            description=(
                "Show the directory structure of the workspace or another "
                "allowed path. Output is deterministic, directories are shown "
                "before files, and traversal is bounded by max_depth and "
                "limit. Use directories_only to inspect only folder structure. "
                "Use skip to exclude directory names, relative paths, or glob "
                "patterns such as 'node_modules', 'src/generated', "
                "'**/__pycache__', or '.git*'. Common generated, dependency, "
                "cache, and version-control directories are skipped by default."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "Directory to inspect. Relative paths resolve "
                                "from the active workspace. Filesystem aliases "
                                "such as @tmp are supported. Defaults to '.'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="max_depth",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum directory depth to descend below the "
                                "requested root. 0 shows only the root. "
                                f"Defaults to {DEFAULT_MAX_DEPTH} and cannot "
                                f"exceed {MAX_MAX_DEPTH}."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="directories_only",
                        schema=JsonSchema.boolean(
                            description=(
                                "If true, show directories only and omit "
                                "files. Defaults to false."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="skip",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Additional entries to skip. Each value may "
                                "be an exact basename, a path relative to the "
                                "tree root, or a glob pattern. Examples: "
                                "'node_modules', 'src/generated', "
                                "'**/__pycache__', '*.egg-info'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="hidden",
                        schema=JsonSchema.boolean(
                            description=(
                                "Show dotfiles and dot-directories. Defaults "
                                "to false. Explicit skip rules still apply."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="limit",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum number of entries to emit, excluding "
                                f"the root line. Defaults to {DEFAULT_LIMIT} "
                                f"and cannot exceed {MAX_LIMIT}."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="use_default_skips",
                        schema=JsonSchema.boolean(
                            description=(
                                "Apply built-in skips for common dependency, "
                                "VCS, cache, and build directories. Defaults "
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
        super().__init__(
            context=context,
            definition=self.DEFINITION,
        )

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        return render_tree(
            self.context.workspace,
            path=arguments.get(
                "path",
                ".",
            ),
            max_depth=arguments.get(
                "max_depth",
                DEFAULT_MAX_DEPTH,
            ),
            directories_only=arguments.get(
                "directories_only",
                False,
            ),
            skip=arguments.get(
                "skip",
                (),
            ),
            hidden=arguments.get(
                "hidden",
                False,
            ),
            limit=arguments.get(
                "limit",
                DEFAULT_LIMIT,
            ),
            use_default_skips=arguments.get(
                "use_default_skips",
                True,
            ),
        )