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
from ..tool import Tool, ToolDefinition


class Tree(Tool):
    """
    Show a structural view of the repository.

    The same tool produces two different renderings, selected by the
    ``kind`` argument:

    * ``kind="aider"`` (default) — an Aider-style semantic repo map
      containing important definitions, signatures, and code locations,
      ranked to fit a token budget. This is not a plain directory listing.
    * ``kind="directory"`` — a sandboxed filesystem tree rooted at
      ``path`` with configurable depth, hidden-file visibility, and skip
      patterns. Execution is routed through the sandbox so only
      allowed paths are exposed.
    """

    TOOL_ID = "tree"
    CAPABILITIES = ToolCapabilities()

    CACHEABLE = True
    INVALIDATES_TOOL_CACHE = False

    # ------------------------------------------------------------------
    # Merged schema
    #
    # A single object schema carries every argument from both kinds.
    # The ``kind`` selector switches execution between the Aider repo
    # map (``ExecutionContext.repo_map.render``) and the sandboxed
    # directory tree (``ExecutionContext.filesystem.execute`` /
    # ``TreeInput``). ``additional_properties=False`` plus the ``kind``
    # enum together reject unknown values.
    # ------------------------------------------------------------------

    REPO_MAP_DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="tree",
            description=(
                "Show a structural view of the repository. "
                "By default (kind=\"aider\") this returns an Aider-style "
                "semantic repo map containing important files, "
                "definitions, signatures, and relevant code locations, "
                "ranked to fit a token budget; it is not a plain directory "
                "listing. Pass kind=\"directory\" to render a sandboxed "
                "filesystem tree rooted at path, with configurable depth, "
                "hidden-file visibility, and skip patterns. Use read when "
                "you need exact source code."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="kind",
                        schema=JsonSchema.string(
                            description=(
                                "Selector for the rendering mode. "
                                "\"aider\" (default) returns a semantic "
                                "repo map; \"directory\" returns a "
                                "sandboxed filesystem tree."
                            ),
                            enum=(
                                "aider",
                                "directory",
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "Aider mode: project-relative subtree or "
                                "@tmp path to map (defaults to the entire "
                                "project). Directory mode: root path of the "
                                "filesystem tree to render."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="focus",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Aider mode only. Identifiers, filenames, "
                                "or project-relative path fragments whose "
                                "definitions and related code should "
                                "receive higher ranking. Ignored in "
                                "directory mode."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="max_tokens",
                        schema=JsonSchema.integer(
                            description=(
                                "Aider mode only. Maximum semantic "
                                "repository-map size in model tokens. "
                                f"Defaults to {DEFAULT_MAP_TOKENS} and "
                                f"cannot exceed {MAX_MAP_TOKENS}. Ignored "
                                "in directory mode."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="max_depth",
                        schema=JsonSchema.integer(
                            description=(
                                f"Directory mode only. Maximum tree depth "
                                f"(0..{MAX_TREE_DEPTH}). Defaults to "
                                f"{DEFAULT_TREE_DEPTH}. Ignored in Aider "
                                "mode."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="directories_only",
                        schema=JsonSchema.boolean(
                            description=(
                                "Directory mode only. When true, omit "
                                "file entries and render directories "
                                "only. Ignored in Aider mode."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="hidden",
                        schema=JsonSchema.boolean(
                            description=(
                                "Directory mode only. When true, include "
                                "dot-files and dot-directories that would "
                                "otherwise be skipped. Ignored in Aider "
                                "mode."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="skip",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Directory mode only. Additional glob or "
                                "name patterns to skip on top of the "
                                "default skip set. Ignored in Aider mode."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="use_default_skips",
                        schema=JsonSchema.boolean(
                            description=(
                                "Directory mode only. When false, the "
                                "default skip set (e.g. .git, node_modules) "
                                "is not applied. Ignored in Aider mode."
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
        """Execute the execute operation.

        Dispatch on ``kind``:

        * ``kind="aider"`` (default) — render the Aider-style semantic
          repo map. Byte-equivalent to the legacy behavior: it reads
          ``path``, ``focus``, and ``max_tokens`` from ``arguments``
          with the same defaults and calls
          ``ExecutionContext.repo_map.render``.
        * ``kind="directory"`` — construct a ``TreeInput`` from the
          arguments and delegate to ``ExecutionContext.filesystem``
          so the existing ``ScopedFilesystem`` enforces the sandbox
          boundary. The tool never imports
          ``citra.utils.directory_tree.render_tree``.

        If ``kind`` is omitted, the presence of any directory-only
        argument (``max_depth``, ``directories_only``, ``hidden``,
        ``skip``, ``use_default_skips``) auto-selects the directory
        branch; otherwise the Aider branch is the default. This keeps
        existing directory-style call sites working without an explicit
        ``kind`` while preserving the legacy Aider surface.

        A lightweight embedded context that lacks ``repo_map`` and
        ``config`` continues to route through ``TreeInput.parse`` and
        the sandbox, so the directory branch remains valid there.
        """
        kind = arguments.get(
            "kind",
        )

        if kind is None:
            kind = (
                "directory"
                if self._has_directory_only_fields(arguments)
                else "aider"
            )

        if kind == "directory":
            # Sandboxed directory tree; ScopedFilesystem enforces
            # path boundaries.
            return self.context.filesystem.execute(
                TreeInput.parse(arguments)
            ).to_budgeted(
                model_id=self.context.model_config().id,
                token_count=4_000,
            )

        # ``kind`` is either ``"aider"`` (explicit) or an unknown value
        # that slipped past schema validation; in both cases the Aider
        # branch is the safe default.
        if (
            not hasattr(self.context, "repo_map")
            or not hasattr(self.context, "config")
        ):
            # Compatibility for lightweight embedded contexts without
            # repo_map / config.
            return self.context.filesystem.execute(
                TreeInput.parse(arguments)
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
        """Handle format call log.

        The log surface depends on ``kind``:

        * ``kind="aider"`` (default and the legacy surface) — ``path=...``
          with optional ``focus=N`` and optional ``max_tokens=...``.
        * ``kind="directory"`` — ``kind=directory | path=...`` plus
          optional ``depth=...``, ``dirs-only=true``, ``skip=N``, and
          ``hidden=true`` segments, matching the
          ``tests/test_tool_logging.py::TestTree`` expectations.

        If ``kind`` is not specified explicitly, the presence of any
        directory-only argument (``max_depth``, ``directories_only``,
        ``hidden``, ``skip``, ``use_default_skips``) auto-selects the
        directory mode; that keeps existing directory-style call sites
        working without an explicit ``kind``.
        """
        kind = arguments.get(
            "kind",
        )

        if kind is None:
            kind = "directory" if self._has_directory_only_fields(arguments) else "aider"

        if kind == "directory":
            return self._format_directory_call_log(arguments)

        return self._format_aider_call_log(arguments)

    @staticmethod
    def _has_directory_only_fields(
        arguments: dict[str, Any],
    ) -> bool:
        """Return True if any directory-only argument is present."""
        for field in (
            "max_depth",
            "directories_only",
            "hidden",
            "skip",
            "use_default_skips",
        ):
            if field in arguments:
                return True
        return False

    def _format_aider_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Render the legacy Aider-style call log surface."""
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

    def _format_directory_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Render the directory-tree call log surface."""
        parts = [
            "kind=directory",
            f"path={arguments.get('path', '.')}",
        ]

        max_depth = arguments.get(
            "max_depth",
        )

        if max_depth is not None:
            parts.append(
                f"depth={max_depth}"
            )

        if arguments.get(
            "directories_only",
        ):
            parts.append(
                "dirs-only=true"
            )

        skip = arguments.get(
            "skip",
        )

        if skip:
            parts.append(
                f"skip={len(skip)}"
            )

        if arguments.get(
            "hidden",
        ):
            parts.append(
                "hidden=true"
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