from difflib import unified_diff
from typing import Any, override

from citra.sandbox.filesystem_ops import EditInput

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..capabilities import ToolCapabilities
from ..tool import Tool
from ._post_edit import post_edit_result

_TRUNCATE_LENGTH = 120


def _edit_definition(
    *,
    name: str,
    path_name: str,
    old_name: str,
    new_name: str,
    all_name: str | None = None,
    instruction_name: str | None = None,
    include_line: bool = False,
    include_diagnostics: bool = False,
    description: str,
) -> ChatCompletionTool:
    """Handle edit definition."""
    properties: list[JsonProperty] = [
        JsonProperty(
            name=path_name,
            schema=JsonSchema.string(
                description="Path of the file to modify.",
            ),
        ),
    ]

    if instruction_name is not None:
        properties.append(
            JsonProperty(
                name=instruction_name,
                schema=JsonSchema.string(
                    description=(
                        "A clear semantic description of the intended change."
                    ),
                ),
            )
        )

    if include_line:
        properties.append(
            JsonProperty(
                name="line",
                schema=JsonSchema.integer(
                    description=(
                        "1-based line number before which the new text "
                        "should be inserted. Cannot be used with old text."
                    ),
                ),
                required=False,
            )
        )

    properties.extend(
        (
            JsonProperty(
                name=old_name,
                schema=JsonSchema.string(
                    description=(
                        "Exact existing text to replace. "
                        "Whitespace and indentation must match."
                    ),
                ),
                required=not include_line,
            ),
            JsonProperty(
                name=new_name,
                schema=JsonSchema.string(
                    description=(
                        "Replacement text. May be empty to delete "
                        "the matched text."
                    ),
                ),
            ),
        )
    )

    if all_name is not None:
        properties.append(
            JsonProperty(
                name=all_name,
                schema=JsonSchema.boolean(
                    description=(
                        "Replace all occurrences instead of requiring "
                        "the old text to identify a unique match."
                    ),
                ),
                required=False,
            )
        )

    if include_diagnostics:
        properties.append(
            JsonProperty(
                name="diagnostics",
                schema=JsonSchema.boolean(
                    description=(
                        "Deprecated compatibility flag. LSP diagnostics "
                        "and configured lint checks run automatically "
                        "after every successful edit."
                    ),
                ),
                required=False,
            )
        )

    return ChatCompletionTool(
        function=FunctionDefinition(
            name=name,
            description=description,
            parameters=JsonSchema.object(
                properties=tuple(properties),
                additional_properties=False,
            ),
        ),
    )


class Edit(Tool):
    """
    Performs exact text replacement in an existing file.

    Unless all=true, the text being replaced must occur exactly once.
    """

    TOOL_ID = "edit"
    CAPABILITIES = ToolCapabilities()

    # ------------------------------------------------------------------
    # Citra-native fallback
    # ------------------------------------------------------------------

    CITRA_DEFINITION = _edit_definition(
        name="edit",
        path_name="path",
        old_name="old",
        new_name="new",
        all_name="all",
        include_line=True,
        include_diagnostics=True,
        description=(
            "Replace an exact text fragment in an existing file. "
            "By default the old text must occur exactly once. "
            "Use all=true to replace every occurrence. "
            "Text can also be inserted at a specific line."
        ),
    )

    # ------------------------------------------------------------------
    # Claude Code
    #
    # Edit(
    #     file_path,
    #     old_string,
    #     new_string,
    #     replace_all?,
    # )
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Gemini CLI
    #
    # replace(
    #     file_path,
    #     instruction,
    #     old_string,
    #     new_string,
    #     allow_multiple?,
    # )
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Qwen Code
    #
    # edit(
    #     file_path,
    #     old_string,
    #     new_string,
    #     replace_all?,
    # )
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Kimi Code
    #
    # Edit(
    #     path,
    #     old_string,
    #     new_string,
    #     replace_all?,
    # )
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # ZCode / GLM
    #
    # ZCode exposes separate Edit/Write tools. Its public material does
    # not currently document the complete Edit schema as clearly as
    # Claude/Qwen/Kimi do. The Claude-compatible snake_case shape is the
    # best-supported mapping for GLM-family models at present.
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Reference harness definitions.
    #
    # These are genuine schemas, but model_id alone cannot tell us that
    # the model is running under one of these harnesses.
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Model-facing definition resolution
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

    # ------------------------------------------------------------------
    # Model-facing -> Citra-internal argument normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _first_argument(
        arguments: dict[str, Any],
        *names: str,
    ) -> Any:
        """Handle first argument."""
        for name in names:
            if name in arguments:
                return arguments[name]

        return None

    def _normalize_arguments(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle normalize arguments."""
        path = self._first_argument(
            arguments,
            "path",
            "file_path",
            "filePath",
        )

        new = self._first_argument(
            arguments,
            "new",
            "new_string",
            "newString",
        )

        old = self._first_argument(
            arguments,
            "old",
            "old_string",
            "oldString",
        )

        replace_all = self._first_argument(
            arguments,
            "all",
            "replace_all",
            "replaceAll",
            "allow_multiple",
        )

        normalized: dict[str, Any] = {
            "path": path,
            "new": new,
        }

        if old is not None:
            normalized["old"] = old

        if "line" in arguments:
            normalized["line"] = arguments["line"]

        if replace_all is not None:
            normalized["all"] = bool(
                replace_all
            )

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
        filesystem_arguments = self._normalize_arguments(
            arguments
        )

        result: str = self.context.filesystem.execute(
            EditInput.parse(filesystem_arguments)
        ).to_budgeted(model_id=self.context.model_config().id, token_count=4_000)

        if result != "ok":
            return result

        return post_edit_result(
            self.context,
            filesystem_arguments["path"],
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Render the intended edit as a contextual whole-file diff."""
        normalized = self._normalize_arguments(
            arguments
        )

        path = normalized.get(
            "path",
            "",
        )

        old = normalized.get("old")
        new = normalized.get("new")
        line = normalized.get("line")
        replace_all = normalized.get(
            "all",
            False,
        )

        old_text = "" if old is None else str(old)
        new_text = "" if new is None else str(new)
        before: str | None = None
        try:
            resolved = self.context.workspace.resolve_path(path)
            before = resolved.read_text(encoding="utf-8")
        except (AttributeError, OSError, TypeError, UnicodeError, ValueError):
            # Lightweight test contexts and inaccessible paths still get the
            # replacement-fragment diff used by legacy logs.
            pass

        if before is not None:
            if line is not None:
                lines = before.splitlines(keepends=True)
                line_number = int(line)
                if 1 <= line_number <= len(lines) + 1:
                    lines.insert(line_number - 1, new_text)
                    old_text = before
                    new_text = "".join(lines)
            elif old is not None:
                count = before.count(old_text)
                if count and (count == 1 or replace_all):
                    new_content = before.replace(
                        old_text,
                        new_text,
                        -1 if replace_all else 1,
                    )
                    old_text = before
                    new_text = new_content

        diff = "\n".join(
            unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )
        metadata: list[str] = []
        if line is not None:
            metadata.append(f"insert@line={line}")
        if replace_all:
            metadata.append("all=true")
        if metadata:
            diff += "\n" + " | ".join(metadata)
        return diff.rstrip()

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        """Handle format result log."""
        return self._truncate(
            str(result)
        )

    @staticmethod
    def _truncate(
        value: str,
    ) -> str:
        """Handle truncate."""
        if len(value) <= _TRUNCATE_LENGTH:
            return value

        return (
            value[:_TRUNCATE_LENGTH]
            + "..."
        )
