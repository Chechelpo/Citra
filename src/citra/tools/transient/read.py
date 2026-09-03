"""Model-facing read tool backed by the sandbox filesystem worker."""

from citra.sandbox.filesystem_ops import ReadRawInput
from citra.sandbox.filesystem_ops import ReadOutput
from citra.sandbox.filesystem_ops import ReadInput
from citra.utils.lsp import LspError
from citra.utils.lsp.errors import LspDiagnosticsTimeout
from citra.utils.lsp import LspUnavailable
from citra.utils.lsp.diagnostics import format_diagnostics
from citra.utils.lsp import detect_language
from pathlib import Path
from typing import Any, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..tool import Tool, ToolDefinition

_TRUNCATE_LENGTH = 120


class Read(Tool):
    """Represent Read."""
    TOOL_ID = "read"

    CACHEABLE = True
    INVALIDATES_TOOL_CACHE = False

    """Read literal paths or globs without host-process filesystem I/O."""

    MAX_REQUESTS = 20
    MAX_FILES_PER_CALL = 20

    READ_REQUEST_SCHEMA = JsonSchema.object(
        properties=(
            JsonProperty(
                name="path",
                schema=JsonSchema.string(
                    description=(
                        "File path or glob pattern. Examples: 'main.py', "
                        "'src/**/*.py', or 'pyproject.toml'."
                    ),
                ),
            ),
            JsonProperty(
                name="offset",
                schema=JsonSchema.integer(
                    description="Zero-based line offset. Defaults to 0.",
                ),
                required=False,
            ),
            JsonProperty(
                name="limit",
                schema=JsonSchema.integer(
                    description="Maximum number of lines to return.",
                ),
                required=False,
            ),
            JsonProperty(
                name="diagnose",
                schema=JsonSchema.boolean(
                    description="Whether to also run diagnosis (lsp) on the read file"
                ),
                required=False,
            ),
        ),
        additional_properties=False,
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="read",
            description=(
                "Read exact file contents and implementations. "
                "Use tree for discovery and structure."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description="Single file path or glob pattern.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="offset",
                        schema=JsonSchema.integer(
                            description="Zero-based offset for top-level path.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="limit",
                        schema=JsonSchema.integer(
                            description="Line limit for top-level path.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="requests",
                        schema=JsonSchema.array(
                            READ_REQUEST_SCHEMA,
                            description="Batch of independent read requests.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="diagnose",
                        schema=JsonSchema.boolean(
                            description=(
                                "Also run language-server diagnostics on the file. "
                                "Only supported for exact file paths."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
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
            ToolDefinition(
                definition=cls.DEFINITION,
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
        result: str = self.context.filesystem.execute(
            ReadInput.parse(arguments)
        ).to_budgeted(model_id=self.context.model_config().id, token_count=4_000)

        diagnostic_results = self._run_diagnostics(
            arguments
        )

        if not diagnostic_results:
            return result

        return (
            result
            + "\n\nDIAGNOSTICS:\n"
            + "\n\n".join(diagnostic_results)
        )

    def _run_diagnostics(
        self,
        arguments: dict[str, Any],
    ) -> list[str]:
        """Handle run diagnostics."""
        results: list[str] = []

        path = arguments.get("path")

        if (
            path is not None
            and arguments.get("diagnose", False)
        ):
            results.append(
                self._diagnose_path(
                    str(path)
                )
            )

        requests = arguments.get("requests") or ()

        for request in requests:
            if not request.get("diagnose", False):
                continue

            request_path = request.get("path")

            if not request_path:
                continue

            results.append(
                self._diagnose_path(
                    str(request_path)
                )
            )

        return results

    def _diagnose_path(
        self,
        path: str,
    ) -> str:
        """Handle diagnose path."""
        display_path = path

        if self._looks_like_glob(path):
            return (
                f"{display_path}: diagnostics unavailable for glob patterns"
            )

        resolved = self.context.workspace.resolve_path(
            path
        )

        diagnostic = self._diagnose_file(
            resolved
        )

        return (
            f"{display_path}:\n"
            f"{diagnostic or 'ok'}"
        )

    def _diagnose_file(
        self,
        path: Path,
    ) -> str | None:
        """Handle diagnose file."""
        manager = self.context.lsp_manager

        if manager is None:
            return "LSP diagnostics unavailable"

        language = detect_language(
            path
        )

        if language is None:
            return (
                "unsupported: no language server is configured for "
                f"{path.suffix or 'this file'}"
            )

        read_result = self.context.filesystem.execute(
            ReadRawInput.parse(
                {
                    "path": str(path),
                }
            )
        )

        try:
            rendered = format_diagnostics(
                manager.diagnostics(
                    path,
                    read_result.content,
                ),
                path=path,
                display_path=self.context.workspace.display_path,
            )
        except LspDiagnosticsTimeout:
            return "LSP diagnostics timed out"
        except LspUnavailable:
            return "LSP unavailable"
        except LspError as error:
            return f"LSP error: {error}"

        return rendered or None

    @staticmethod
    def _looks_like_glob(
        path: str,
    ) -> bool:
        """Handle looks like glob."""
        return any(
            character in path
            for character in (
                "*",
                "?",
                "[",
            )
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Handle format call log."""
        path = arguments.get("path")
        requests = arguments.get("requests")

        if path is not None:
            parts = [f"path={self._truncate(path)}"]

            offset = arguments.get("offset")
            if offset:
                parts.append(f"offset={offset}")

            limit = arguments.get("limit")
            if limit is not None:
                parts.append(f"limit={limit}")

            return " | ".join(parts)

        if requests:
            return f"batch={len(requests)} request(s)"

        return "no path"

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        """Handle format result log."""
        text = str(result)

        if not text:
            return "empty result"

        file_count = text.count("===== ")
        lines = text.splitlines()

        parts = [
            f"{len(lines)} lines",
            f"{len(text)} chars",
        ]

        if file_count:
            parts.insert(
                0,
                f"{file_count} file(s)",
            )

        return " | ".join(parts)

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
