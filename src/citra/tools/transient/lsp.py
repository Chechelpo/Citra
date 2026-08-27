"""Model-facing semantic navigation backed by persistent language servers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..lsp.diagnostics import format_diagnostics
from ..lsp.errors import (
    LspDiagnosticsTimeout,
    LspError,
    LspUnavailable,
    LspUnsupportedCapability,
)
from ..lsp.language import detect_language
from ..lsp.manager import LspManager
from ..lsp.positions import SourcePosition
from ..lsp.protocol import uri_to_path
from ..tool import Tool, ToolDefinition


class Lsp(Tool):
    TOOL_ID = "lsp"

    INVALIDATES_TOOL_CACHE = False

    """
    Expose high-value language intelligence without raw JSON-RPC.
    """

    # Citra-internal action names.
    POSITION_ACTIONS = frozenset(
        {
            "hover",
            "definition",
            "go_to_definition",
            "declaration",
            "type_definition",
            "implementation",
            "references",
        }
    )

    ACTIONS = (
        "status",
        "diagnostics",
        "hover",
        "definition",
        "go_to_definition",
        "declaration",
        "type_definition",
        "implementation",
        "references",
        "document_symbols",
    )

    # Model-facing names for the Claude/Qwen/OpenCode-style LSP tool.
    #
    # The operations not normally present in a given native harness
    # (status, diagnostics, declaration, typeDefinition) are Citra
    # extensions backed by real execution semantics.
    NATIVE_ACTIONS = (
        "status",
        "diagnostics",
        "hover",
        "goToDefinition",
        "declaration",
        "typeDefinition",
        "goToImplementation",
        "findReferences",
        "documentSymbol",
    )

    NATIVE_ACTION_MAP = {
        "status": "status",
        "diagnostics": "diagnostics",
        "hover": "hover",
        "goToDefinition": "go_to_definition",
        "declaration": "declaration",
        "typeDefinition": "type_definition",
        "goToImplementation": "implementation",
        "findReferences": "references",
        "documentSymbol": "document_symbols",
    }

    # ------------------------------------------------------------------
    # Citra-native fallback
    #
    # lsp(
    #     action,
    #     path?,
    #     line?,
    #     character?,
    #     include_declaration?,
    # )
    # ------------------------------------------------------------------

    CITRA_DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="lsp",
            description=(
                "Use a persistent sandboxed language server for semantic "
                "code intelligence. Supports diagnostics, hover, document "
                "symbols, references, definitions, declarations, type "
                "definitions, and implementations. line and character are "
                "1-based. Use status to inspect available language servers."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Semantic operation.",
                            enum=ACTIONS,
                        ),
                    ),
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "Source path in any allowed Citra filesystem "
                                "root. Required except for status."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="line",
                        schema=JsonSchema.integer(
                            description=(
                                "1-based source line for position operations."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="character",
                        schema=JsonSchema.integer(
                            description=(
                                "1-based character/column for position operations."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="include_declaration",
                        schema=JsonSchema.boolean(
                            description=(
                                "Include the declaration itself in reference "
                                "results. Defaults to true."
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
    # Claude Code
    #
    # LSP(
    #     operation,
    #     filePath?,
    #     line?,
    #     character?,
    #     includeDeclaration?,
    # )
    # ------------------------------------------------------------------

    CLAUDE_CODE_DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="LSP",
            description=(
                "Interact with a language server for semantic code "
                "intelligence including definitions, references, hover "
                "information, symbols, diagnostics, and implementations. "
                "Line and character positions are 1-based."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="operation",
                        schema=JsonSchema.string(
                            description="LSP operation to perform.",
                            enum=NATIVE_ACTIONS,
                        ),
                    ),
                    JsonProperty(
                        name="filePath",
                        schema=JsonSchema.string(
                            description=(
                                "Path of the source file. Required for "
                                "file-based operations."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="line",
                        schema=JsonSchema.integer(
                            description="1-based source line.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="character",
                        schema=JsonSchema.integer(
                            description="1-based character offset.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="includeDeclaration",
                        schema=JsonSchema.boolean(
                            description=(
                                "Whether reference results should include "
                                "the declaration itself."
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
    # Qwen Code
    #
    # lsp(
    #     operation,
    #     filePath?,
    #     line?,
    #     character?,
    #     includeDeclaration?,
    # )
    #
    # Qwen has an even larger native LSP surface, including workspace
    # symbols, workspace diagnostics, code actions, and call hierarchy.
    # Only advertise operations Citra can actually execute here.
    # ------------------------------------------------------------------

    QWEN_CODE_DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="lsp",
            description=(
                "Use Language Server Protocol code intelligence for precise "
                "semantic navigation, references, diagnostics, symbols, "
                "hover information, and implementations."
            ),
            parameters=CLAUDE_CODE_DEFINITION.function.parameters,
        ),
    )

    # Current OpenCode's experimental LSP tool has essentially the same
    # core schema as the Qwen profile:
    #
    #   lsp(operation, filePath, line, character, ...)
    #
    # Keep this for future harness-aware resolution.
    OPENCODE_DEFINITION = QWEN_CODE_DEFINITION

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        del context

        return (
            ToolDefinition(
                definition=cls.CLAUDE_CODE_DEFINITION,
                model_family_matchers=(
                    "claude",
                ),
            ),
            ToolDefinition(
                definition=cls.QWEN_CODE_DEFINITION,
                model_family_matchers=(
                    "qwen",
                ),
            ),

            # Gemini, Kimi, GPT/Codex, and GLM/ZCode currently have no
            # sufficiently established native model-facing LSP tool to
            # imitate, so they use Citra's own schema.
            ToolDefinition(
                definition=cls.CITRA_DEFINITION,
            ),
        )

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        super().__init__(
            context=context,
        )

    # ------------------------------------------------------------------
    # Cache policy
    # ------------------------------------------------------------------

    def is_cacheable(
        self,
        arguments: dict[str, Any],
    ) -> bool:
        action = arguments.get(
            "action",
            arguments.get("operation"),
        )

        return action != "status"

    # ------------------------------------------------------------------
    # Argument normalization
    # ------------------------------------------------------------------

    @classmethod
    def _normalize_arguments(
        cls,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        raw_action = arguments.get(
            "action",
            arguments.get("operation"),
        )

        if raw_action is None:
            raise ValueError(
                "LSP invocation contained no action or operation."
            )

        action = cls.NATIVE_ACTION_MAP.get(
            str(raw_action),
            str(raw_action),
        )

        normalized: dict[str, Any] = {
            "action": action,
        }

        path = arguments.get(
            "path",
            arguments.get("filePath"),
        )

        if path is not None:
            normalized["path"] = path

        if "line" in arguments:
            normalized["line"] = arguments["line"]

        if "character" in arguments:
            normalized["character"] = arguments["character"]

        include_declaration = arguments.get(
            "include_declaration",
            arguments.get("includeDeclaration"),
        )

        if include_declaration is not None:
            normalized["include_declaration"] = (
                include_declaration
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
        arguments = self._normalize_arguments(
            arguments
        )

        manager = self.context.lsp_manager

        if not isinstance(
            manager,
            LspManager,
        ):
            return (
                "unavailable: LSP services are disabled "
                "in this execution context"
            )

        action = arguments["action"]

        if action == "status":
            return json.dumps(
                manager.status(),
                indent=2,
                ensure_ascii=False,
            )

        path_raw = arguments.get(
            "path"
        )

        if not path_raw:
            raise ValueError(
                "'path' is required for this LSP action."
            )

        path = self.context.workspace.resolve_path(
            path_raw
        )

        language = detect_language(
            path
        )

        if language is None:
            return (
                "unsupported: no language server is configured for "
                f"{path.suffix or 'this file'}"
            )

        text = self.context.filesystem.execute(
            "read_raw",
            {
                "path": str(path),
            },
        )

        try:
            if action == "diagnostics":
                self._reject(
                    arguments,
                    "line",
                    "character",
                    "include_declaration",
                )

                rendered = format_diagnostics(
                    manager.diagnostics(
                        path,
                        text,
                    ),
                    path=path,
                    display_path=(
                        self.context.workspace.display_path
                    ),
                )

                return rendered or "none"

            handle = manager.client_for(
                path
            )

            client = handle.client

            uri = client.sync_document(
                path,
                text,
                handle.language,
            )

            if action == "document_symbols":
                self._reject(
                    arguments,
                    "line",
                    "character",
                    "include_declaration",
                )

                return self._format_symbols(
                    client.document_symbols(
                        uri
                    ),
                    path,
                )

            if action not in self.POSITION_ACTIONS:
                raise ValueError(
                    f"Unsupported LSP action: {action}"
                )

            position = self._position(
                arguments,
                text,
            )

            if action == "hover":
                self._reject(
                    arguments,
                    "include_declaration",
                )

                return self._format_hover(
                    client.hover(
                        uri,
                        position,
                    )
                )

            if action == "references":
                return self._format_locations(
                    client.references(
                        uri,
                        position,
                        include_declaration=arguments.get(
                            "include_declaration",
                            True,
                        ),
                    )
                )

            self._reject(
                arguments,
                "include_declaration",
            )

            definition_kind = (
                "definition"
                if action == "go_to_definition"
                else action
            )

            return self._format_locations(
                client.definitions(
                    definition_kind,
                    uri,
                    position,
                )
            )

        except LspUnavailable as error:
            return (
                f"unavailable: {error}"
            )

        except LspUnsupportedCapability as error:
            return (
                f"unsupported: {error}"
            )

        except LspDiagnosticsTimeout as error:
            return (
                f"unavailable: {error}"
            )

        except LspError as error:
            self.context.logger.exception(
                "LSP operation failed for %s",
                path_raw,
            )

            return (
                "unavailable: language server operation failed: "
                f"{error}"
            )

        except Exception as error:
            self.context.logger.exception(
                "Unexpected LSP failure for %s",
                path_raw,
            )

            return (
                "unavailable: language server operation failed: "
                f"{error}"
            )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        normalized = self._normalize_arguments(
            arguments
        )

        action = normalized.get(
            "action",
            "?",
        )

        path = normalized.get(
            "path"
        )

        parts = [
            f"action={action}",
        ]

        if path is not None:
            parts.append(
                f"path={path}"
            )

        line = normalized.get(
            "line"
        )

        character = normalized.get(
            "character"
        )

        if (
            line is not None
            and character is not None
        ):
            parts.append(
                f"pos={line}:{character}"
            )

        return " | ".join(
            parts
        )

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        text = str(result)

        if text == "none":
            return "none"

        lines = text.splitlines()
        return f"{len(lines)} line(s)"

    @staticmethod
    def _position(arguments: dict[str, Any], text: str) -> SourcePosition:
        line = arguments.get("line")
        character = arguments.get("character")
        if not isinstance(line, int) or not isinstance(character, int):
            raise ValueError("'line' and 'character' are required for this LSP action.")
        if line < 1 or character < 1:
            raise ValueError("LSP line and character values are 1-based and must be positive.")
        lines = text.splitlines()

        # ``splitlines`` omits the final empty line and returns no entries for
        # an empty document, even though line 1/column 1 is a valid LSP
        # position in both cases.
        if not lines or text.endswith(("\n", "\r")):
            lines.append("")
        if line > len(lines):
            raise ValueError(f"Line {line} is outside the {len(lines)}-line document.")
        source_line = lines[line - 1]
        if character > len(source_line) + 1:
            raise ValueError(
                f"Character {character} is outside line {line} "
                f"(maximum {len(source_line) + 1})."
            )
        # LSP defaults to UTF-16 code units; terminal/editor columns are
        # presented as Unicode code points. Astral characters occupy two LSP
        # units and must be accounted for before semantic navigation.
        prefix = source_line[: character - 1]
        utf16_character = len(prefix.encode("utf-16-le")) // 2
        return SourcePosition(line=line - 1, character=utf16_character)

    def _display_uri(self, uri: str) -> str:
        try:
            return self.context.workspace.display_path(uri_to_path(uri))
        except Exception:
            return uri

    def _format_locations(self, value: Any) -> str:
        if value is None:
            return "none"
        items = value if isinstance(value, list) else [value]
        lines: list[str] = []
        for item in items[:200]:
            if not isinstance(item, dict):
                continue
            uri = item.get("uri") or item.get("targetUri")
            range_value = item.get("range") or item.get("targetSelectionRange") or item.get("targetRange")
            if not isinstance(uri, str) or not isinstance(range_value, dict):
                continue
            start = range_value.get("start", {})
            if not isinstance(start, dict):
                continue
            lines.append(
                f"{self._display_uri(uri)}:{int(start.get('line', 0)) + 1}:"
                f"{int(start.get('character', 0)) + 1}"
            )
        return "\n".join(lines) or "none"

    @staticmethod
    def _format_hover(value: Any) -> str:
        if not isinstance(value, dict):
            return "none"
        contents = value.get("contents")
        if isinstance(contents, str):
            return contents
        if isinstance(contents, dict):
            return str(contents.get("value") or contents.get("language") or "none")
        if isinstance(contents, list):
            rendered: list[str] = []
            for item in contents:
                if isinstance(item, str):
                    rendered.append(item)
                elif isinstance(item, dict) and item.get("value") is not None:
                    rendered.append(str(item["value"]))
            return "\n\n".join(rendered) or "none"
        return "none"


    def _format_symbols(self, value: Any, path: Path) -> str:
        if not isinstance(value, list) or not value:
            return "none"
        lines: list[str] = []

        def walk(items: Iterable[Any], depth: int = 0) -> None:
            for item in items:
                if len(lines) >= 500 or not isinstance(item, dict):
                    continue
                range_value = item.get("selectionRange") or item.get("range")
                if not isinstance(range_value, dict):
                    location = item.get("location")
                    range_value = location.get("range") if isinstance(location, dict) else {}
                start = range_value.get("start", {}) if isinstance(range_value, dict) else {}
                lines.append(
                    f"{'  ' * depth}{item.get('name', '<unnamed>')} "
                    f"({self.context.workspace.display_path(path)}:"
                    f"{int(start.get('line', 0)) + 1}:"
                    f"{int(start.get('character', 0)) + 1})"
                )
                children = item.get("children")
                if isinstance(children, list):
                    walk(children, depth + 1)

        walk(value)
        return "\n".join(lines) or "none"

    @staticmethod
    def _reject(arguments: dict[str, Any], *names: str) -> None:
        supplied = [name for name in names if name in arguments]
        if supplied:
            raise ValueError("Arguments not valid for this action: " + ", ".join(supplied))
