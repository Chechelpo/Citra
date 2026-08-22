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
from ..lsp.manager import LspManager
from ..lsp.positions import SourcePosition
from ..lsp.protocol import uri_to_path
from ..tool import Tool


class Lsp(Tool):
    """Expose high-value language intelligence without raw JSON-RPC."""

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

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="lsp",
            description=(
                "Use a persistent sandboxed language server for semantic code "
                "intelligence. Supports Python plus JavaScript/TypeScript. "
                "Actions include diagnostics, hover, document symbols, "
                "references, and go-to-definition (plus declaration, type "
                "definition, and implementation). line and character are "
                "1-based. Use status to see whether Pyright and "
                "typescript-language-server are installed."
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
                                "Source path in the agent workspace or @source. "
                                "Required except for status."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="line",
                        schema=JsonSchema.integer(
                            description="1-based source line for position actions.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="character",
                        schema=JsonSchema.integer(
                            description="1-based character/column for position actions.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="include_declaration",
                        schema=JsonSchema.boolean(
                            description="Include the declaration in reference results.",
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    def __init__(self, context: ExecutionContext) -> None:
        super().__init__(context=context, definition=self.DEFINITION)

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        manager = self.context.lsp_manager
        if not isinstance(manager, LspManager):
            raise RuntimeError("LSP services are unavailable in this execution context.")
        action = arguments["action"]
        if action == "status":
            self._reject(arguments, "path", "line", "character", "include_declaration")
            return json.dumps(manager.status(), indent=2, ensure_ascii=False)
        path_raw = arguments.get("path")
        if not path_raw:
            raise ValueError("'path' is required for this LSP action.")
        path = self.context.workspace.resolve_path(path_raw)
        text = self.context.filesystem.execute("read_raw", {"path": str(path)})
        client, language = manager.client_for(path)
        uri = client.sync_document(path, text, language)

        if action == "diagnostics":
            self._reject(arguments, "line", "character", "include_declaration")
            return self._format_diagnostics(client.diagnostics(uri), path)
        if action == "document_symbols":
            self._reject(arguments, "line", "character", "include_declaration")
            return self._format_symbols(client.document_symbols(uri), path)
        if action not in self.POSITION_ACTIONS:
            raise ValueError(f"Unsupported LSP action: {action}")
        position = self._position(arguments, text)
        if action == "hover":
            self._reject(arguments, "include_declaration")
            return self._format_hover(client.hover(uri, position))
        if action == "references":
            return self._format_locations(
                client.references(
                    uri,
                    position,
                    include_declaration=arguments.get("include_declaration", True),
                )
            )
        self._reject(arguments, "include_declaration")
        definition_kind = "definition" if action == "go_to_definition" else action
        return self._format_locations(client.definitions(definition_kind, uri, position))

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

    def _format_diagnostics(self, value: Any, path: Path) -> str:
        if not isinstance(value, list) or not value:
            return "none"
        severities = {1: "error", 2: "warning", 3: "information", 4: "hint"}
        lines: list[str] = []
        for item in value[:250]:
            if not isinstance(item, dict):
                continue
            range_value = item.get("range", {})
            start = range_value.get("start", {}) if isinstance(range_value, dict) else {}
            severity = severities.get(item["severity"], "diagnostic")
            source = f" [{item['source']}]" if item.get("source") else ""
            lines.append(
                f"{self.context.workspace.display_path(path)}:"
                f"{int(start.get('line', 0)) + 1}:"
                f"{int(start.get('character', 0)) + 1}: "
                f"{severity}{source}: {item.get('message', '')}"
            )
        return "\n".join(lines) or "none"

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
