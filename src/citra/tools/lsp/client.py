"""High-level lifecycle and semantic operations for one language server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Condition
import time
from typing import Any

from .capabilities import LspCapabilities
from .config import LspConfig, ServerConfig
from .errors import LspDiagnosticsTimeout, LspStartupError
from .language import Language, language_id_for_path
from .positions import SourcePosition, position_to_lsp
from .protocol import path_to_uri
from .transport import JsonRpcTransport


@dataclass
class _Document:
    version: int
    text: str
    language: Language


class LspClient:
    """Own one initialized connection and its open-document state."""

    def __init__(
        self,
        transport: JsonRpcTransport,
        *,
        root: Path,
        server: ServerConfig,
        config: LspConfig,
        name: str,
    ) -> None:
        self.transport = transport
        self.root = root.resolve()
        self.server = server
        self.config = config
        self.name = name
        self.capabilities = LspCapabilities()
        self._documents: dict[str, _Document] = {}
        self._diagnostics: dict[str, list[dict[str, Any]]] = {}
        self._diagnostic_condition = Condition()
        self._closed = False

    def initialize(self) -> None:
        result = self.transport.request(
            "initialize",
            {
                "processId": None,
                "clientInfo": {"name": "Citra", "version": "0.2"},
                "rootUri": path_to_uri(self.root),
                "workspaceFolders": [
                    {"uri": path_to_uri(self.root), "name": self.root.name or "workspace"}
                ],
                "capabilities": {
                    "workspace": {
                        "configuration": True,
                        "workspaceFolders": True,
                        "symbol": {"dynamicRegistration": True},
                    },
                    "textDocument": {
                        "synchronization": {
                            "didSave": True,
                            "dynamicRegistration": True,
                        },
                        "publishDiagnostics": {
                            "relatedInformation": True,
                            "versionSupport": True,
                        },
                        "diagnostic": {"dynamicRegistration": True},
                        "hover": {"contentFormat": ["markdown", "plaintext"]},
                        "definition": {"linkSupport": True},
                        "declaration": {"linkSupport": True},
                        "typeDefinition": {"linkSupport": True},
                        "implementation": {"linkSupport": True},
                        "references": {"dynamicRegistration": True},
                        "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                    },
                    "general": {"positionEncodings": ["utf-16"]},
                },
                "initializationOptions": dict(self.server.initialization_options),
            },
            timeout=self.config.startup_timeout,
        )
        if not isinstance(result, dict):
            raise LspStartupError(f"{self.name} returned an invalid initialize result.")
        raw_capabilities = result.get("capabilities")
        if raw_capabilities is not None and not isinstance(raw_capabilities, dict):
            raise LspStartupError(f"{self.name} returned invalid server capabilities.")
        self.capabilities = LspCapabilities.from_server_capabilities(raw_capabilities)
        self.transport.notify("initialized", {})
        if self.server.settings:
            self.transport.notify(
                "workspace/didChangeConfiguration",
                {"settings": dict(self.server.settings)},
            )

    def handle_notification(self, method: str, params: Any) -> None:
        if method != "textDocument/publishDiagnostics" or not isinstance(params, dict):
            return
        uri = params.get("uri")
        diagnostics = params.get("diagnostics")
        if not isinstance(uri, str) or not isinstance(diagnostics, list):
            return
        cleaned = [item for item in diagnostics if isinstance(item, dict)]
        with self._diagnostic_condition:
            self._diagnostics[uri] = cleaned
            self._diagnostic_condition.notify_all()

    def handle_request(self, method: str, params: Any) -> Any:
        if method == "workspace/configuration":
            items = params.get("items", []) if isinstance(params, dict) else []
            return [dict(self.server.settings) for _ in items]
        if method == "workspace/workspaceFolders":
            return [{"uri": path_to_uri(self.root), "name": self.root.name or "workspace"}]
        if method == "client/registerCapability":
            registrations = (
                params.get("registrations", [])
                if isinstance(params, dict)
                else []
            )
            if isinstance(registrations, list):
                self.capabilities = self.capabilities.with_dynamic_registration(
                    [item for item in registrations if isinstance(item, dict)]
                )
            return None
        if method == "client/unregisterCapability":
            values = []
            if isinstance(params, dict):
                # The specification historically shipped the misspelled
                # ``unregisterations`` key; accept both spellings.
                values = params.get(
                    "unregistrations",
                    params.get("unregisterations", []),
                )
            if isinstance(values, list):
                self.capabilities = self.capabilities.with_dynamic_unregistration(
                    [item for item in values if isinstance(item, dict)]
                )
            return None
        if method == "window/workDoneProgress/create":
            return None
        if method == "workspace/applyEdit":
            return {"applied": False, "failureReason": "Citra LSP access is read-only."}
        if method == "window/showMessageRequest":
            return None
        return None

    def sync_document(self, path: Path, text: str, language: Language) -> str:
        uri = path_to_uri(path)
        current = self._documents.get(uri)
        if current is None:
            self._documents[uri] = _Document(version=1, text=text, language=language)
            self._clear_push_diagnostics(uri)
            self.transport.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": language_id_for_path(path, language),
                        "version": 1,
                        "text": text,
                    }
                },
            )
            return uri
        if current.text != text or current.language is not language:
            version = current.version + 1
            self._documents[uri] = _Document(version=version, text=text, language=language)
            self._clear_push_diagnostics(uri)
            self.transport.notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": text}],
                },
            )
        return uri

    def _clear_push_diagnostics(self, uri: str) -> None:
        """Require a fresh publish after opening or changing a document."""
        with self._diagnostic_condition:
            self._diagnostics.pop(uri, None)

    def hover(self, uri: str, position: SourcePosition) -> Any:
        self.capabilities.require("hover")
        return self._position_request("textDocument/hover", uri, position)

    def definitions(self, kind: str, uri: str, position: SourcePosition) -> Any:
        methods = {
            "definition": ("definition", "textDocument/definition"),
            "declaration": ("declaration", "textDocument/declaration"),
            "type_definition": ("type_definition", "textDocument/typeDefinition"),
            "implementation": ("implementation", "textDocument/implementation"),
        }
        capability, method = methods[kind]
        self.capabilities.require(capability)
        return self._position_request(method, uri, position)

    def references(self, uri: str, position: SourcePosition, *, include_declaration: bool) -> Any:
        self.capabilities.require("references")
        return self.transport.request(
            "textDocument/references",
            {
                "textDocument": {"uri": uri},
                "position": position_to_lsp(position),
                "context": {"includeDeclaration": include_declaration},
            },
            timeout=self.config.request_timeout,
        )

    def document_symbols(self, uri: str) -> Any:
        self.capabilities.require("document_symbols")
        return self.transport.request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": uri}},
            timeout=self.config.request_timeout,
        )

    def diagnostics(self, uri: str) -> list[dict[str, Any]]:
        if self.capabilities.diagnostics_pull:
            result = self.transport.request(
                "textDocument/diagnostic",
                {"textDocument": {"uri": uri}},
                timeout=self.config.diagnostics_timeout,
            )
            if isinstance(result, dict):
                items = result.get("items", [])
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]

        deadline = time.monotonic() + self.config.diagnostics_timeout
        with self._diagnostic_condition:
            while uri not in self._diagnostics:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LspDiagnosticsTimeout(
                        f"{self.name} did not publish diagnostics within "
                        f"{self.config.diagnostics_timeout:.1f}s."
                    )
                self._diagnostic_condition.wait(remaining)
            return list(self._diagnostics[uri])

    def _position_request(self, method: str, uri: str, position: SourcePosition) -> Any:
        return self.transport.request(
            method,
            {
                "textDocument": {"uri": uri},
                "position": position_to_lsp(position),
            },
            timeout=self.config.request_timeout,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.transport.request("shutdown", timeout=min(5.0, self.config.request_timeout))
            self.transport.notify("exit")
        except Exception:
            pass
        finally:
            self.transport.close()
