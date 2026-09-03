"""High-level lifecycle and semantic operations for one language server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Condition
import time
from typing import Any, Callable

from .capabilities import LspCapabilities
from .config import LspConfig, ServerConfig
from .errors import LspDiagnosticsTimeout, LspProtocolError, LspStartupError
from .language import Language, language_id_for_path
from .positions import SourcePosition, position_to_lsp
from .protocol import path_to_uri
from .transport import JsonRpcTransport


@dataclass
class _Document:
    """Represent Document."""
    version: int
    text: str
    language: Language
    generation: int


@dataclass
class _PushDiagnostics:
    """Represent PushDiagnostics."""
    version: int | None
    generation: int
    items: list[dict[str, Any]]


@dataclass
class _PullDiagnostics:
    """Represent PullDiagnostics."""
    result_id: str | None
    items: list[dict[str, Any]]


TsserverBridge = Callable[[str, Any, dict[str, Any]], Any]
MirrorSync = Callable[[Path, str, Language], None]


def configuration_for_section(settings: dict[str, Any], section: str | None) -> Any:
    """Resolve a dotted ``workspace/configuration`` section."""
    if not section:
        return settings
    current: Any = settings
    for component in section.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(component)
        if current is None:
            return None
    return current


def _document_end_position(text: str) -> dict[str, int]:
    """Return the UTF-16 LSP position immediately after *text*."""
    line = text.count("\n")
    tail = text.rsplit("\n", 1)[-1]
    if tail.endswith("\r"):
        tail = tail[:-1]
    character = len(tail.encode("utf-16-le")) // 2
    return {"line": line, "character": character}


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
        """Initialize the instance."""
        self.transport = transport
        self.root = root.resolve()
        self.server = server
        self.config = config
        self.name = name
        self.capabilities = LspCapabilities()
        self._static_capabilities = LspCapabilities()
        self._dynamic_registrations: dict[str, dict[str, Any]] = {}
        self._documents: dict[str, _Document] = {}
        self._push_diagnostics: dict[str, _PushDiagnostics] = {}
        self._pull_diagnostics: dict[str, _PullDiagnostics] = {}
        self._diagnostic_registration_options: dict[str, Any] | None = None
        self._diagnostic_condition = Condition()
        self._closed = False
        self._text_document_sync = 1
        self._open_close_sync = True
        self._cold_diagnostics_pending = True
        self._tsserver_bridge: TsserverBridge | None = None
        self._mirror_sync: MirrorSync | None = None

    def set_tsserver_bridge(self, bridge: TsserverBridge | None) -> None:
        """Handle set tsserver bridge."""
        self._tsserver_bridge = bridge

    def set_mirror_sync(self, callback: MirrorSync | None) -> None:
        """Handle set mirror sync."""
        self._mirror_sync = callback

    def initialize(self) -> None:
        """Handle initialize."""
        result = self.transport.request(
            "initialize",
            {
                "processId": None,
                "clientInfo": {"name": "Citra", "version": "0.3"},
                "rootUri": path_to_uri(self.root),
                "rootPath": str(self.root),
                "workspaceFolders": [
                    {"uri": path_to_uri(self.root), "name": self.root.name or "workspace"}
                ],
                "capabilities": {
                    "workspace": {
                        "configuration": True,
                        "workspaceFolders": True,
                        "diagnostics": {"refreshSupport": True},
                    },
                    "textDocument": {
                        "synchronization": {
                            # Citra currently owns didOpen/didChange directly; it
                            # does not implement dynamic sync registration or
                            # didSave notifications, so do not advertise them.
                            "didSave": False,
                            "dynamicRegistration": False,
                        },
                        "publishDiagnostics": {
                            "relatedInformation": True,
                            "versionSupport": True,
                        },
                        "diagnostic": {
                            "dynamicRegistration": True,
                            "relatedDocumentSupport": False,
                        },
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
        raw_capabilities = raw_capabilities or {}
        self._static_capabilities = LspCapabilities.from_server_capabilities(raw_capabilities)
        self.capabilities = self._static_capabilities
        self._text_document_sync, self._open_close_sync = self._parse_sync(raw_capabilities.get("textDocumentSync"))
        self.transport.notify("initialized", {})
        if self.server.settings:
            self.transport.notify(
                "workspace/didChangeConfiguration",
                {"settings": dict(self.server.settings)},
            )

    @staticmethod
    def _parse_sync(value: Any) -> tuple[int, bool]:
        """Return ``(change_kind, open_close)`` from textDocumentSync."""
        if isinstance(value, int) and not isinstance(value, bool):
            change = value if value in {0, 1, 2} else 1
            return change, change != 0
        if isinstance(value, dict):
            change = value.get("change", 0)
            if not isinstance(change, int) or isinstance(change, bool) or change not in {0, 1, 2}:
                change = 0
            return change, bool(value.get("openClose", False))
        if value is None or value is False:
            return 0, False
        return 1, True

    def handle_notification(self, method: str, params: Any) -> None:
        """Handle handle notification."""
        if method == "textDocument/publishDiagnostics":
            self._handle_publish_diagnostics(params)
            return
        if method == "tsserver/request":
            self._handle_tsserver_request(params)
            return
        # Advisory notifications are deliberately accepted and ignored.
        if method in {"window/logMessage", "window/showMessage", "$/progress"}:
            return

    def _handle_publish_diagnostics(self, params: Any) -> None:
        """Handle handle publish diagnostics."""
        if not isinstance(params, dict):
            return
        uri = params.get("uri")
        diagnostics = params.get("diagnostics")
        if not isinstance(uri, str) or not isinstance(diagnostics, list):
            return
        current = self._documents.get(uri)
        if current is None:
            return
        version_raw = params.get("version")
        version = version_raw if isinstance(version_raw, int) else None
        if version is not None and version != current.version:
            return
        cleaned = [item for item in diagnostics if isinstance(item, dict)]
        with self._diagnostic_condition:
            self._push_diagnostics[uri] = _PushDiagnostics(
                version=version,
                generation=current.generation,
                items=cleaned,
            )
            self._diagnostic_condition.notify_all()

    def _handle_tsserver_request(self, params: Any) -> None:
        """Handle handle tsserver request."""
        bridge = self._tsserver_bridge
        if bridge is None or not isinstance(params, list):
            return
        nested = len(params) == 1 and isinstance(params[0], list)
        payload = params[0] if nested else params
        if len(payload) < 2:
            return
        seq = payload[0]
        command = payload[1]
        args = payload[2] if len(payload) >= 3 else None
        if not isinstance(command, str):
            return
        try:
            response = bridge(command, args, {"isAsync": True, "lowPriority": True})
            if isinstance(response, dict) and "body" in response:
                response = response.get("body")
        except Exception:
            # Vue requires a response even on bridge failure to avoid leaving
            # its pending tsserver request alive indefinitely.
            response = None
        response_params: list[Any] = [seq, response]
        if nested:
            response_params = [response_params]
        self.transport.notify("tsserver/response", response_params)

    def handle_request(self, method: str, params: Any) -> Any:
        """Handle handle request."""
        if method == "workspace/configuration":
            items = params.get("items", []) if isinstance(params, dict) else []
            settings = dict(self.server.settings)
            if not isinstance(items, list):
                return []
            return [
                configuration_for_section(
                    settings,
                    item.get("section") if isinstance(item, dict) else None,
                )
                for item in items
            ]
        if method == "workspace/workspaceFolders":
            return [{"uri": path_to_uri(self.root), "name": self.root.name or "workspace"}]
        if method == "workspace/diagnostic/refresh":
            with self._diagnostic_condition:
                self._pull_diagnostics.clear()
            return None
        if method == "client/registerCapability":
            registrations = params.get("registrations", []) if isinstance(params, dict) else []
            if isinstance(registrations, list):
                cleaned = [item for item in registrations if isinstance(item, dict)]
                with self._diagnostic_condition:
                    for item in cleaned:
                        registration_id = item.get("id")
                        method_name = item.get("method")
                        if not isinstance(registration_id, str) or not isinstance(method_name, str):
                            continue
                        # Dynamic registrations are identities, not booleans. A
                        # server may register a replacement capability before
                        # unregistering the previous id. Pyright does this when
                        # refreshing its pull-diagnostic feature after settings
                        # changes. Tracking only the method would therefore let
                        # the old unregister incorrectly disable the new one.
                        self._dynamic_registrations[registration_id] = dict(item)
                    self._refresh_dynamic_capabilities_locked()
                    # A server such as modern Pyright can decide to use pull
                    # diagnostics from the client initialize capabilities, but
                    # register textDocument/diagnostic only after initialized.
                    # Wake an in-flight diagnostics() call so it can switch
                    # from waiting for push notifications to issuing the pull.
                    self._diagnostic_condition.notify_all()
            return None
        if method == "client/unregisterCapability":
            values: Any = []
            if isinstance(params, dict):
                values = params.get("unregistrations", params.get("unregisterations", []))
            if isinstance(values, list):
                cleaned = [item for item in values if isinstance(item, dict)]
                with self._diagnostic_condition:
                    for item in cleaned:
                        registration_id = item.get("id")
                        if isinstance(registration_id, str):
                            self._dynamic_registrations.pop(registration_id, None)
                    self._refresh_dynamic_capabilities_locked()
                    self._diagnostic_condition.notify_all()
            return None
        if method == "tsserver/request" and self._tsserver_bridge is not None:
            if isinstance(params, list) and len(params) >= 2 and isinstance(params[1], str):
                return self._tsserver_bridge(
                    params[1],
                    params[2] if len(params) >= 3 else None,
                    {"isAsync": True, "lowPriority": True},
                )
            return None
        if method == "window/workDoneProgress/create":
            return None
        if method == "workspace/applyEdit":
            return {"applied": False, "failureReason": "Citra LSP access is read-only."}
        if method == "window/showMessageRequest":
            return None
        return None

    def _refresh_dynamic_capabilities_locked(self) -> None:
        """Rebuild effective capabilities from active registration identities.

        The caller must hold ``_diagnostic_condition``. Dynamic LSP features can
        overlap during replacement, so unregistering one id must not disable a
        method that is still registered under another id.
        """
        registrations = list(self._dynamic_registrations.values())
        self.capabilities = self._static_capabilities.with_dynamic_registration(registrations)

        diagnostic_options: dict[str, Any] | None = None
        # Dict insertion order gives us the newest surviving registration last.
        for item in registrations:
            if item.get("method") != "textDocument/diagnostic":
                continue
            options = item.get("registerOptions")
            diagnostic_options = dict(options) if isinstance(options, dict) else {}
        self._diagnostic_registration_options = diagnostic_options

    def sync_document(self, path: Path, text: str, language: Language) -> str:
        """Handle sync document."""
        uri = path_to_uri(path)
        current = self._documents.get(uri)
        if current is None:
            document = _Document(version=1, text=text, language=language, generation=1)
            self._documents[uri] = document
            self._mark_diagnostics_stale(uri)
            if self._open_close_sync:
                self.transport.notify(
                    "textDocument/didOpen",
                    {
                        "textDocument": {
                            "uri": uri,
                            "languageId": language_id_for_path(path, language),
                            "version": document.version,
                            "text": text,
                        }
                    },
                )
            self._mirror(path, text, language)
            return uri

        if current.text != text or current.language is not language:
            document = _Document(
                version=current.version + 1,
                text=text,
                language=language,
                generation=current.generation + 1,
            )
            self._documents[uri] = document
            self._mark_diagnostics_stale(uri)
            if self._text_document_sync != 0:
                if self._text_document_sync == 2:
                    content_changes = [
                        {
                            "range": {
                                "start": {"line": 0, "character": 0},
                                "end": _document_end_position(current.text),
                            },
                            "text": text,
                        }
                    ]
                else:
                    content_changes = [{"text": text}]
                self.transport.notify(
                    "textDocument/didChange",
                    {
                        "textDocument": {"uri": uri, "version": document.version},
                        "contentChanges": content_changes,
                    },
                )
            self._mirror(path, text, language)
        return uri

    def _mirror(self, path: Path, text: str, language: Language) -> None:
        """Handle mirror."""
        callback = self._mirror_sync
        if callback is not None:
            callback(path, text, language)

    def _mark_diagnostics_stale(self, uri: str) -> None:
        """Handle mark diagnostics stale."""
        with self._diagnostic_condition:
            self._push_diagnostics.pop(uri, None)
            self._pull_diagnostics.pop(uri, None)

    def hover(self, uri: str, position: SourcePosition) -> Any:
        """Handle hover."""
        self.capabilities.require("hover")
        return self._position_request("textDocument/hover", uri, position)

    def definitions(self, kind: str, uri: str, position: SourcePosition) -> Any:
        """Handle definitions."""
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
        """Handle references."""
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
        """Handle document symbols."""
        self.capabilities.require("document_symbols")
        return self.transport.request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": uri}},
            timeout=self.config.request_timeout,
        )

    def execute_command(self, command: str, arguments: list[Any]) -> Any:
        """Execute the execute command operation."""
        return self.transport.request(
            "workspace/executeCommand",
            {"command": command, "arguments": arguments},
            timeout=self.config.request_timeout,
        )

    def diagnostics(self, uri: str, *, timeout: float | None = None) -> list[dict[str, Any]]:
        """Handle diagnostics."""
        effective_timeout = self._diagnostics_timeout() if timeout is None else timeout
        if effective_timeout <= 0:
            raise ValueError("LSP diagnostics timeout must be positive.")

        document = self._documents.get(uri)
        if document is None:
            return []

        # Pull diagnostic support may be registered dynamically after the
        # initialize response. Modern Pyright does exactly this: it suppresses
        # push diagnostics as soon as the client advertises pull support, then
        # registers textDocument/diagnostic after ``initialized``. Therefore a
        # diagnostics request that starts in push mode must be able to switch
        # to pull mode while it is waiting.
        deadline = time.monotonic() + effective_timeout
        while True:
            with self._diagnostic_condition:
                if self.capabilities.diagnostics_pull:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LspDiagnosticsTimeout(
                            f"{self.name} did not make diagnostics available within "
                            f"{effective_timeout:.1f}s."
                        )
                    use_pull = True
                else:
                    use_pull = False
                    published = self._push_diagnostics.get(uri)
                    current = self._documents.get(uri)
                    if current is None:
                        return []
                    if published is not None and published.generation == current.generation:
                        if published.version is None or published.version == current.version:
                            self._cold_diagnostics_pending = False
                            return list(published.items)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LspDiagnosticsTimeout(
                            f"{self.name} did not publish diagnostics within "
                            f"{effective_timeout:.1f}s."
                        )
                    self._diagnostic_condition.wait(remaining)
                    continue

            if use_pull:
                return self._pull_document_diagnostics(uri, timeout=remaining)

    def _pull_document_diagnostics(self, uri: str, *, timeout: float) -> list[dict[str, Any]]:
        """Handle pull document diagnostics."""
        previous = self._pull_diagnostics.get(uri)
        params: dict[str, Any] = {"textDocument": {"uri": uri}}
        provider = (self.capabilities.raw or {}).get("diagnosticProvider")
        if not isinstance(provider, dict):
            provider = self._diagnostic_registration_options
        if isinstance(provider, dict) and isinstance(provider.get("identifier"), str):
            params["identifier"] = provider["identifier"]
        if previous is not None and previous.result_id:
            params["previousResultId"] = previous.result_id
        result = self.transport.request(
            "textDocument/diagnostic",
            params,
            timeout=timeout,
        )
        if not isinstance(result, dict):
            raise LspProtocolError(
                f"{self.name} returned an invalid textDocument/diagnostic response."
            )
        kind = result.get("kind", "full")
        result_id = result.get("resultId")
        result_id = result_id if isinstance(result_id, str) else None
        if kind == "unchanged":
            if previous is None:
                raise LspProtocolError(
                    f"{self.name} returned an unchanged diagnostic report without a previous result."
                )
            self._cold_diagnostics_pending = False
            return list(previous.items)
        if kind != "full":
            raise LspProtocolError(
                f"{self.name} returned unsupported diagnostic report kind {kind!r}."
            )
        items = result.get("items")
        if not isinstance(items, list):
            raise LspProtocolError(
                f"{self.name} returned a full diagnostic report without an items list."
            )
        cleaned = [item for item in items if isinstance(item, dict)]
        self._pull_diagnostics[uri] = _PullDiagnostics(result_id, cleaned)
        self._cold_diagnostics_pending = False
        return list(cleaned)

    def _diagnostics_timeout(self) -> float:
        """Handle diagnostics timeout."""
        if not self._cold_diagnostics_pending:
            return self.config.diagnostics_timeout
        cold = self.server.cold_diagnostics_timeout
        if cold is None:
            cold = self.config.cold_diagnostics_timeout
        return max(self.config.diagnostics_timeout, cold)

    def _position_request(self, method: str, uri: str, position: SourcePosition) -> Any:
        """Handle position request."""
        return self.transport.request(
            method,
            {"textDocument": {"uri": uri}, "position": position_to_lsp(position)},
            timeout=self.config.request_timeout,
        )

    def close(self) -> None:
        """Handle close."""
        if self._closed:
            return
        self._closed = True
        process = self.transport.process
        try:
            if self._open_close_sync and process.poll() is None:
                for uri in tuple(self._documents):
                    try:
                        self.transport.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
                    except Exception:
                        break
            self.transport.request("shutdown", timeout=min(5.0, self.config.request_timeout))
            self.transport.notify("exit")
        except Exception:
            pass
        finally:
            self.transport.close()
        try:
            process.wait(timeout=1.0)
        except Exception:
            # The manager owns forced termination because it knows the sandbox
            # process-group policy. Closing a standalone client remains best-effort.
            pass
