"""Lifecycle-scoped language-server discovery, startup, reuse, and shutdown."""

from __future__ import annotations

from pathlib import Path
import shutil
from threading import RLock
from typing import Any

from citra.context.turn_workspace import WorkspaceContext
from citra.utils.sandbox import WorkspaceSandbox

from .client import LspClient
from .config import LspConfig, ServerConfig
from .errors import LspStartupError, LspUnavailable
from .language import (
    Language,
    detect_language,
    server_for_language,
    supports_language,
)
from .servers import SERVERS
from .transport import JsonRpcTransport


class LspManager:
    """Keep at most one server per adapter/project-root for the process."""

    def __init__(
        self,
        workspace: WorkspaceContext,
        sandbox: WorkspaceSandbox,
        *,
        config: LspConfig | None = None,
    ) -> None:
        self.workspace = workspace
        self.sandbox = sandbox
        self.config = config or LspConfig()
        self._clients: dict[tuple[str, Path], LspClient] = {}
        self._lock = RLock()
        self._closed = False

    def client_for(self, path: Path) -> tuple[LspClient, Language]:
        if self._closed:
            raise LspUnavailable("The LSP manager has already been closed.")
        path = self.workspace.require_allowed_path(path)
        language = detect_language(path)
        if language is None or not supports_language(language):
            raise LspUnavailable(f"No language server is configured for {path.suffix or 'this file'}.")
        server_id = server_for_language(language)
        root = self._root_for(path)
        key = (server_id, root)
        with self._lock:
            existing = self._clients.get(key)
            if existing is not None and existing.transport.process.poll() is None:
                return existing, language
            if existing is not None:
                existing.close()
                self._clients.pop(key, None)
            client = self._start(server_id, root)
            self._clients[key] = client
            return client, language

    def status(self) -> dict[str, Any]:
        servers: list[dict[str, Any]] = []
        with self._lock:
            running = {
                server_id: sum(
                    1
                    for (candidate, _), client in self._clients.items()
                    if candidate == server_id and client.transport.process.poll() is None
                )
                for server_id in SERVERS
            }
        for definition in SERVERS.values():
            executable = shutil.which(definition.executable)
            servers.append(
                {
                    "id": definition.id,
                    "available": executable is not None,
                    "executable": executable,
                    "running": running.get(definition.id, 0),
                    "install_hint": None if executable else definition.install_hint,
                }
            )
        return {"enabled": self.config.enabled, "servers": servers}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            clients = tuple(self._clients.values())
            self._clients.clear()
        for client in clients:
            process = client.transport.process
            client.close()
            try:
                process.wait(timeout=2)
            except Exception:
                self.sandbox.terminate_process(process)

    def _start(self, server_id: str, root: Path) -> LspClient:
        if not self.config.enabled:
            raise LspUnavailable("LSP support is disabled by configuration.")
        definition = SERVERS[server_id]
        executable = shutil.which(definition.executable)
        if executable is None:
            raise LspUnavailable(
                f"Language server {definition.executable!r} is not installed. "
                f"{definition.install_hint}"
            )
        server_config = ServerConfig(
            command=(executable, *definition.arguments),
            extensions=(),
        )
        process = self.sandbox.popen(
            server_config.command,
            cwd=root,
            network=False,
            environment=dict(server_config.environment),
        )
        holder: dict[str, LspClient] = {}

        def handle_notification(method: str, params: Any) -> None:
            # A server may emit a log notification immediately after process
            # start, before the high-level client has been assigned. Such a
            # notification is advisory and must not kill the transport reader.
            client = holder.get("client")
            if client is not None:
                client.handle_notification(method, params)

        def handle_request(method: str, params: Any) -> Any:
            client = holder.get("client")
            if client is None:
                return None
            return client.handle_request(method, params)

        transport = JsonRpcTransport(
            process,
            notification_handler=handle_notification,
            request_handler=handle_request,
        )
        client = LspClient(
            transport,
            root=root,
            server=server_config,
            config=self.config,
            name=server_id,
        )
        holder["client"] = client
        try:
            client.initialize()
        except Exception as error:
            transport.close()
            self.sandbox.terminate_process(process)
            raise LspStartupError(f"Could not initialize {server_id}: {error}") from error
        return client

    def _root_for(self, path: Path) -> Path:
        if self.workspace._is_within(self.workspace.workspace, path):
            return self.workspace.workspace
        if self.workspace._is_within(self.workspace.source_workspace, path):
            return self.workspace.source_workspace
        raise LspUnavailable(f"LSP path is outside the active workspace: {path}")
