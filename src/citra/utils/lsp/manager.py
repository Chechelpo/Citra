"""Lifecycle-scoped language-server discovery, startup, reuse, and shutdown."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any

from citra.context.workspace_context import WorkspaceContext
from citra.logging import Logger
from citra.sandbox.sandbox import WorkspaceSandbox
from citra.sandbox.filesystem_ops import ReadRawInput
from citra.sandbox.sandboxed_filesystem import SandboxedFilesystem

from .client import LspClient
from .config import LspConfig, ServerConfig
from .diagnostics import format_diagnostics, json_fallback_diagnostics
from .errors import LspDiagnosticsTimeout, LspError, LspStartupError, LspUnavailable
from .installer import InstallResult, available_managers, candidate_for, execute_install
from .interpreters import ResolvedInterpreter
from .language import Language, detect_language, server_for_language, supports_language
from .servers import SERVER_ALIASES, SERVERS
from .servers.base import ServerDefinition
from .transport import JsonRpcTransport


logger = logging.getLogger(__name__)
_activity_logger = Logger(__name__)


def _merge_dict(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Shallow-merge ``overlay`` on top of ``base``.

    The overlay wins on key conflicts. Both inputs are treated as plain
    mappings; the result is a fresh ``dict`` so callers can mutate it
    without leaking state into the frozen ``ServerConfig``.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        merged[key] = value
    return merged


def _merge_settings(
    base: Mapping[str, Any],
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    """Deep-merge two ``workspace/didChangeConfiguration`` payloads.

    The overlay wins on key conflicts at every depth. Nested mappings are
    merged recursively so the resolver can set
    ``settings["python"]["pythonPath"]`` without clobbering
    ``settings["python"]["analysis"]`` (and vice versa).
    """
    merged: dict[str, Any] = {}
    for key, value in base.items():
        merged[key] = value
    for key, overlay_value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(overlay_value, Mapping):
            merged[key] = _merge_settings(base_value, overlay_value)
        else:
            merged[key] = overlay_value
    return merged


@dataclass(frozen=True)
class ClientKey:
    """Represent ClientKey."""
    root: Path
    server_id: str


@dataclass(frozen=True)
class LspClientHandle:
    """Represent LspClientHandle."""
    client: LspClient
    language: Language
    cold_start: bool

    def __iter__(self):
        # Backwards compatible with older ``client, language = client_for(...)`` callers.
        """Handle iter."""
        yield self.client
        yield self.language

    def __len__(self) -> int:
        """Handle len."""
        return 2

    def __getitem__(self, index: int):
        """Handle getitem."""
        return (self.client, self.language)[index]


class LspManager:
    """Keep one healthy server per server identity/project root for the process."""

    def __init__(
        self,
        workspace: WorkspaceContext,
        sandbox: WorkspaceSandbox,
        *,
        config: LspConfig | None = None,
    ) -> None:
        """Initialize the instance."""
        self.workspace = workspace
        self.sandbox = sandbox
        self.config = config or LspConfig()
        self._clients: dict[ClientKey, LspClient] = {}
        self._typescript_vue_roots: set[Path] = set()
        self._lock = RLock()
        self._closed = False

    def client_for(self, path: Path) -> LspClientHandle:
        """Handle client for."""
        if self._closed:
            raise LspUnavailable("The LSP manager has already been closed.")
        try:
            path = self.workspace.require_allowed_path(path)
        except ValueError as error:
            raise LspUnavailable(str(error)) from error
        language = detect_language(path)
        if language is None:
            raise LspUnavailable(
                f"No language server is configured for {path.suffix or 'this file'}."
            )
        if not supports_language(language):
            raise LspUnavailable(
                f"No language server is configured for {language.value}."
            )
        server_id = self._normalize_server_target(server_for_language(language))
        root = self._root_for(path)
        if language is Language.VUE:
            self._ensure_vue_typescript(root)
        client, started = self._client_for_server(server_id, root)
        return LspClientHandle(client=client, language=language, cold_start=started)

    def _client_for_server(self, server_id: str, root: Path) -> tuple[LspClient, bool]:
        """Handle client for server."""
        key = ClientKey(root.resolve(), server_id)
        with self._lock:
            existing = self._clients.get(key)
            if existing is not None and existing.transport.process.poll() is None:
                return existing, False
            if existing is not None:
                self._dispose_client(key, existing)
            client = self._start(server_id, root)
            self._clients[key] = client
            return client, True

    def status(self) -> dict[str, Any]:
        """Handle status."""
        _activity_logger.debug("Collecting sandbox language-server status")
        with self._lock:
            running = {
                server_id: sum(
                    1
                    for key, client in self._clients.items()
                    if key.server_id == server_id
                    and client.transport.process.poll() is None
                )
                for server_id in SERVERS
            }
        servers: list[dict[str, Any]] = []
        managers = self._available_managers()
        for definition in SERVERS.values():
            available, details = self._availability(definition)
            executable = details["executable"]
            candidate = candidate_for(
                definition,
                managers=managers,
            )
            servers.append(
                {
                    "id": definition.id,
                    "languages": [language.value for language in definition.languages],
                    "installed": executable is not None,
                    "available": available,
                    "executable": executable,
                    "running": running.get(definition.id, 0),
                    "optional_dependencies": details["optional_dependencies"],
                    "installation_method": candidate.manager if candidate else None,
                    "install_hint": None if available else definition.install_hint,
                }
            )
        result = {"enabled": self.config.enabled, "servers": servers}
        _activity_logger.info(
            "Collected sandbox language-server status",
            enabled=self.config.enabled,
            installed=sum(bool(item["installed"]) for item in servers),
            available=sum(bool(item["available"]) for item in servers),
            running=sum(int(item["running"]) for item in servers),
        )
        return result

    def diagnostics(
        self,
        path: Path,
        text: str,
    ) -> list[dict[str, Any]]:
        """Handle diagnostics."""
        if not self.config.enabled:
            raise LspUnavailable("LSP support is disabled by configuration.")
        path = self.workspace.require_allowed_path(path)
        language = detect_language(path)
        if language is None:
            raise LspUnavailable(
                f"No language server is configured for {path.suffix or 'this file'}."
            )
        try:
            handle = self.client_for(path)
        except LspUnavailable:
            if language is Language.JSON and self.config.json_fallback:
                return json_fallback_diagnostics(text)
            raise
        uri = handle.client.sync_document(path, text, handle.language)
        if handle.cold_start:
            cold = handle.client.server.cold_diagnostics_timeout
            if cold is None:
                cold = self.config.cold_diagnostics_timeout
            return handle.client.diagnostics(
                uri,
                timeout=max(self.config.diagnostics_timeout, cold),
            )
        return handle.client.diagnostics(uri)

    def diagnostics_for_path(
        self,
        path_raw: str,
        *,
        filesystem: SandboxedFilesystem,
    ) -> str | None:
        """Handle diagnostics for path."""
        path = self.workspace.resolve_path(path_raw)
        language = detect_language(path)
        if language is None or not supports_language(language):
            return None
        text = filesystem.execute(ReadRawInput(path=str(path)))
        try:
            diagnostics = self.diagnostics(path, text.content)
        except (LspUnavailable, LspDiagnosticsTimeout):
            # Automatic Edit/Write diagnostics are advisory. A language server
            # that is temporarily unable to produce diagnostics must not turn a
            # successful filesystem mutation into a logged operational failure.
            return None
        if not diagnostics:
            return None
        return format_diagnostics(
            diagnostics,
            path=path,
            display_path=self.workspace.display_path,
        )

    def stop(self, target: str | None = None) -> int:
        """Handle stop."""
        server_id = self._normalize_server_target(target) if target else None
        with self._lock:
            selected = [
                (key, client)
                for key, client in self._clients.items()
                if server_id is None or key.server_id == server_id
            ]
            for key, client in selected:
                self._clients.pop(key, None)
                if key.server_id == "vue":
                    self._typescript_vue_roots.discard(key.root)
        for key, client in selected:
            self._shutdown_client(client)
        return len(selected)

    def restart(self, target: str | None = None) -> int:
        """Handle restart."""
        server_id = self._normalize_server_target(target) if target else None
        with self._lock:
            roots = [
                key
                for key, client in self._clients.items()
                if (server_id is None or key.server_id == server_id)
                and client.transport.process.poll() is None
            ]
        stopped = self.stop(target)
        restarted = 0
        for key in roots:
            try:
                if key.server_id == "vue":
                    # Restart must rebuild the coordinated TypeScript bridge;
                    # calling _start("vue") directly would otherwise bypass
                    # the plugin/dependency setup performed by client_for().
                    self._ensure_vue_typescript(key.root)
                self._client_for_server(key.server_id, key.root)
                restarted += 1
            except LspUnavailable as error:
                logger.info(
                    "Could not restart optional LSP server %s for %s: %s",
                    key.server_id,
                    key.root,
                    error,
                )
            except LspError:
                logger.exception(
                    "Could not restart LSP server %s for %s", key.server_id, key.root
                )
        return restarted if roots else stopped

    def install(
        self,
        target: str,
        *,
        dry_run: bool = False,
    ) -> tuple[InstallResult, ...]:
        """Handle install."""
        if not target:
            raise ValueError("An LSP install target is required.")
        normalized = target.casefold()
        if normalized == "missing":
            # Include unavailable definitions even when the executable exists
            # but a mandatory bridge/runtime dependency is missing. Definitions
            # without a safe recipe are retained so the command can report them
            # under Skipped rather than silently omitting them.
            definitions = [
                definition
                for definition in SERVERS.values()
                if not self._availability(definition)[0]
            ]
        elif normalized == "all":
            # "all" means every server Citra actually knows how to install on
            # this host, not every configured server definition.
            definitions = [
                definition
                for definition in SERVERS.values()
                if candidate_for(
                    definition,
                    managers=self._available_managers(),
                )
                is not None
            ]
        else:
            definitions = [self._definition_for_target(target)]

        results: list[InstallResult] = []
        for definition in definitions:
            if (
                normalized not in {"all", "missing"}
                and self._availability(definition)[0]
            ):
                results.append(
                    InstallResult(
                        server_id=definition.id,
                        command=None,
                        dry_run=dry_run,
                        returncode=0,
                        output="already installed and available",
                        executable_found=self._which(definition.executable),
                    )
                )
                continue
            candidate = candidate_for(
                definition,
                managers=self._available_managers(),
            )
            if candidate is None:
                results.append(
                    InstallResult(
                        server_id=definition.id,
                        command=None,
                        dry_run=dry_run,
                        returncode=None,
                        output="no supported installer configured for this system",
                        executable_found=self._which(definition.executable),
                    )
                )
                continue
            if not dry_run:
                self.workspace.require_soft_capacity("env")
            result = execute_install(
                definition,
                candidate,
                dry_run=dry_run,
                sandbox=self.sandbox,
                cwd=self.workspace.workspace,
                environment=self.workspace.environment(),
                resolver=self._which,
            )
            results.append(result)
            if not dry_run:
                self.workspace.write_runtime_manifest()
        return tuple(results)

    def close(self, *, force: bool = False) -> None:
        """Handle close."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            clients = tuple(self._clients.values())
            self._clients.clear()
            self._typescript_vue_roots.clear()
        for client in clients:
            self._shutdown_client(client, force=force)

    def _format_diagnostics(
        self,
        diagnostics: list[Any],
        *,
        path: Path,
    ) -> str | None:
        """Compatibility wrapper around the subsystem's single formatter."""
        return format_diagnostics(
            diagnostics,
            path=path,
            display_path=self.workspace.display_path,
        )

    def _start(self, server_id: str, root: Path) -> LspClient:
        """Handle start."""
        if not self.config.enabled:
            raise LspUnavailable("LSP support is disabled by configuration.")
        definition = SERVERS[server_id]
        executable = self._which(definition.executable)
        if definition.id == "jdtls" and executable is not None:
            java = self._which("java")
            if java is None:
                raise LspUnavailable(
                    "jdtls is unavailable: Java 21 or newer is required."
                )
            major = self._java_major_version(java)
            if major is None:
                raise LspUnavailable(
                    "jdtls is unavailable: could not determine the Java runtime version."
                )
            if major < 21:
                raise LspUnavailable(
                    f"jdtls is unavailable: Java 21 or newer is required (found Java {major})."
                )
        if definition.id == "ruby":
            if self._which("ruby") is None:
                raise LspUnavailable(
                    "Ruby LSP is unavailable: a Ruby runtime is required."
                )
            command = self._ruby_command(root, executable)
            if command is None:
                raise LspUnavailable(
                    "No Ruby language server is installed for this project."
                )
        else:
            if executable is None:
                raise LspUnavailable(
                    f"Language server {definition.executable!r} is not installed. {definition.install_hint}"
                )
            missing = [
                dependency
                for dependency in definition.requires
                if self._which(dependency) is None
            ]
            if missing:
                raise LspUnavailable(
                    f"{definition.id} is unavailable; missing dependency: {', '.join(missing)}"
                )
            command = self._command_for(definition, executable, root)

        initialization_options = dict(definition.initialization_options)
        if server_id == "typescript" and root.resolve() in self._typescript_vue_roots:
            plugin_location = self._vue_plugin_location(root)
            if plugin_location is None:
                raise LspUnavailable(
                    "Vue TypeScript bridge is unavailable: @vue/typescript-plugin was not found."
                )
            initialization_options = {
                **initialization_options,
                "plugins": [
                    {
                        "name": "@vue/typescript-plugin",
                        "location": str(plugin_location),
                        "languages": ["vue"],
                        "configNamespace": "typescript",
                    }
                ],
            }

        server_config = ServerConfig(
            command=command,
            settings=dict(definition.settings),
            initialization_options=initialization_options,
            cold_diagnostics_timeout=definition.cold_diagnostics_timeout,
        )
        server_config, path_prepend = self._apply_interpreter_resolver(
            server_id, definition, server_config
        )
        try:
            process = self.sandbox.popen(
                server_config.command,
                cwd=root,
                network=False,
                environment=dict(server_config.environment),
                path_prepend=path_prepend,
            )
        except Exception as error:
            raise LspUnavailable(f"Could not start {server_id}: {error}") from error
        holder: dict[str, LspClient] = {}

        def handle_notification(method: str, params: Any) -> None:
            """Handle handle notification."""
            client = holder.get("client")
            if client is not None:
                client.handle_notification(method, params)

        def handle_request(method: str, params: Any) -> Any:
            """Handle handle request."""
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
        if server_id == "vue":
            self._wire_vue_client(client, root)
        try:
            client.initialize()
        except Exception as error:
            transport.close()
            self.sandbox.terminate_process(process)
            raise LspStartupError(
                f"Could not initialize {server_id}: {error}"
            ) from error

        return client

    def _wire_vue_client(self, client: LspClient, root: Path) -> None:
        """Handle wire vue client."""
        def mirror(path: Path, text: str, language: Language) -> None:
            """Handle mirror."""
            ts_client, _ = self._client_for_server("typescript", root)
            ts_client.sync_document(path, text, language)

        def bridge(command_name: str, args: Any, options: dict[str, Any]) -> Any:
            """Handle bridge."""
            ts_client, _ = self._client_for_server("typescript", root)
            return ts_client.execute_command(
                "typescript.tsserverRequest",
                [command_name, args, options],
            )

        client.set_mirror_sync(mirror)
        client.set_tsserver_bridge(bridge)

    def _ensure_vue_typescript(self, root: Path) -> None:
        """Handle ensure vue typescript."""
        if self._which("vue-language-server") is None:
            raise LspUnavailable(
                "Vue language server is unavailable: vue-language-server is not installed."
            )
        if self._which("node") is None:
            raise LspUnavailable(
                "Vue language server is unavailable: a Node.js runtime is required."
            )
        if self._which("typescript-language-server") is None:
            raise LspUnavailable(
                "Vue language server is unavailable: typescript-language-server is required for the Vue TypeScript bridge."
            )
        plugin_location = self._vue_plugin_location(root)
        if plugin_location is None:
            raise LspUnavailable(
                "Vue language server is unavailable: @vue/typescript-plugin is required for the TypeScript bridge."
            )
        root = root.resolve()
        key = ClientKey(root, "typescript")
        with self._lock:
            if root in self._typescript_vue_roots:
                return
            self._typescript_vue_roots.add(root)
            existing = self._clients.pop(key, None)
        if existing is not None:
            self._shutdown_client(existing)
        # Start now so Vue never starts without a usable bridge partner. If
        # startup/capability validation fails, roll back the marker so a later
        # request cannot incorrectly bypass the dependency check.
        try:
            ts_client, _ = self._client_for_server("typescript", root)
            provider = (ts_client.capabilities.raw or {}).get("executeCommandProvider")
            commands = (
                provider.get("commands", []) if isinstance(provider, dict) else []
            )
            if (
                not isinstance(commands, list)
                or "typescript.tsserverRequest" not in commands
            ):
                raise LspUnavailable(
                    "Vue language server is unavailable: typescript-language-server "
                    "does not support the typescript.tsserverRequest bridge command."
                )
        except Exception:
            with self._lock:
                self._typescript_vue_roots.discard(root)
                failed_client = self._clients.pop(key, None)
            if failed_client is not None:
                self._shutdown_client(failed_client)
            raise

    def _vue_plugin_location(self, root: Path | None) -> Path | None:
        """Handle vue plugin location."""
        candidates: list[Path] = []
        if root is not None:
            candidates.extend(
                [
                    root / "node_modules" / "@vue" / "typescript-plugin",
                    root / "node_modules" / "@vue" / "language-server",
                ]
            )
        vue_executable = self._which("vue-language-server")
        if vue_executable:
            resolved = Path(vue_executable).resolve()
            for parent in resolved.parents:
                if parent.name == "language-server" and parent.parent.name == "@vue":
                    candidates.extend(
                        [
                            parent,
                            parent.parent / "typescript-plugin",
                            parent / "node_modules" / "@vue" / "typescript-plugin",
                        ]
                    )
                    break
        node_path = os.environ.get("NODE_PATH")
        if node_path:
            for entry in node_path.split(os.pathsep):
                if entry:
                    base = Path(entry)
                    candidates.extend(
                        [
                            base / "@vue" / "typescript-plugin",
                            base / "@vue" / "language-server",
                        ]
                    )
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()
        return None

    def _ruby_command(
        self,
        root: Path,
        executable: str | None,
    ) -> tuple[str, ...] | None:
        """Handle ruby command."""
        bundle = self._which("bundle")
        if bundle and (root / "Gemfile").is_file():
            return (bundle, "exec", "ruby-lsp")
        if executable:
            return (executable,)
        return None

    def _command_for(
        self,
        definition: ServerDefinition,
        executable: str,
        root: Path,
    ) -> tuple[str, ...]:
        """Handle command for."""
        if definition.command_factory is not None:
            return definition.command_factory(executable, root, self.workspace.cache)
        return (executable, *definition.arguments)

    def _root_for(self, path: Path) -> Path:
        """Handle root for."""
        path = self.workspace.require_allowed_path(path)
        # Follow the same public root contract as model-facing filesystem tools.
        # Prefer the project/tmp roots, then any additional lifecycle root
        # exposed through ``allowed_roots``. This keeps @tmp as one project root
        # rather than creating a server per temporary file.
        roots = [
            self.workspace.workspace,
            self.workspace.tmp,
            self.workspace.home,
            self.workspace.cache,
            self.workspace.config,
            self.workspace.data,
            self.workspace.runtime,
        ]
        for value in self.workspace.allowed_roots:
            root = Path(value)
            if root not in roots:
                roots.append(root)
        for root in roots:
            root = root.resolve()
            try:
                path.relative_to(root)
                return root
            except ValueError:
                continue
        raise LspUnavailable(f"LSP path is outside the active workspace: {path}")

    def _dispose_client(self, key: ClientKey, client: LspClient) -> None:
        """Handle dispose client."""
        self._clients.pop(key, None)
        self._shutdown_client(client)

    def _shutdown_client(self, client: LspClient, *, force: bool = False) -> None:
        """Handle shutdown client."""
        process = client.transport.process
        if force:
            self.sandbox.terminate_process(process, force=True)
            client.transport.close()
            return
        client.close()
        try:
            process.wait(timeout=2)
        except Exception:
            self.sandbox.terminate_process(process)

    def _availability(
        self, definition: ServerDefinition
    ) -> tuple[bool, dict[str, Any]]:
        """Handle availability."""
        executable = self._which(definition.executable)
        dependencies = {
            dependency: self._which(dependency) for dependency in definition.requires
        }
        optional: dict[str, Any] = {"requirements": dependencies}
        available = executable is not None and all(dependencies.values())

        if definition.id == "jdtls":
            java = dependencies.get("java") or self._which("java")
            major = self._java_major_version(java) if java else None
            optional["java_major"] = major
            optional["java_compatible"] = major is not None and major >= 21
            available = available and bool(optional["java_compatible"])

        if definition.id == "vue":
            plugin = (
                self._vue_plugin_location(self.workspace.workspace)
                or self._vue_plugin_location(None)
            )
            optional["typescript_bridge"] = self._which("typescript-language-server")
            # status() is consumed directly by the model-facing LSP tool via
            # json.dumps(), so keep its public payload JSON-native. Internal
            # discovery helpers may use Path objects, but they must not leak
            # through status data.
            optional["vue_typescript_plugin"] = (
                str(plugin) if plugin is not None else None
            )
            available = (
                available
                and optional["typescript_bridge"] is not None
                and plugin is not None
            )

        return available, {
            "executable": executable,
            "optional_dependencies": optional,
        }

    def _apply_interpreter_resolver(
        self,
        server_id: str,
        definition: ServerDefinition,
        server_config: ServerConfig,
    ) -> tuple[ServerConfig, tuple[str, ...]]:
        """Invoke the server's interpreter resolver and merge the result.

        The resolver hook is opt-in: a ``ServerDefinition`` without an
        ``interpreter_resolver`` returns the config unchanged. When the
        resolver returns a :class:`ResolvedInterpreter`, the manager merges
        its ``environment`` (last-wins on the spawned process), ``settings``
        (deep-merge by section so e.g. ``python.pythonPath`` from the
        resolver does not clobber the existing ``python.analysis.*``
        config), ``initialization_options`` (shallow merge), and exposes the
        ``path_prepend`` list to :meth:`WorkspaceSandbox.popen` so the
        resolved toolchain is on the server's ``PATH``.

        A resolver that raises is logged at debug level and treated as "no
        interpreter" — language-server startup is never blocked by a
        resolver failure.
        """
        resolver = definition.interpreter_resolver
        if resolver is None:
            return server_config, ()

        project_root = self.workspace.workspace
        try:
            resolved: ResolvedInterpreter = resolver(
                project_root, workspace=self.workspace
            )
        except Exception as error:  # noqa: BLE001 - resolvers must never block startup
            logger.debug("interpreter resolver for %s failed: %s", server_id, error)
            return server_config, ()

        merged_environment = {**resolved.environment, **server_config.environment}
        merged_settings = _merge_settings(server_config.settings, resolved.settings)
        merged_init_options = _merge_dict(
            dict(server_config.initialization_options),
            dict(resolved.initialization_options),
        )
        new_config = replace(
            server_config,
            environment=merged_environment,
            settings=merged_settings,
            initialization_options=merged_init_options,
        )
        return new_config, tuple(resolved.path_prepend)

    def _which(self, command: str) -> str | None:
        """Resolve a command exactly as a process inside the sandbox can see it.

        Immutable host-discovered commands are resolved through the sandbox's
        runtime manifest.  Mutable dependency-environment bins are refreshed
        on every miss so ``/lsp status`` immediately observes servers installed
        during the current Citra process, including installs performed through
        an ordinary sandbox shell rather than ``/lsp install``.
        """
        path = self.sandbox.resolve_command(command)
        source = "runtime"
        if path is None:
            path = self.workspace.refresh_staged_command(command)
            source = "dependency-environment"
        if path is None:
            _activity_logger.trace(
                "Sandbox command is unavailable",
                command=command,
            )
            return None
        _activity_logger.debug(
            "Resolved sandbox command",
            command=command,
            executable=str(path),
            source=source,
        )
        return str(path)

    def _available_managers(self) -> tuple[str, ...]:
        """Handle available managers."""
        return available_managers(self._which)

    def _java_major_version(self, java: str) -> int | None:
        """Handle java major version."""
        try:
            completed = self.sandbox.run(
                [java, "-version"],
                cwd=self.workspace.workspace,
                timeout=5,
                network=False,
            )
        except Exception:
            return None
        match = re.search(
            r'version\s+"(?P<version>[0-9]+(?:\.[0-9]+)?)',
            completed.output or "",
        )
        if match is None:
            return None
        version = match.group("version")
        parts = version.split(".")
        try:
            first = int(parts[0])
            if first == 1 and len(parts) > 1:
                return int(parts[1])
            return first
        except ValueError:
            return None

    def _normalize_server_target(self, target: str) -> str:
        """Handle normalize server target."""
        return self._definition_for_target(target).id

    def _definition_for_target(self, target: str) -> ServerDefinition:
        """Handle definition for target."""
        normalized = target.casefold().strip()
        normalized = SERVER_ALIASES.get(normalized, normalized)
        definition = SERVERS.get(normalized)
        if definition is not None:
            return definition
        for language in Language:
            if normalized in {language.value, language.name.casefold()}:
                server_id = self._normalize_server_target(server_for_language(language))
                return SERVERS[server_id]
        raise ValueError(f"Unknown LSP server or language: {target}")
