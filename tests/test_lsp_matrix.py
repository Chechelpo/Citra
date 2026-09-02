from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import io
import logging
import shutil
import subprocess
import sys
import tempfile
import unittest
import types
from unittest.mock import patch

from citra.commands.lsp import LspCommand
from citra.utils.lsp.client import LspClient
from citra.utils.lsp.config import LspConfig, ServerConfig
from citra.utils.lsp.errors import LspDiagnosticsTimeout, LspUnavailable
from citra.utils.lsp.installer import available_managers, execute_install
from citra.utils.lsp.language import Language
from citra.utils.lsp.manager import ClientKey, LspManager
from citra.utils.lsp.servers import SERVERS
from citra.utils.lsp.transport import JsonRpcTransport

from tests.test_lsp_reliability import FakeFilesystem, FakeSandbox, WorkspaceStub


FAKE_SERVER = Path(__file__).with_name("fake_lsp_server.py")


def _load_transient_tool(module_name: str, class_name: str):
    """Load one transient tool without executing its eager optional-dependency package init."""
    package_name = "citra.tools._lsp_test_transient"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(Path(__file__).parents[1] / "src/citra/tools/transient")]
        sys.modules[package_name] = package
    qualified = f"{package_name}.{module_name}"
    module = sys.modules.get(qualified)
    if module is None:
        path = Path(__file__).parents[1] / "src/citra/tools/transient" / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(qualified, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    return getattr(module, class_name)


class _MutatingFilesystem:
    def execute(self, operation: str, arguments: dict[str, object]) -> str:
        path = Path(str(arguments["path"]))
        if operation == "read_raw":
            return path.read_text(encoding="utf-8")
        if operation == "write":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(arguments["content"]), encoding="utf-8")
            return "ok"
        if operation == "edit":
            text = path.read_text(encoding="utf-8")
            old = str(arguments.get("old", ""))
            new = str(arguments.get("new", ""))
            if old not in text:
                return "error: old text not found"
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
            return "ok"
        raise AssertionError(operation)


class _DummyTransport:
    def __init__(self) -> None:
        self.notifications: list[tuple[str, object]] = []

    def notify(self, method: str, params=None) -> None:
        self.notifications.append((method, params))


class LspMatrixRegressionTests(unittest.TestCase):
    def _client(self, mode: str, root: Path) -> LspClient:
        process = subprocess.Popen(
            [sys.executable, str(FAKE_SERVER), mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        holder: dict[str, LspClient] = {}
        transport = JsonRpcTransport(
            process,
            notification_handler=lambda method, params: holder["client"].handle_notification(
                method, params
            ),
            request_handler=lambda method, params: holder["client"].handle_request(method, params),
        )
        client = LspClient(
            transport,
            root=root,
            server=ServerConfig(command=("fake",)),
            config=LspConfig(
                startup_timeout=2.0,
                request_timeout=1.0,
                diagnostics_timeout=0.1,
                cold_diagnostics_timeout=0.3,
            ),
            name="fake",
        )
        holder["client"] = client
        client.initialize()
        self.addCleanup(client.close)
        return client

    def test_every_allowed_root_is_a_stable_lsp_root(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox())
            for root in workspace.allowed_roots:
                path = root / "project" / "sample.py"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x = 1\n", encoding="utf-8")
                self.assertEqual(manager._root_for(path), root)

    def test_text_document_sync_options_are_respected_without_inventing_defaults(self):
        self.assertEqual(LspClient._parse_sync({"openClose": True, "change": 2}), (2, True))
        self.assertEqual(LspClient._parse_sync({"change": 2}), (2, False))
        self.assertEqual(LspClient._parse_sync({"openClose": True}), (0, True))
        self.assertEqual(LspClient._parse_sync({}), (0, False))
        self.assertEqual(LspClient._parse_sync(2), (2, True))
        self.assertEqual(LspClient._parse_sync(0), (0, False))

    def test_stale_versioned_push_diagnostics_are_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            client = self._client("stale", root)
            path = root / "sample.py"
            uri = client.sync_document(path, "x = 1\n", Language.PYTHON)
            self.assertEqual(client.diagnostics(uri)[0]["message"], "initial")
            client.sync_document(path, "x = 2\n", Language.PYTHON)
            diagnostics = client.diagnostics(uri)
            self.assertEqual(diagnostics[0]["message"], "current")

    def test_pull_diagnostic_refresh_discards_previous_result_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            client = self._client("pull", root)
            uri = client.sync_document(root / "sample.rb", "def broken(\n", Language.RUBY)
            first = client.diagnostics(uri)
            self.assertTrue(first)
            self.assertIsNone(client.handle_request("workspace/diagnostic/refresh", {}))
            second = client.diagnostics(uri)
            self.assertEqual(first, second)

    def test_vue_tsserver_bridge_handles_direct_and_wrapped_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            transport = _DummyTransport()
            client = LspClient(
                transport,  # type: ignore[arg-type]
                root=Path(td),
                server=ServerConfig(command=("fake",)),
                config=LspConfig(),
                name="vue",
            )
            calls: list[tuple[str, object, dict[str, object]]] = []

            def bridge(command: str, args: object, options: dict[str, object]):
                calls.append((command, args, options))
                return {"body": {"ok": True}}

            client.set_tsserver_bridge(bridge)
            client.handle_notification("tsserver/request", [7, "quickinfo", {"file": "x.vue"}])
            client.handle_notification("tsserver/request", [[8, "definition", {"file": "x.vue"}]])

            self.assertEqual(calls[0][0], "quickinfo")
            self.assertEqual(calls[0][2], {"isAsync": True, "lowPriority": True})
            self.assertEqual(
                transport.notifications,
                [
                    ("tsserver/response", [7, {"ok": True}]),
                    ("tsserver/response", [[8, {"ok": True}]]),
                ],
            )

    def test_jdtls_data_directory_is_unique_per_root_and_under_cache(self):
        definition = SERVERS["jdtls"]
        factory = definition.command_factory
        self.assertIsNotNone(factory)
        assert factory is not None
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cache = base / "cache"
            root_a = base / "a"
            root_b = base / "b"
            root_a.mkdir()
            root_b.mkdir()
            command_a = factory("/usr/bin/jdtls", root_a, cache)
            command_b = factory("/usr/bin/jdtls", root_b, cache)
            data_a = Path(command_a[command_a.index("-data") + 1])
            data_b = Path(command_b[command_b.index("-data") + 1])
            data_a.relative_to(cache / "lsp" / "jdtls")
            data_b.relative_to(cache / "lsp" / "jdtls")
            self.assertNotEqual(data_a, data_b)

    def test_jdtls_requires_java_21_or_newer(self):
        with patch(
            "citra.tools.lsp.manager.subprocess.run",
            return_value=SimpleNamespace(stdout='openjdk version "17.0.12" 2024-07-16\n'),
        ):
            self.assertEqual(LspManager._java_major_version("/usr/bin/java"), 17)
        with patch(
            "citra.tools.lsp.manager.subprocess.run",
            return_value=SimpleNamespace(stdout='openjdk version "21.0.7" 2025-04-15\n'),
        ):
            self.assertEqual(LspManager._java_major_version("/usr/bin/java"), 21)

    def test_every_configured_server_is_optional_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox())
            self.addCleanup(manager.close)
            with patch("citra.tools.lsp.manager.shutil.which", return_value=None):
                for definition in SERVERS.values():
                    language = definition.languages[0]
                    extension = language.file_extensions[0]
                    path = workspace.workspace / f"missing-{definition.id}{extension}"
                    path.write_text("{}" if language is Language.JSON else "", encoding="utf-8")
                    with self.subTest(server=definition.id):
                        with self.assertRaises(LspUnavailable):
                            manager.client_for(path)
                        rendered = manager.diagnostics_for_path(
                            path.name,
                            filesystem=FakeFilesystem(),
                        )
                        self.assertIsNone(rendered)

    def test_client_handle_exposes_cold_start_then_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            path = workspace.workspace / "sample.py"
            path.write_text("x = 1\n", encoding="utf-8")
            manager = LspManager(workspace, FakeSandbox("basic"), config=LspConfig(startup_timeout=2))
            self.addCleanup(manager.close)
            real_which = shutil.which
            with patch(
                "citra.tools.lsp.manager.shutil.which",
                side_effect=lambda name: "/fake/pyright-langserver"
                if name == "pyright-langserver"
                else real_which(name),
            ):
                first = manager.client_for(path)
                second = manager.client_for(path)
            self.assertTrue(first.cold_start)
            self.assertFalse(second.cold_start)
            self.assertIs(first.client, second.client)

    def test_manager_uses_cold_timeout_only_until_first_diagnostics_succeed(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            path = workspace.workspace / "sample.py"
            manager = LspManager(
                workspace,
                FakeSandbox("cold-only"),
                config=LspConfig(
                    startup_timeout=2.0,
                    request_timeout=1.0,
                    diagnostics_timeout=0.04,
                    cold_diagnostics_timeout=0.3,
                ),
            )
            self.addCleanup(manager.close)
            real_which = shutil.which
            with patch(
                "citra.tools.lsp.manager.shutil.which",
                side_effect=lambda name: "/fake/pyright-langserver"
                if name == "pyright-langserver"
                else real_which(name),
            ):
                self.assertTrue(manager.diagnostics(path, 'x: int = "wrong"\n'))
                started = __import__("time").monotonic()
                with self.assertRaises(LspDiagnosticsTimeout):
                    manager.diagnostics(path, "x: int = 1\n")
                elapsed = __import__("time").monotonic() - started
            self.assertLess(elapsed, 0.2)

    def test_restart_vue_rebuilds_typescript_bridge_before_vue(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox())
            root = workspace.workspace.resolve()
            calls: list[tuple[str, Path]] = []

            class LiveProcess:
                def poll(self):
                    return None

            class ExistingClient:
                transport = SimpleNamespace(process=LiveProcess())

            from citra.utils.lsp.manager import ClientKey
            manager._clients[ClientKey(root, "vue")] = ExistingClient()  # type: ignore[assignment]

            with patch.object(manager, "stop", return_value=1), patch.object(
                manager, "_ensure_vue_typescript", side_effect=lambda value: calls.append(("ensure", value))
            ), patch.object(
                manager, "_client_for_server", side_effect=lambda server, value: (calls.append((server, value)) or (ExistingClient(), True))
            ):
                restarted = manager.restart("vue")

            self.assertEqual(restarted, 1)
            self.assertEqual(calls, [("ensure", root), ("vue", root)])

    def test_vue_without_typescript_bridge_is_cleanly_unavailable(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            path = workspace.workspace / "App.vue"
            path.write_text("<template><div /></template>\n", encoding="utf-8")
            manager = LspManager(workspace, FakeSandbox())
            self.addCleanup(manager.close)
            real_which = shutil.which

            def which(name: str):
                if name == "vue-language-server":
                    return "/fake/vue-language-server"
                if name == "node":
                    return real_which("node") or "/fake/node"
                return None

            with patch("citra.tools.lsp.manager.shutil.which", side_effect=which):
                with self.assertRaisesRegex(LspUnavailable, "typescript-language-server"):
                    manager.client_for(path)

    def test_vue_missing_server_does_not_start_typescript_dependency(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox())
            self.addCleanup(manager.close)
            root = workspace.workspace.resolve()

            def which(name: str):
                if name in {"node", "typescript-language-server"}:
                    return f"/fake/{name}"
                return None

            with patch("citra.tools.lsp.manager.shutil.which", side_effect=which), patch.object(
                manager, "_client_for_server"
            ) as client_for_server:
                with self.assertRaisesRegex(LspUnavailable, "vue-language-server"):
                    manager._ensure_vue_typescript(root)
            client_for_server.assert_not_called()

    def test_failed_vue_bridge_validation_discards_plugin_typescript_client(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(
                workspace,
                FakeSandbox("basic"),
                config=LspConfig(startup_timeout=1.0, request_timeout=1.0),
            )
            self.addCleanup(manager.close)
            root = workspace.workspace.resolve()

            def which(name: str):
                if name in {"vue-language-server", "typescript-language-server", "node"}:
                    return f"/fake/{name}"
                return None

            with patch("citra.tools.lsp.manager.shutil.which", side_effect=which), patch.object(
                manager, "_vue_plugin_location", return_value=root / "fake-vue-plugin"
            ):
                with self.assertRaisesRegex(LspUnavailable, "typescript.tsserverRequest"):
                    manager._ensure_vue_typescript(root)

            self.assertNotIn(root, manager._typescript_vue_roots)
            self.assertNotIn(ClientKey(root, "typescript"), manager._clients)

    def test_every_missing_server_is_graceful_through_explicit_and_mutating_tools(self):
        LspTool = _load_transient_tool("lsp", "Lsp")
        EditTool = _load_transient_tool("edit", "Edit")
        WriteTool = _load_transient_tool("write", "Write")
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            filesystem = _MutatingFilesystem()
            manager = LspManager(
                workspace,
                FakeSandbox(),
                config=LspConfig(json_fallback=False),
            )
            self.addCleanup(manager.close)
            context = SimpleNamespace(
                lsp_manager=manager,
                workspace=workspace,
                filesystem=filesystem,
                logger=logging.getLogger("lsp-optionality-test"),
            )
            context.diagnostics_for_path = lambda raw: manager.diagnostics_for_path(
                raw, filesystem=filesystem
            )
            lsp_tool = LspTool(context)
            edit_tool = EditTool(context)
            write_tool = WriteTool(context)

            with patch("citra.tools.lsp.manager.shutil.which", return_value=None):
                for definition in SERVERS.values():
                    language = definition.languages[0]
                    extension = language.file_extensions[0]
                    relative = f"optional/{definition.id}{extension}"
                    path = workspace.resolve_path(relative)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("before\n", encoding="utf-8")
                    explicit = lsp_tool._execute({"action": "diagnostics", "path": str(path)})
                    self.assertTrue(
                        explicit.startswith("unavailable:"),
                        f"{definition.id} should degrade explicitly: {explicit}",
                    )
                    written = write_tool._execute(
                        {"path": str(path), "content": "before\n", "diagnostics": True}
                    )
                    self.assertEqual(written, "ok", definition.id)
                    edited = edit_tool._execute(
                        {
                            "path": str(path),
                            "old": "before",
                            "new": "after",
                            "diagnostics": True,
                        }
                    )
                    self.assertEqual(edited, "ok", definition.id)
                    self.assertEqual(path.read_text(encoding="utf-8"), "after\n")

    def test_edit_and_write_diagnostics_run_automatically(self):
        EditTool = _load_transient_tool("edit", "Edit")
        WriteTool = _load_transient_tool("write", "Write")
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            filesystem = _MutatingFilesystem()
            calls: list[str] = []
            context = SimpleNamespace(filesystem=filesystem)
            context.diagnostics_for_path = lambda raw: calls.append(raw) or None
            edit_tool = EditTool(context)
            write_tool = WriteTool(context)
            path = workspace.workspace / "sample.py"
            path.write_text("before\n", encoding="utf-8")
            self.assertEqual(
                edit_tool._execute({"path": str(path), "old": "before", "new": "middle"}),
                "ok",
            )
            self.assertEqual(
                write_tool._execute({"path": str(path), "content": "after\n"}),
                "ok",
            )
            self.assertEqual(calls, [str(path), str(path)])

    def test_manager_close_terminates_language_server_processes(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            path = workspace.workspace / "sample.py"
            path.write_text("x = 1\n", encoding="utf-8")
            manager = LspManager(
                workspace,
                FakeSandbox("basic"),
                config=LspConfig(startup_timeout=2.0, request_timeout=1.0),
            )
            with patch(
                "citra.tools.lsp.manager.shutil.which",
                side_effect=lambda name: sys.executable if name in {"pyright-langserver", "node"} else None,
            ):
                handle = manager.client_for(path)
                process = handle.client.transport.process
                self.assertIsNone(process.poll())
                manager.close()
            self.assertEqual(manager._clients, {})
            process.wait(timeout=2)
            self.assertIsNotNone(process.poll())

    def test_package_manager_priority_prefers_pacman_before_aur_helpers(self):
        available = {"pacman", "paru", "yay", "npm"}
        with patch(
            "citra.tools.lsp.installer.shutil.which",
            side_effect=lambda name: f"/usr/bin/{name}" if name in available else None,
        ):
            managers = available_managers()
        self.assertLess(managers.index("pacman"), managers.index("paru"))
        self.assertLess(managers.index("pacman"), managers.index("yay"))
        self.assertLess(managers.index("yay"), managers.index("npm"))

    def test_install_missing_excludes_servers_that_are_already_available(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox())
            real_availability = manager._availability

            def availability(definition):
                if definition.id == "pyright":
                    return True, "/usr/bin/pyright-langserver", {}
                return real_availability(definition)

            with patch.object(manager, "_availability", side_effect=availability), patch(
                "citra.tools.lsp.manager.shutil.which", return_value=None
            ), patch(
                "citra.tools.lsp.installer.shutil.which",
                side_effect=lambda name: "/usr/bin/npm" if name == "npm" else None,
            ):
                results = manager.install("missing", dry_run=True)
            self.assertNotIn("pyright", {result.server_id for result in results})

    def test_install_all_selects_only_recipes_available_on_host(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox())
            with patch("citra.tools.lsp.manager.shutil.which", return_value=None), patch(
                "citra.tools.lsp.installer.shutil.which",
                side_effect=lambda name: "/usr/bin/npm" if name == "npm" else None,
            ):
                results = manager.install("all", dry_run=True)
            selected = {result.server_id for result in results}
            self.assertEqual(
                selected,
                {"pyright", "typescript", "vue", "json", "css", "html", "yaml", "bash"},
            )
            self.assertTrue(all(result.dry_run for result in results))

    def test_known_go_and_cargo_install_recipes_are_exposed(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox())
            with patch("citra.tools.lsp.manager.shutil.which", return_value=None), patch(
                "citra.tools.lsp.installer.shutil.which",
                side_effect=lambda name: f"/usr/bin/{name}" if name in {"go", "cargo"} else None,
            ):
                sql = manager.install("sql", dry_run=True)[0]
                taplo = manager.install("taplo", dry_run=True)[0]
            self.assertEqual(sql.command, ("go", "install", "github.com/sqls-server/sqls@latest"))
            self.assertEqual(taplo.command, ("cargo", "install", "taplo-cli", "--locked"))

    def test_install_without_supported_package_manager_is_cleanly_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox())
            with patch("citra.tools.lsp.manager.shutil.which", return_value=None), patch(
                "citra.tools.lsp.installer.shutil.which", return_value=None
            ):
                result = manager.install("pyright", dry_run=True)[0]
            self.assertIsNone(result.command)
            self.assertIsNone(result.returncode)
            self.assertIn("no supported installer", result.output)

    def test_unknown_install_target_is_rejected_without_execution(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox())
            with patch("citra.tools.lsp.installer.subprocess.Popen") as popen:
                with self.assertRaisesRegex(ValueError, "Unknown LSP server or language"):
                    manager.install("definitely-not-a-language-server")
            popen.assert_not_called()

    def test_failed_installer_returns_status_and_does_not_raise(self):
        definition = SERVERS["pyright"]
        candidate = next(item for item in definition.install_candidates if item.manager == "npm")

        class FailedProcess:
            stdout = io.StringIO("npm failed\n")

            def wait(self):
                return 9

        with patch("citra.tools.lsp.installer.subprocess.Popen", return_value=FailedProcess()), patch(
            "citra.tools.lsp.installer.shutil.which", return_value=None
        ), patch("builtins.print") as printed:
            result = execute_install(definition, candidate, dry_run=False)

        self.assertEqual(result.returncode, 9)
        self.assertFalse(result.success)
        self.assertIn("npm failed", result.output)
        self.assertIsNone(result.executable_found)
        printed.assert_called_once_with("$ " + " ".join(candidate.command), flush=True)

    def test_successful_installer_rechecks_expected_executable(self):
        definition = SERVERS["pyright"]
        candidate = next(item for item in definition.install_candidates if item.manager == "npm")

        class SuccessfulProcess:
            stdout = io.StringIO("installed\n")

            def wait(self):
                return 0

        with patch("citra.tools.lsp.installer.subprocess.Popen", return_value=SuccessfulProcess()), patch(
            "citra.tools.lsp.installer.shutil.which",
            side_effect=lambda name: "/usr/local/bin/pyright-langserver"
            if name == "pyright-langserver"
            else None,
        ), patch("builtins.print"):
            result = execute_install(definition, candidate, dry_run=False)

        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.success)
        self.assertEqual(result.executable_found, "/usr/local/bin/pyright-langserver")

    def test_lsp_command_stop_and_restart_manage_running_instances(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            path = workspace.workspace / "sample.py"
            path.write_text("x = 1\n", encoding="utf-8")
            manager = LspManager(
                workspace,
                FakeSandbox("basic"),
                config=LspConfig(startup_timeout=2.0, request_timeout=1.0),
            )
            self.addCleanup(manager.close)
            command = LspCommand(SimpleNamespace(lsp_manager=manager))  # type: ignore[arg-type]
            which = lambda name: sys.executable if name in {"pyright-langserver", "node"} else None
            with patch("citra.tools.lsp.manager.shutil.which", side_effect=which):
                first = manager.client_for(path).client.transport.process
                stopped = command.run("stop pyright")
                self.assertIn("stopped: 1", stopped.output)
                self.assertIsNotNone(first.poll())

                second = manager.client_for(path).client.transport.process
                restarted = command.run("restart pyright")
                self.assertIn("restarted: 1", restarted.output)
                running = manager.client_for(path).client.transport.process
                self.assertIsNot(second, running)
                self.assertIsNotNone(second.poll())
                self.assertIsNone(running.poll())

    def test_lsp_command_defaults_to_status_and_dry_run_never_executes(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox())
            context = SimpleNamespace(lsp_manager=manager)
            command = LspCommand(context)  # type: ignore[arg-type]

            with patch("citra.tools.lsp.manager.shutil.which", return_value=None), patch(
                "citra.tools.lsp.installer.shutil.which",
                side_effect=lambda name: "/usr/bin/npm" if name == "npm" else None,
            ):
                status = command.run("")
                with patch("citra.tools.lsp.installer.subprocess.Popen") as popen:
                    install = command.run("install pyright --dry-run")

            self.assertIn("LSP: enabled", status.output)
            self.assertIn("pyright", status.output)
            self.assertIn("dry-run: not executed", install.output)
            popen.assert_not_called()

    def test_model_lsp_status_ignores_stale_path_arguments(self):
        LspTool = _load_transient_tool("lsp", "Lsp")
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox())
            context = SimpleNamespace(
                lsp_manager=manager,
                workspace=workspace,
                filesystem=FakeFilesystem(),
                logger=logging.getLogger(__name__),
            )
            tool = LspTool(context)
            with patch("citra.tools.lsp.manager.shutil.which", return_value=None):
                result = tool.execute(
                    {
                        "action": "status",
                        "path": "src/citra/utils/lsp/language.py",
                        "line": 1,
                        "character": 1,
                    }
                )
            self.assertIn('"enabled"', result)
            self.assertIn('"servers"', result)

    def test_json_fallback_is_used_only_for_plain_json(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox())
            self.addCleanup(manager.close)
            bad_json = workspace.workspace / "bad.json"
            bad_json.write_text('{"foo": }', encoding="utf-8")
            bad_jsonc = workspace.workspace / "bad.jsonc"
            bad_jsonc.write_text('{"foo": }', encoding="utf-8")
            with patch("citra.tools.lsp.manager.shutil.which", return_value=None):
                rendered = manager.diagnostics_for_path("bad.json", filesystem=FakeFilesystem())
                self.assertIn("error [json]", rendered or "")
                self.assertIsNone(
                    manager.diagnostics_for_path("bad.jsonc", filesystem=FakeFilesystem())
                )

    def test_lsp_disabled_does_not_run_json_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(
                workspace,
                FakeSandbox(),
                config=LspConfig(enabled=False, json_fallback=True),
            )
            bad_json = workspace.workspace / "bad.json"
            bad_json.write_text('{"foo": }', encoding="utf-8")
            self.assertIsNone(
                manager.diagnostics_for_path("bad.json", filesystem=FakeFilesystem())
            )
            with self.assertRaisesRegex(LspUnavailable, "disabled"):
                manager.diagnostics(bad_json, bad_json.read_text(encoding="utf-8"))


class RealClangdIntegrationTests(unittest.TestCase):
    def test_real_clangd_reports_invalid_and_clears_valid_c(self):
        executable = shutil.which("clangd")
        if executable is None:
            self.skipTest("clangd is not installed")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            process = subprocess.Popen(
                [executable],
                cwd=root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            holder: dict[str, LspClient] = {}
            transport = JsonRpcTransport(
                process,
                notification_handler=lambda method, params: holder["client"].handle_notification(
                    method, params
                ),
                request_handler=lambda method, params: holder["client"].handle_request(method, params),
            )
            client = LspClient(
                transport,
                root=root,
                server=ServerConfig(command=(executable,)),
                config=LspConfig(
                    startup_timeout=10,
                    request_timeout=10,
                    diagnostics_timeout=5,
                    cold_diagnostics_timeout=10,
                ),
                name="clangd",
            )
            holder["client"] = client
            try:
                client.initialize()
                path = root / "sample.c"
                uri = client.sync_document(path, "int main( { return 0; }\n", Language.C)
                self.assertTrue(client.diagnostics(uri))
                client.sync_document(path, "int main(void) { return 0; }\n", Language.C)
                self.assertEqual(client.diagnostics(uri), [])
            finally:
                client.close()
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
