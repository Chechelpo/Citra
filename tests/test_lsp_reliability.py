from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from citra.utils.lsp.client import LspClient, configuration_for_section
from citra.utils.lsp.config import LspConfig, ServerConfig
from citra.utils.lsp.diagnostics import json_fallback_diagnostics
from citra.utils.lsp.language import Language, detect_language, server_for_language
from citra.utils.lsp.installer import candidate_for, execute_install
from citra.utils.lsp.manager import LspManager
from citra.utils.lsp.servers import SERVERS
from citra.utils.lsp.transport import JsonRpcTransport


HERE = Path(__file__).resolve().parent
FAKE_SERVER = HERE / "fake_lsp_server.py"


class FakeSandbox:
    def __init__(self, mode: str = "configuration") -> None:
        self.mode = mode

    def popen(self, command, *, cwd=None, network=False, environment=None):
        del command, network, environment
        return subprocess.Popen(
            [sys.executable, str(FAKE_SERVER), self.mode],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def terminate_process(self, process) -> None:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)


class WorkspaceStub:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source_workspace = root / "source"
        self.workspace = root / "workspace"
        self.home = root / "home"
        self.tmp = root / "tmp"
        self.cache = root / "cache"
        self.config = root / "config"
        self.data = root / "data"
        self.runtime = root / "runtime"
        self.library = root / "library"
        for value in self.allowed_roots + (self.library,):
            value.mkdir(parents=True, exist_ok=True)

    @property
    def allowed_roots(self):
        return (
            self.source_workspace,
            self.workspace,
            self.home,
            self.tmp,
            self.cache,
            self.config,
            self.data,
            self.runtime,
        )

    def require_allowed_path(self, path):
        resolved = Path(path).resolve()
        for root in self.allowed_roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                pass
        raise ValueError("outside")

    def resolve_path(self, value):
        raw = str(value)
        if raw.startswith("@tmp/"):
            return self.require_allowed_path(self.tmp / raw[5:])
        if raw.startswith("./"):
            return self.require_allowed_path(self.workspace / raw[2:])
        return self.require_allowed_path(self.workspace / raw)

    def display_path(self, path):
        path = Path(path).resolve()
        for name, root in (("tmp", self.tmp), ("source", self.source_workspace)):
            try:
                return f"@{name}/{path.relative_to(root).as_posix()}"
            except ValueError:
                pass
        return path.relative_to(self.workspace).as_posix()


class FakeFilesystem:
    def execute(self, operation, arguments):
        assert operation == "read_raw"
        return Path(arguments["path"]).read_text()


class LspReliabilityTests(unittest.TestCase):
    def _client(self, mode: str, root: Path, server: ServerConfig | None = None) -> LspClient:
        process = subprocess.Popen(
            [sys.executable, str(FAKE_SERVER), mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        holder = {}
        transport = JsonRpcTransport(
            process,
            notification_handler=lambda m, p: holder["client"].handle_notification(m, p),
            request_handler=lambda m, p: holder["client"].handle_request(m, p),
        )
        client = LspClient(
            transport,
            root=root,
            server=server or ServerConfig(command=("fake",)),
            config=LspConfig(
                startup_timeout=1.0,
                request_timeout=1.0,
                diagnostics_timeout=0.03,
                cold_diagnostics_timeout=0.25,
            ),
            name="fake",
        )
        holder["client"] = client
        client.initialize()
        self.addCleanup(client.close)
        return client

    def test_configuration_is_section_aware(self):
        settings = {"python": {"analysis": {"diagnosticMode": "openFilesOnly"}}, "pyright": {}}
        self.assertEqual(configuration_for_section(settings, "python.analysis"), settings["python"]["analysis"])
        self.assertEqual(configuration_for_section(settings, "pyright"), {})
        self.assertIsNone(configuration_for_section(settings, "missing"))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            client = self._client(
                "configuration",
                root,
                ServerConfig(command=("fake",), settings=settings),
            )
            uri = client.sync_document(root / "bad.py", 'x: int = "wrong"\n', Language.PYTHON)
            result = client.diagnostics(uri)
            self.assertEqual(result[0]["message"], "configuration ok")

    def test_cold_then_warm_push_diagnostics_and_incremental_sync(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            client = self._client("push", root)
            path = root / "bad.py"
            uri = client.sync_document(path, 'x: int = "wrong"\n', Language.PYTHON)
            self.assertEqual(client.diagnostics(uri)[0]["message"], "cold push")
            client.sync_document(path, "x: int = 1\n", Language.PYTHON)
            self.assertEqual(client.diagnostics(uri)[0]["message"], "warm push")

    def test_automatic_diagnostics_timeout_is_advisory(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            path = workspace.workspace / "sample.py"
            path.write_text("x: int = 1\n")
            manager = LspManager(
                workspace,
                FakeSandbox("cold-only"),
                config=LspConfig(
                    startup_timeout=1.0,
                    request_timeout=1.0,
                    diagnostics_timeout=0.03,
                    cold_diagnostics_timeout=0.25,
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
                # First/open diagnostics arrive, but this fake intentionally
                # publishes nothing after didChange. Automatic diagnostics are
                # advisory and should therefore degrade to no result, not leak
                # LspDiagnosticsTimeout into ExecutionContext logging.
                first = manager.diagnostics_for_path("sample.py", filesystem=FakeFilesystem())
                self.assertIn("cold push", first or "")
                path.write_text('x: int = "wrong"\n')
                self.assertIsNone(
                    manager.diagnostics_for_path("sample.py", filesystem=FakeFilesystem())
                )

    def test_pull_full_then_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            client = self._client("pull", root)
            uri = client.sync_document(root / "bad.rb", "def x(\n", Language.RUBY)
            first = client.diagnostics(uri)
            second = client.diagnostics(uri)
            self.assertEqual(first, second)
            self.assertEqual(first[0]["source"], "fake-pull")

    def test_language_catalog(self):
        expected = {
            Language.PYTHON, Language.JAVASCRIPT, Language.TYPESCRIPT, Language.VUE,
            Language.JAVA, Language.RUBY, Language.JSON, Language.JSONC, Language.CSS,
            Language.SCSS, Language.LESS, Language.SQL, Language.HTML, Language.YAML,
            Language.BASH, Language.C, Language.CPP, Language.GO, Language.RUST,
            Language.LUA, Language.TOML,
        }
        self.assertTrue(expected.issubset(set(Language)))
        self.assertEqual(detect_language("x.scss"), Language.SCSS)
        self.assertEqual(server_for_language(Language.JSONC), "json")

    def test_tmp_root_and_pyright_configuration_through_manager(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            path = workspace.tmp / "lsp_test" / "test.py"
            path.parent.mkdir(parents=True)
            path.write_text('x: int = "wrong"\n')
            manager = LspManager(
                workspace,
                FakeSandbox("configuration"),
                config=LspConfig(startup_timeout=1, request_timeout=1, diagnostics_timeout=.1, cold_diagnostics_timeout=.3),
            )
            self.addCleanup(manager.close)
            real_which = shutil.which
            with patch("citra.tools.lsp.manager.shutil.which", side_effect=lambda name: "/fake/pyright-langserver" if name == "pyright-langserver" else real_which(name)):
                rendered = manager.diagnostics_for_path("@tmp/lsp_test/test.py", filesystem=FakeFilesystem())
            self.assertIn("@tmp/lsp_test/test.py", rendered or "")
            self.assertIn("configuration ok", rendered or "")

    def test_dead_server_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            path = workspace.workspace / "test.py"
            path.write_text("x=1\n")
            manager = LspManager(workspace, FakeSandbox("push"), config=LspConfig(startup_timeout=2))
            self.addCleanup(manager.close)
            real_which = shutil.which
            with patch("citra.tools.lsp.manager.shutil.which", side_effect=lambda name: "/fake/pyright-langserver" if name == "pyright-langserver" else real_which(name)):
                first = manager.client_for(path).client
                first.transport.process.kill()
                first.transport.process.wait(timeout=2)
                second = manager.client_for(path).client
            self.assertIsNot(first, second)
            self.assertNotEqual(first.transport.process.pid, second.transport.process.pid)

    def test_json_fallback(self):
        self.assertEqual(json_fallback_diagnostics('{"ok": 1}'), [])
        diagnostic = json_fallback_diagnostics('{"foo": }')[0]
        self.assertEqual(diagnostic["source"], "json")
        self.assertEqual(diagnostic["severity"], 1)


    def test_dynamic_registration_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            client = self._client("dynamic", Path(td))
            deadline = time.monotonic() + 1.0
            while not client.capabilities.diagnostics_pull and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(client.capabilities.diagnostics_pull)

    def test_diagnostics_switches_to_late_dynamic_pull_registration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            client = self._client("dynamic-pull-delayed", root)
            uri = client.sync_document(root / "sample.py", 'x: int = "wrong"\n', Language.PYTHON)
            started = time.monotonic()
            diagnostics = client.diagnostics(uri, timeout=0.8)
            self.assertEqual(diagnostics, [])
            self.assertTrue(client.capabilities.diagnostics_pull)
            self.assertLess(time.monotonic() - started, 0.7)


    def test_dynamic_pull_replacement_keeps_new_registration_active(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            client = self._client("basic", root)
            old = {
                "id": "diag-old",
                "method": "textDocument/diagnostic",
                "registerOptions": {"identifier": "old"},
            }
            new = {
                "id": "diag-new",
                "method": "textDocument/diagnostic",
                "registerOptions": {"identifier": "new"},
            }
            client.handle_request("client/registerCapability", {"registrations": [old]})
            client.handle_request("client/registerCapability", {"registrations": [new]})
            # Pyright's DynamicFeature replacement sequence disposes the old
            # registration only after the replacement has been registered.
            client.handle_request(
                "client/unregisterCapability",
                {"unregistrations": [{"id": "diag-old", "method": "textDocument/diagnostic"}]},
            )
            self.assertTrue(client.capabilities.diagnostics_pull)
            self.assertEqual(client._diagnostic_registration_options, {"identifier": "new"})

            path = root / "sample.py"
            uri = client.sync_document(path, "x: int = 1\n", Language.PYTHON)
            self.assertEqual(client.diagnostics(uri), [])
            client.sync_document(path, 'x: int = "wrong"\n', Language.PYTHON)
            # Warm diagnostics must remain pull-based after the replacement.
            self.assertEqual(client.diagnostics(uri), [])

            client.handle_request(
                "client/unregisterCapability",
                {"unregistrations": [{"id": "diag-new", "method": "textDocument/diagnostic"}]},
            )
            self.assertFalse(client.capabilities.diagnostics_pull)

    def test_pyright_style_dynamic_replacement_over_protocol_stays_pull(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            client = self._client("dynamic-replace", root)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if set(client._dynamic_registrations) == {"diag-new"}:
                    break
                time.sleep(0.01)
            self.assertEqual(set(client._dynamic_registrations), {"diag-new"})
            self.assertTrue(client.capabilities.diagnostics_pull)
            self.assertEqual(
                client._diagnostic_registration_options,
                {"identifier": "fake-new"},
            )

            path = root / "sample.py"
            uri = client.sync_document(path, "x: int = 1\n", Language.PYTHON)
            self.assertEqual(client.diagnostics(uri), [])
            client.sync_document(path, 'x: int = "wrong"\n', Language.PYTHON)
            self.assertEqual(client.diagnostics(uri), [])

    def test_arch_recipe_precedes_ecosystem_fallback_and_dry_run_is_non_mutating(self):
        definition = SERVERS["pyright"]
        with patch("citra.tools.lsp.installer.shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name in {"pacman", "npm"} else None):
            candidate = candidate_for(definition)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.manager, "pacman")
        with patch("citra.tools.lsp.installer.subprocess.Popen") as popen, patch(
            "citra.tools.lsp.installer.shutil.which", return_value=None
        ):
            result = execute_install(definition, candidate, dry_run=True)
        popen.assert_not_called()
        self.assertTrue(result.dry_run)
        self.assertIn("pacman", result.output)

    def test_install_missing_skips_servers_without_safe_recipe(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox(), config=LspConfig())
            with patch("citra.tools.lsp.manager.shutil.which", return_value=None), patch(
                "citra.tools.lsp.installer.shutil.which", side_effect=lambda name: "/usr/bin/npm" if name == "npm" else None
            ):
                results = manager.install("missing", dry_run=True)
            selected = {result.server_id: result for result in results}
            self.assertIn("pyright", selected)
            self.assertTrue(selected["pyright"].dry_run)
            self.assertIsNone(selected["jdtls"].command)
            self.assertIn("no supported installer", selected["jdtls"].output)

    def test_every_server_status_is_safe_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox(), config=LspConfig())
            with patch("citra.tools.lsp.manager.shutil.which", return_value=None), patch(
                "citra.tools.lsp.installer.shutil.which", return_value=None
            ):
                status = manager.status()
            self.assertEqual(len(status["servers"]), len(SERVERS))
            self.assertTrue(all(not item["installed"] for item in status["servers"]))

    def test_status_payload_is_json_serializable_with_vue_plugin_path(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = LspManager(workspace, FakeSandbox(), config=LspConfig())
            plugin = workspace.workspace / "node_modules" / "@vue" / "typescript-plugin"
            plugin.mkdir(parents=True)

            def which(name: str):
                if name in {"vue-language-server", "typescript-language-server", "node"}:
                    return f"/usr/bin/{name}"
                return None

            with patch("citra.tools.lsp.manager.shutil.which", side_effect=which), patch(
                "citra.tools.lsp.installer.shutil.which", return_value=None
            ):
                status = manager.status()

            # Regression: _vue_plugin_location() returns a Path. The public
            # status structure must normalize it before the transient LSP tool
            # passes the payload to json.dumps().
            encoded = json.dumps(status)
            self.assertIn(str(plugin.resolve()), encoded)


class RealServerIntegrationTests(unittest.TestCase):
    def _real_client(self, executable: str, args: tuple[str, ...], root: Path, server: ServerConfig) -> LspClient:
        path = shutil.which(executable)
        if path is None:
            self.skipTest(f"{executable} is not installed")
        process = subprocess.Popen(
            [path, *args],
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        holder = {}
        transport = JsonRpcTransport(
            process,
            notification_handler=lambda m, p: holder["client"].handle_notification(m, p),
            request_handler=lambda m, p: holder["client"].handle_request(m, p),
        )
        client = LspClient(
            transport,
            root=root,
            server=server,
            config=LspConfig(startup_timeout=30, request_timeout=15, diagnostics_timeout=10, cold_diagnostics_timeout=45),
            name=executable,
        )
        holder["client"] = client
        client.initialize()
        self.addCleanup(client.close)
        return client

    def test_real_pyright_diagnostics_if_installed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            definition = SERVERS["pyright"]
            client = self._real_client(
                definition.executable,
                definition.arguments,
                root,
                ServerConfig(command=(definition.executable, *definition.arguments), settings=definition.settings),
            )
            bad = root / "bad.py"
            uri = client.sync_document(bad, 'x: int = "wrong"\n', Language.PYTHON)
            self.assertTrue(client.diagnostics(uri))
            good = root / "good.py"
            good_uri = client.sync_document(good, "x: int = 1\n", Language.PYTHON)
            self.assertEqual(client.diagnostics(good_uri), [])

    def test_real_typescript_diagnostics_if_installed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            definition = SERVERS["typescript"]
            client = self._real_client(
                definition.executable,
                definition.arguments,
                root,
                ServerConfig(command=(definition.executable, *definition.arguments)),
            )
            bad = root / "bad.ts"
            uri = client.sync_document(bad, 'const x: number = "wrong";\n', Language.TYPESCRIPT)
            self.assertTrue(client.diagnostics(uri))
            good = root / "good.ts"
            good_uri = client.sync_document(good, "const x: number = 1;\n", Language.TYPESCRIPT)
            self.assertEqual(client.diagnostics(good_uri), [])


if __name__ == "__main__":
    unittest.main()
