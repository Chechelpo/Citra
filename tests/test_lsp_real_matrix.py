from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from citra.utils.lsp.config import LspConfig
from citra.utils.lsp.language import Language
from citra.utils.lsp.manager import LspManager
from citra.utils.lsp.positions import SourcePosition
from citra.utils.lsp.servers import SERVERS

from tests.test_lsp_reliability import WorkspaceStub


class HostSandbox:
    """Small host subprocess adapter used only by opt-in real-server tests."""

    def popen(self, command, *, cwd=None, network=False, environment=None):
        del network
        env = os.environ.copy()
        env.update(environment or {})
        return subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def terminate_process(self, process) -> None:
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


class RealLspMatrixTests(unittest.TestCase):
    """Integration checks that prove useful protocol behavior, not mere startup."""

    def _manager(self, workspace: WorkspaceStub, *, cold: float = 45.0) -> LspManager:
        manager = LspManager(
            workspace,
            HostSandbox(),  # type: ignore[arg-type]
            config=LspConfig(
                startup_timeout=30.0,
                request_timeout=15.0,
                diagnostics_timeout=10.0,
                cold_diagnostics_timeout=cold,
            ),
        )
        self.addCleanup(manager.close)
        return manager

    def _require(self, *executables: str) -> None:
        missing = [name for name in executables if shutil.which(name) is None]
        if missing:
            self.skipTest("missing optional language server dependency: " + ", ".join(missing))

    def _assert_bad_then_good(
        self,
        *,
        executable: str,
        extension: str,
        language: Language,
        bad: str,
        good: str,
        cold: float = 45.0,
    ) -> None:
        self._require(executable)
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = self._manager(workspace, cold=cold)
            path = workspace.workspace / f"sample{extension}"
            path.write_text(bad, encoding="utf-8")
            first = manager.diagnostics(path, bad)
            self.assertTrue(first, f"{executable} returned no diagnostic for deliberate invalid {language.value}")
            path.write_text(good, encoding="utf-8")
            second = manager.diagnostics(path, good)
            self.assertEqual(second, [], f"{executable} retained diagnostics after the file became valid")

    def test_real_pyright_diagnostics_under_tmp_if_installed(self):
        self._require("pyright-langserver")
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = self._manager(workspace)
            path = workspace.tmp / "lsp_test" / "sample.py"
            path.parent.mkdir(parents=True)
            bad = 'x: int = "wrong"\n'
            path.write_text(bad, encoding="utf-8")
            self.assertTrue(manager.diagnostics(path, bad))
            good = "x: int = 1\n"
            path.write_text(good, encoding="utf-8")
            self.assertEqual(manager.diagnostics(path, good), [])
            self.assertEqual(manager._root_for(path), workspace.tmp)

    def test_real_typescript_diagnostics_under_tmp_if_installed(self):
        self._require("typescript-language-server")
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = self._manager(workspace)
            path = workspace.tmp / "lsp_test" / "sample.ts"
            path.parent.mkdir(parents=True)
            bad = 'const x: number = "wrong";\n'
            path.write_text(bad, encoding="utf-8")
            self.assertTrue(manager.diagnostics(path, bad))
            good = "const x: number = 1;\n"
            path.write_text(good, encoding="utf-8")
            self.assertEqual(manager.diagnostics(path, good), [])
            self.assertEqual(manager._root_for(path), workspace.tmp)

    def test_real_json_language_server_if_installed(self):
        self._assert_bad_then_good(
            executable="vscode-json-language-server",
            extension=".json",
            language=Language.JSON,
            bad='{"foo":\n}\n',
            good='{"foo": 1}\n',
        )

    def test_real_css_diagnostics_if_installed(self):
        self._assert_bad_then_good(
            executable="vscode-css-language-server",
            extension=".css",
            language=Language.CSS,
            bad="a { color: red;\n",
            good="a { color: red; }\n",
        )

    def test_real_yaml_diagnostics_if_installed(self):
        self._assert_bad_then_good(
            executable="yaml-language-server",
            extension=".yaml",
            language=Language.YAML,
            bad="foo: [1, 2\n",
            good="foo: [1, 2]\n",
        )

    def test_real_ruby_lsp_syntax_diagnostics_if_installed(self):
        self._assert_bad_then_good(
            executable="ruby-lsp",
            extension=".rb",
            language=Language.RUBY,
            bad="def broken(\nend\n",
            good="def okay\nend\n",
        )

    def test_real_gopls_diagnostics_if_installed(self):
        self._assert_bad_then_good(
            executable="gopls",
            extension=".go",
            language=Language.GO,
            bad="package main\nfunc main( {\n",
            good="package main\nfunc main() {}\n",
        )

    def test_real_rust_analyzer_diagnostics_if_installed(self):
        self._assert_bad_then_good(
            executable="rust-analyzer",
            extension=".rs",
            language=Language.RUST,
            bad="fn main( {\n",
            good="fn main() {}\n",
        )

    def test_real_lua_diagnostics_if_installed(self):
        self._assert_bad_then_good(
            executable="lua-language-server",
            extension=".lua",
            language=Language.LUA,
            bad="local = 1\n",
            good="local x = 1\n",
        )

    def test_real_taplo_diagnostics_if_installed(self):
        self._assert_bad_then_good(
            executable="taplo",
            extension=".toml",
            language=Language.TOML,
            bad="values = [1,\n",
            good="values = [1]\n",
        )

    def test_real_html_navigation_if_installed(self):
        self._require("vscode-html-language-server")
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = self._manager(workspace)
            text = '<html><body><div id="target"></div></body></html>\n'
            path = workspace.workspace / "index.html"
            path.write_text(text, encoding="utf-8")
            handle = manager.client_for(path)
            uri = handle.client.sync_document(path, text, handle.language)
            self.assertTrue(handle.client.capabilities.document_symbols)
            symbols = handle.client.document_symbols(uri)
            self.assertIsInstance(symbols, list)

    def test_real_bash_navigation_if_installed(self):
        self._require("bash-language-server")
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = self._manager(workspace)
            text = "function hello() { echo hi; }\nhello\n"
            path = workspace.workspace / "sample.sh"
            path.write_text(text, encoding="utf-8")
            handle = manager.client_for(path)
            uri = handle.client.sync_document(path, text, handle.language)
            self.assertTrue(handle.client.capabilities.document_symbols)
            symbols = handle.client.document_symbols(uri)
            self.assertIsInstance(symbols, list)

    def test_real_sqls_without_database_does_not_fail_if_installed(self):
        self._require("sqls")
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = self._manager(workspace)
            text = "SELECT 1;\n"
            path = workspace.workspace / "query.sql"
            path.write_text(text, encoding="utf-8")
            handle = manager.client_for(path)
            uri = handle.client.sync_document(path, text, handle.language)
            # SQLS navigation should remain usable without forcing Citra to
            # provide database credentials. We only require a prompt response;
            # an empty/null hover is a legitimate answer without schema data.
            if handle.client.capabilities.hover:
                handle.client.hover(uri, SourcePosition(line=0, character=1))
            elif handle.client.capabilities.document_symbols:
                handle.client.document_symbols(uri)
            else:
                self.skipTest("installed sqls exposes no deterministic offline navigation capability")
            self.assertIsNone(handle.client.transport.process.poll())

    def test_real_vue_semantic_diagnostics_use_typescript_bridge_if_available(self):
        self._require("vue-language-server", "typescript-language-server")
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = self._manager(workspace)
            vue_definition = SERVERS["vue"]
            available = next(
                item for item in manager.status()["servers"] if item["id"] == vue_definition.id
            )["available"]
            if not available:
                self.skipTest("Vue TypeScript plugin/bridge dependency is unavailable")
            text = (
                '<script setup lang="ts">\n'
                'const x: number = "wrong";\n'
                '</script>\n'
                '<template><div>{{ x }}</div></template>\n'
            )
            path = workspace.workspace / "App.vue"
            path.write_text(text, encoding="utf-8")
            diagnostics = manager.diagnostics(path, text)
            self.assertTrue(diagnostics, "Vue semantic type error produced no diagnostic through TS bridge")

    def test_real_jdtls_diagnostics_and_cache_data_dir_if_available(self):
        self._require("jdtls", "java")
        java = shutil.which("java")
        assert java is not None
        if (LspManager._java_major_version(java) or 0) < 21:
            self.skipTest("JDTLS requires Java 21+")
        with tempfile.TemporaryDirectory() as td:
            workspace = WorkspaceStub(Path(td))
            manager = self._manager(workspace, cold=90.0)
            text = "public class Main { public static void main(String[] args) { int x = ; } }\n"
            path = workspace.workspace / "Main.java"
            path.write_text(text, encoding="utf-8")
            diagnostics = manager.diagnostics(path, text)
            self.assertTrue(diagnostics, "JDTLS returned no diagnostic for deliberate Java syntax error")
            handle = manager.client_for(path)
            command = handle.client.server.command
            self.assertIn("-data", command)
            data_dir = Path(command[command.index("-data") + 1]).resolve()
            data_dir.relative_to((workspace.cache / "lsp" / "jdtls").resolve())


if __name__ == "__main__":
    unittest.main()
