"""Independent verification driver for the Vue TypeScript bridge fix.

Scopes the test to bridge-detection behaviour:
- A1: status() reports vue_typescript_plugin; /lsp status formatter omits "missing".
- A2: _vue_plugin_location returns the path of @vue/typescript-plugin.
- A4: workspace-local hit still wins; payload is JSON-serializable.
- A5: host bind target + sandbox-writable staged-env npm prefix.
- Negative case: nothing installed -> reported missing; LspUnavailable raised with
  the @vue/typescript-plugin message when the executable and node and
  typescript-language-server are present but the plugin is missing.

Pre-existing test-infrastructure issues are sidestepped by mocking
LspManager._which directly (the test file's citra.tools.lsp.manager.shutil.which
patch path no longer exists in the codebase).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path("/mnt/storage/citra_processes/citra-process-55359-9z9pmhm0/workspace")
os.environ.setdefault("CITRA_ROOT", str(ROOT / ".citra"))
os.environ.setdefault("CITRA_INSTALL_ROOT", str(ROOT))
os.environ.setdefault("CITRA_CONFIG_PATH", str(ROOT / ".citra/config"))
os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))
os.environ.setdefault("PYTHONNOUSERSITE", "1")
sys.path.insert(0, str(ROOT / "src"))

from citra.commands.lsp import LspCommand  # noqa: E402
from citra.utils.lsp.config import LspConfig  # noqa: E402
from citra.utils.lsp.manager import LspManager  # noqa: E402
from citra.utils.lsp.errors import LspUnavailable  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from test_lsp_reliability import FakeSandbox, WorkspaceStub  # noqa: E402


def make_which(names):
    def _which(self, command):
        if command in names:
            return f"/usr/bin/{command}"
        return None
    return _which


PASS = "PASS"
FAIL = "FAIL"


def run_scenario(name, run):
    try:
        run()
    except AssertionError as exc:
        print(f"[{FAIL}] {name}: {exc}")
        return FAIL
    except Exception as exc:
        print(f"[{FAIL}] {name}: unexpected {type(exc).__name__}: {exc}")
        return FAIL
    else:
        print(f"[{PASS}] {name}")
        return PASS


def get_vue(status):
    servers = {item["id"]: item for item in status["servers"]}
    return servers["vue"]


def scenario_workspace_local_hit():
    with tempfile.TemporaryDirectory() as td:
        workspace = WorkspaceStub(Path(td))
        plugin = workspace.workspace / "node_modules" / "@vue" / "typescript-plugin"
        plugin.mkdir(parents=True)
        manager = LspManager(workspace, FakeSandbox(), config=LspConfig())
        names = {"vue-language-server", "typescript-language-server", "node"}
        with patch.object(LspManager, "_which", new=make_which(names)):
            status = manager.status()
        encoded = json.dumps(status)
        assert json.loads(encoded), "payload must round-trip"
        vue_row = get_vue(status)
        plugin_path = vue_row["optional_dependencies"]["vue_typescript_plugin"]
        assert plugin_path == str(plugin.resolve()), (
            f"expected {plugin.resolve()!s}, got {plugin_path!r}"
        )
        assert vue_row["installed"], "Vue executable should be reported installed"
        assert vue_row["available"], "Vue bridge should be reported available"


def scenario_bind_target():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        host_node_modules = root / "host_lib" / "node_modules"
        host_node_modules.mkdir(parents=True)
        (host_node_modules / "@vue" / "typescript-plugin").mkdir(parents=True)
        (host_node_modules / "@vue" / "language-server").mkdir(parents=True)
        workspace = WorkspaceStub(
            root,
            runtime_readonly_binds=((root / "host_src", host_node_modules),),
        )
        manager = LspManager(workspace, FakeSandbox(), config=LspConfig())
        names = {"vue-language-server", "typescript-language-server", "node"}
        with patch.object(LspManager, "_which", new=make_which(names)):
            status = manager.status()
            located = manager._vue_plugin_location(workspace.workspace)
        encoded = json.dumps(status)
        assert json.loads(encoded), "payload must round-trip"
        vue_row = get_vue(status)
        plugin = vue_row["optional_dependencies"]["vue_typescript_plugin"]
        assert plugin, "vue_typescript_plugin must be set when bind target is visible"
        assert str(host_node_modules.resolve()) in plugin, (
            f"bind target {host_node_modules.resolve()} not in {plugin!r}"
        )
        assert located is not None
        assert str(host_node_modules.resolve()) in str(located), (
            f"located {located!s} does not include host bind target"
        )
        rendered = LspCommand._format_status(status)
        assert "@vue/typescript-plugin missing" not in rendered, rendered
        assert vue_row["installed"], "Vue executable should be reported installed"
        assert vue_row["available"], "Vue bridge should be reported available"


def scenario_staged_env():
    with tempfile.TemporaryDirectory() as td:
        workspace = WorkspaceStub(Path(td))
        staged = workspace.env / "npm" / "lib" / "node_modules"
        (staged / "@vue" / "typescript-plugin").mkdir(parents=True)
        (staged / "@vue" / "language-server").mkdir(parents=True)
        manager = LspManager(workspace, FakeSandbox(), config=LspConfig())
        names = {"vue-language-server", "typescript-language-server", "node"}
        with patch.object(LspManager, "_which", new=make_which(names)):
            located = manager._vue_plugin_location(None)
        assert located is not None, "expected plugin path under staged npm prefix"
        assert located == (staged / "@vue" / "typescript-plugin").resolve(), (
            f"expected {(staged / '@vue' / 'typescript-plugin').resolve()!s}, got {located!s}"
        )


def scenario_negative_plugin_missing_with_executable():
    with tempfile.TemporaryDirectory() as td:
        workspace = WorkspaceStub(Path(td))
        manager = LspManager(workspace, FakeSandbox(), config=LspConfig())
        with patch.object(
            LspManager,
            "_which",
            new=make_which({"vue-language-server", "typescript-language-server", "node"}),
        ):
            status = manager.status()
            located = manager._vue_plugin_location(None)
            raised = False
            try:
                manager._ensure_vue_typescript(workspace.workspace)
            except LspUnavailable as exc:
                raised = True
                message = str(exc)
                assert "@vue/typescript-plugin is required" in message, (
                    f"unexpected message: {message!r}"
                )
        vue_row = get_vue(status)
        assert vue_row["optional_dependencies"]["vue_typescript_plugin"] is None, (
            f"plugin path should be None, got {vue_row['optional_dependencies']['vue_typescript_plugin']!r}"
        )
        assert not vue_row["available"], "Vue should not be available without the plugin"
        assert located is None
        assert raised, "LspUnavailable should be raised when only the plugin is missing"


def scenario_negative_no_executable():
    with tempfile.TemporaryDirectory() as td:
        workspace = WorkspaceStub(Path(td))
        manager = LspManager(workspace, FakeSandbox(), config=LspConfig())
        with patch.object(LspManager, "_which", new=make_which(set())):
            status = manager.status()
            located = manager._vue_plugin_location(None)
            raised = False
            try:
                manager._ensure_vue_typescript(workspace.workspace)
            except LspUnavailable as exc:
                raised = True
                message = str(exc)
                assert "vue-language-server is not installed" in message, (
                    f"unexpected message: {message!r}"
                )
        vue_row = get_vue(status)
        assert vue_row["optional_dependencies"]["vue_typescript_plugin"] is None
        assert not vue_row["installed"]
        assert located is None
        assert raised, "LspUnavailable should fire when nothing is installed"


def scenario_no_env_attribute():
    with tempfile.TemporaryDirectory() as td:
        workspace = WorkspaceStub(Path(td))
        if hasattr(workspace, "env"):
            workspace.__dict__.pop("env", None)
        manager = LspManager(workspace, FakeSandbox(), config=LspConfig())
        roots = manager._sandbox_node_module_roots()
        assert roots == (), f"expected empty tuple, got {roots!r}"


def scenario_path_component_qualification():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        legacy_target = root / "npm_root" / "lib" / "node_modules"
        (legacy_target / "@vue" / "typescript-plugin").mkdir(parents=True)
        (legacy_target / "@vue" / "language-server").mkdir(parents=True)
        workspace = WorkspaceStub(
            root,
            runtime_readonly_binds=((root / "host_npm", legacy_target),),
        )
        manager = LspManager(workspace, FakeSandbox(), config=LspConfig())
        names = {"vue-language-server", "typescript-language-server", "node"}
        with patch.object(LspManager, "_which", new=make_which(names)):
            status = manager.status()
        vue_row = get_vue(status)
        plugin = vue_row["optional_dependencies"]["vue_typescript_plugin"]
        assert plugin, "legacy npm-prefix layout should be detected"
        assert str(legacy_target.resolve()) in plugin, (
            f"legacy target {legacy_target.resolve()} not in {plugin!r}"
        )


def main():
    results = []
    results.append(run_scenario(
        "A4 regression: workspace-local hit + JSON-serializable",
        scenario_workspace_local_hit,
    ))
    results.append(run_scenario(
        "A5+A3+A1: host bind target -> status formatter / plugin path",
        scenario_bind_target,
    ))
    results.append(run_scenario(
        "A5: staged-env npm prefix",
        scenario_staged_env,
    ))
    results.append(run_scenario(
        "A3 negative: executable+bridge present but plugin missing -> LspUnavailable",
        scenario_negative_plugin_missing_with_executable,
    ))
    results.append(run_scenario(
        "Negative: nothing installed -> earlier guard fires",
        scenario_negative_no_executable,
    ))
    results.append(run_scenario(
        "Robustness: workspace without env attribute",
        scenario_no_env_attribute,
    ))
    results.append(run_scenario(
        "Legacy npm-prefix: path-component qualification",
        scenario_path_component_qualification,
    ))

    passed = sum(1 for r in results if r == PASS)
    failed = sum(1 for r in results if r == FAIL)
    print(f"\nSummary: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
