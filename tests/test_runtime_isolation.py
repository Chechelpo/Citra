"""Regression tests for the isolated Agent Runtime contract."""

from __future__ import annotations

import ast
import os
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from citra.config import SandboxPolicy
from citra.config.runtime_discovery import RuntimeDiscovery, RuntimeDiscoveryResult
from citra.context.workspace_context.runtime import (
    CopyPolicy,
    RuntimeAsset,
    RuntimeProvisionError,
    RuntimeProvisioner,
    ToolDefinition,
)
from citra.sandbox.filesystem_ops.read import ReadInput, execute as execute_read
from citra.sandbox.filesystem_ops.scope import ScopedFilesystem
from citra.sandbox.sandbox import WorkspaceSandbox
from citra.sandbox.sandboxed_filesystem import SandboxedFilesystem
from citra.sandbox.sandbox_mode import SandboxMode


class _FixtureDiscovery(RuntimeDiscovery):
    """Supply one deterministic command for registry tests."""

    @classmethod
    def discover(cls) -> RuntimeDiscoveryResult:
        """Return the fixture command closure."""
        return RuntimeDiscoveryResult(
            readonly_binds=(Path("/fixture/bin"),),
            available_commands=("fixture",),
            command_paths=(("fixture", Path("/fixture/bin/fixture")),),
        )


class _WorkspaceFixture:
    """Provide the subset of WorkspaceContext consumed by WorkspaceSandbox."""

    def __init__(self, workspace: Path, command: Path) -> None:
        """Store the fixture project root and resolved command."""
        self.workspace = workspace
        self._command = command

    def resolve_command(self, command: str) -> Path | None:
        """Resolve the fixture command through the isolated launcher."""
        if command == "fixture":
            return self._command
        if command == "python":
            return Path("/runtime/bin/python")
        return None

    def environment(self) -> dict[str, str]:
        """Return the minimal sandbox environment."""
        return {"PATH": "/runtime/bin"}


class RuntimeIsolationTests(unittest.TestCase):
    """Verify mode-specific provisioning and canonical command execution."""

    def _definition(self, prefix: Path) -> ToolDefinition:
        """Create a definition whose executable lives under one prefix."""
        executable = prefix / "bin" / "fixture"
        executable.parent.mkdir(parents=True)
        executable.write_text("fixture-v1\n", encoding="utf-8")
        executable.chmod(0o755)
        asset = RuntimeAsset(
            id="fixture-runtime",
            source=prefix,
            destination=PurePosixPath("rootfs", *prefix.parts[1:]),
            policy=CopyPolicy.BIND_ONLY,
            bind_target=prefix,
        )
        return ToolDefinition(
            id="fixture",
            commands=("fixture",),
            assets=(asset,),
            command_assets={"fixture": asset.id},
            command_sources={"fixture": executable},
        )

    def test_full_sandbox_copies_even_bind_only_declarations(self) -> None:
        """Require host independence whenever full sandbox mode is selected."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            definition = self._definition(root / "host" / "prefix")
            provisioned = RuntimeProvisioner(
                runtime_root=root / "runtime",
                copy_budget_bytes=1_000_000,
                mode=SandboxMode.FULL_SANDBOX,
            ).provision((definition,))

            asset = provisioned.assets["fixture-runtime"]
            self.assertEqual(asset.mode, "copy")
            self.assertIsNotNone(asset.runtime_path)
            self.assertTrue((asset.runtime_path / "bin" / "fixture").is_file())
            (root / "host" / "prefix" / "bin" / "fixture").write_text(
                "fixture-v2\n",
                encoding="utf-8",
            )
            self.assertEqual(
                (asset.runtime_path / "bin" / "fixture").read_text(
                    encoding="utf-8"
                ),
                "fixture-v1\n",
            )
            self.assertEqual(
                provisioned.resolve_command("fixture"),
                Path("/runtime/bin/fixture"),
            )
            self.assertEqual(
                provisioned.readonly_binds,
                ((asset.runtime_path, root / "host" / "prefix"),),
            )
            with self.assertRaises(RuntimeProvisionError):
                RuntimeProvisioner(
                    runtime_root=root / "insufficient-runtime",
                    copy_budget_bytes=0,
                    mode=SandboxMode.FULL_SANDBOX,
                ).provision((definition,))

    def test_partial_sandbox_readonly_binds_even_copy_required_assets(self) -> None:
        """Avoid copying host runtimes in partial sandbox mode."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            definition = self._definition(root / "host" / "prefix")
            asset = definition.assets[0]
            copy_required = RuntimeAsset(
                id=asset.id,
                source=asset.source,
                destination=asset.destination,
                policy=CopyPolicy.COPY_REQUIRED,
                bind_target=asset.bind_target,
            )
            definition = ToolDefinition(
                id=definition.id,
                commands=definition.commands,
                assets=(copy_required,),
                command_assets=definition.command_assets,
                command_sources=definition.command_sources,
            )
            provisioned = RuntimeProvisioner(
                runtime_root=root / "runtime",
                copy_budget_bytes=0,
                mode=SandboxMode.PARTIAL_SANDBOX,
            ).provision((definition,))

            self.assertEqual(provisioned.copied_bytes, 0)
            self.assertEqual(provisioned.assets[asset.id].mode, "ro-bind")
            self.assertEqual(
                provisioned.readonly_binds,
                ((asset.source, asset.bind_target),),
            )

    def test_sandbox_forces_runtime_path_and_maps_mount_targets(self) -> None:
        """Keep caller overrides from reintroducing the controller PATH."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            writable_bin = workspace / ".venv" / "bin"
            writable_bin.mkdir(parents=True)
            runtime = root / "runtime"
            runtime.mkdir()
            policy = SandboxPolicy(mode=SandboxMode.FULL_SANDBOX)
            policy.add_readonly_bind(runtime, Path("/runtime"))
            fixture = _WorkspaceFixture(workspace, Path("/runtime/bin/fixture"))
            sandbox = WorkspaceSandbox(
                fixture,
                policy,
                base_environment={"PATH": "/runtime/bin"},
            )

            environment = sandbox.build_environment(
                overrides={"PATH": "/host/bin"},
                path_prepend=(writable_bin,),
            )
            self.assertEqual(
                environment["PATH"],
                f"{writable_bin}:/runtime/bin",
            )
            self.assertEqual(
                sandbox._resolve_command(("fixture", "--version")),
                ("/runtime/bin/fixture", "--version"),
            )
            arguments = sandbox.build_bwrap_arguments(
                command=("/runtime/bin/fixture",),
                cwd=workspace,
                network=False,
            )
            triples = tuple(zip(arguments, arguments[1:], arguments[2:]))
            self.assertIn(("--ro-bind", str(runtime), "/runtime"), triples)
            filesystem = SandboxedFilesystem(sandbox)
            self.assertEqual(
                filesystem._worker_python,
                Path("/runtime/bin/python"),
            )

    def test_read_operation_parses_and_executes_literal_requests(self) -> None:
        """Prevent regressions in the worker operation that failed in logs."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            (project / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            paths = {
                "HOME": root / "home",
                "CITRA_TMP": root / "tmp",
                "CITRA_CACHE": root / "cache",
                "CITRA_ENV": root / "env",
                "CITRA_RUNTIME": root / "runtime",
                "XDG_CONFIG_HOME": root / "config",
                "XDG_DATA_HOME": root / "data",
                "XDG_RUNTIME_DIR": root / "run",
            }
            for path in paths.values():
                path.mkdir(parents=True)
            environment = {
                "CITRA_PROJECT_ROOT": str(project),
                **{name: str(path) for name, path in paths.items()},
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                filesystem = ScopedFilesystem()
                order = ReadInput.parse(
                    {"path": "sample.txt", "offset": 1, "limit": 1}
                )
                output = execute_read(order, filesystem)
            self.assertEqual(output.render(), "two\n")

    def test_source_has_documentation_and_bounded_module_sizes(self) -> None:
        """Enforce the documented modularity constraints across Citra source."""
        source_root = Path(__file__).parents[1] / "src" / "citra"
        missing: list[str] = []
        oversized: list[str] = []
        for path in source_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if len(text.splitlines()) >= 1_000:
                oversized.append(str(path.relative_to(source_root)))
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                ) and ast.get_docstring(node, clean=False) is None:
                    missing.append(f"{path}:{node.lineno}:{node.name}")
        self.assertEqual(oversized, [])
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
