"""Regression tests for the isolated Agent Runtime contract."""

from __future__ import annotations

import ast
import os
import subprocess
import unittest
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from unittest import mock

from citra.config import SandboxPolicy
from citra.config.runtime_discovery import (
    RuntimeDiscovery,
    RuntimeDiscoveryResult,
    StandardDiscovery,
)
from citra.context.available_tools import (
    default_runtime_assets,
    default_tool_definitions,
)
from citra.context.workspace_context.runtime import (
    CopyPolicy,
    RuntimeAsset,
    RuntimeProvisioner,
    RuntimeProvisionError,
    ToolDefinition,
)
from citra.sandbox.filesystem_ops.read import ReadInput
from citra.sandbox.filesystem_ops.read import execute as execute_read
from citra.sandbox.filesystem_ops.scope import ScopedFilesystem
from citra.sandbox.sandbox import WorkspaceSandbox
from citra.sandbox.sandbox_mode import SandboxMode
from citra.sandbox.sandboxed_filesystem import SandboxedFilesystem


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
                (root / "runtime" / "bin" / "fixture").readlink(),
                Path(
                    "/runtime/rootfs",
                    *definition.assets[0].source.parts[1:],
                    "bin",
                    "fixture",
                ),
            )
            self.assertEqual(
                provisioned.readonly_binds,
                ((asset.runtime_path, root / "host" / "prefix"),),
            )
            self.assertTrue(
                all(source.exists() for source, _target in provisioned.readonly_binds)
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

    def test_broad_usr_discovery_is_reduced_to_exact_runtime_files(self) -> None:
        """Never recursively copy or bind the complete system /usr tree."""
        executable = Path("/usr/bin/env")
        self.assertTrue(executable.is_file())
        discovery = RuntimeDiscoveryResult(
            readonly_binds=(Path("/usr"), executable),
            available_commands=("env",),
            command_paths=(("env", executable),),
        )

        assets = default_runtime_assets(
            mode=SandboxMode.FULL_SANDBOX,
            discovery=discovery,
        )
        definitions = default_tool_definitions(
            mode=SandboxMode.FULL_SANDBOX,
            discovery=discovery,
        )

        self.assertNotIn(Path("/usr"), {asset.source for asset in assets})
        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0].command_sources["env"], executable)
        self.assertEqual(definitions[0].assets[0].source, executable)

    def test_dynamic_loader_symlink_path_is_preserved(self) -> None:
        """Retain the exact ELF interpreter pathname requested by the kernel."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            loader = root / "lib" / "loader-real"
            loader.parent.mkdir()
            loader.write_bytes(b"loader")
            loader_alias = root / "lib64" / "loader"
            loader_alias.parent.mkdir()
            loader_alias.symlink_to(Path("../lib/loader-real"))
            completed = subprocess.CompletedProcess(
                args=("ldd", "fixture"),
                returncode=0,
                stdout=f"{loader_alias} (0x00000000)\n",
            )

            with mock.patch(
                "citra.config.runtime_discovery._base.subprocess.run",
                return_value=completed,
            ):
                dependencies = StandardDiscovery._discover_shared_dependencies(
                    root / "fixture"
                )

            self.assertIn(loader_alias, dependencies)
            self.assertIn(loader, dependencies)

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
            copied_command = runtime / "rootfs" / "usr" / "bin" / "fixture"
            copied_command.parent.mkdir(parents=True)
            copied_command.write_text("fixture\n", encoding="utf-8")
            policy.add_runtime_mounts(
                ((copied_command, Path("/usr/bin/fixture")),)
            )
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
            triples = list(zip(arguments, arguments[1:], arguments[2:]))
            runtime_mount = ("--ro-bind", str(runtime), "/runtime")
            command_mount = (
                "--ro-bind",
                str(copied_command),
                "/usr/bin/fixture",
            )
            self.assertIn(runtime_mount, triples)
            self.assertIn(command_mount, triples)
            self.assertLess(
                triples.index(runtime_mount),
                triples.index(command_mount),
            )
            filesystem = SandboxedFilesystem(sandbox)
            self.assertEqual(
                filesystem._worker_python,
                Path("/runtime/bin/python"),
            )

    def test_sandbox_collapses_redundant_nested_runtime_mounts(self) -> None:
        """Avoid mounting a copied uv Python child below its read-only prefix."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            runtime = root / "runtime"
            runtime.mkdir()
            python_target = Path(
                "/home/felipey/.local/share/uv/python/"
                "cpython-3.12-linux-x86_64-gnu"
            )
            python_source = runtime / "rootfs" / python_target.relative_to("/")
            include_source = python_source / "include" / "python3.12"
            include_source.mkdir(parents=True)
            override_source = root / "python-lib-override"
            override_source.mkdir()

            policy = SandboxPolicy(mode=SandboxMode.FULL_SANDBOX)
            policy.add_readonly_bind(runtime, Path("/runtime"))
            policy.add_runtime_mounts(
                (
                    (python_source, python_target),
                    (include_source, python_target / "include" / "python3.12"),
                    (override_source, python_target / "lib"),
                )
            )
            sandbox = WorkspaceSandbox(
                _WorkspaceFixture(workspace, Path("/runtime/bin/fixture")),
                policy,
                base_environment={"PATH": "/runtime/bin"},
            )

            arguments = sandbox.build_bwrap_arguments(
                command=("/runtime/bin/fixture",),
                cwd=workspace,
                network=False,
            )
            triples = list(zip(arguments, arguments[1:], arguments[2:]))

            self.assertIn(
                ("--ro-bind", str(python_source), str(python_target)),
                triples,
            )
            self.assertNotIn(
                (
                    "--ro-bind",
                    str(include_source),
                    str(python_target / "include" / "python3.12"),
                ),
                triples,
            )
            self.assertIn(
                (
                    "--ro-bind",
                    str(override_source),
                    str(python_target / "lib"),
                ),
                triples,
            )

    def test_explicit_readonly_bind_overrides_generated_runtime_children(self) -> None:
        """Make an operator bind authoritative over generated child mounts."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            runtime = root / "runtime"
            runtime.mkdir()
            host_uv = root / "home" / "felipey" / ".local" / "share" / "uv"
            include_target = (
                host_uv
                / "python"
                / "cpython-3.12-linux-x86_64-gnu"
                / "include"
                / "python3.12"
            )
            include_target.mkdir(parents=True)
            copied_include = runtime / "rootfs" / include_target.relative_to("/")
            copied_include.mkdir(parents=True)
            unrelated_source = runtime / "rootfs" / "usr" / "bin" / "fixture"
            unrelated_source.parent.mkdir(parents=True)
            unrelated_source.write_text("fixture\n", encoding="utf-8")

            policy = SandboxPolicy(
                mode=SandboxMode.FULL_SANDBOX,
                extra_ro_binds=[host_uv],
            )
            policy.add_readonly_bind(runtime, Path("/runtime"))
            policy.add_runtime_mounts(
                (
                    (copied_include, include_target),
                    (unrelated_source, Path("/usr/bin/fixture")),
                )
            )
            sandbox = WorkspaceSandbox(
                _WorkspaceFixture(workspace, Path("/runtime/bin/fixture")),
                policy,
                base_environment={"PATH": "/runtime/bin"},
            )

            arguments = sandbox.build_bwrap_arguments(
                command=("/runtime/bin/fixture",),
                cwd=workspace,
                network=False,
            )
            triples = list(zip(arguments, arguments[1:], arguments[2:]))

            self.assertIn(("--ro-bind", str(host_uv), str(host_uv)), triples)
            self.assertNotIn(
                ("--ro-bind", str(copied_include), str(include_target)),
                triples,
            )
            self.assertIn(
                ("--ro-bind", str(unrelated_source), "/usr/bin/fixture"),
                triples,
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
