"""Regression tests for sandbox-authoritative LSP executable detection."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from citra.utils.lsp.manager import LspManager


class _SandboxStub:
    """Resolve immutable commands from a deterministic runtime manifest."""

    def __init__(self, commands: dict[str, Path]) -> None:
        """Store sandbox-visible command paths."""
        self.commands = commands

    def resolve_command(self, command: str) -> Path | None:
        """Return an immutable runtime command when one was provisioned."""
        return self.commands.get(command)


class _WorkspaceStub:
    """Expose mutable dependency-environment commands to the LSP manager."""

    def __init__(self, root: Path, commands: dict[str, Path]) -> None:
        """Create the workspace paths used while collecting status."""
        self.workspace = root / "workspace"
        self.cache = root / "cache"
        self.workspace.mkdir()
        self.cache.mkdir()
        self.commands = commands
        self.refreshes: list[str] = []

    def refresh_staged_command(self, command: str) -> Path | None:
        """Return a command installed into a mutable sandbox environment."""
        self.refreshes.append(command)
        return self.commands.get(command)


class LspSandboxDetectionTests(unittest.TestCase):
    """Verify status uses both immutable and mutable sandbox command layers."""

    def test_status_detects_server_installed_in_dependency_environment(self) -> None:
        """Report a staged Pyright server as installed and available."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pyright = root / "env" / "npm" / "bin" / "pyright-langserver"
            workspace = _WorkspaceStub(
                root,
                {"pyright-langserver": pyright},
            )
            sandbox = _SandboxStub({"node": Path("/runtime/bin/node")})
            manager = LspManager(workspace, sandbox)  # type: ignore[arg-type]

            status = manager.status()

            pyright_status = next(
                item for item in status["servers"] if item["id"] == "pyright"
            )
            self.assertTrue(pyright_status["installed"])
            self.assertTrue(pyright_status["available"])
            self.assertEqual(pyright_status["executable"], str(pyright))
            self.assertIn("pyright-langserver", workspace.refreshes)


if __name__ == "__main__":
    unittest.main()
