from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from citra.tools.transient.workspace import Workspace


class _Project:
    def __init__(self, root: Path) -> None:
        self.workspace = root.resolve()

    def resolve_path(self, raw: str) -> Path:
        candidate = (self.workspace / raw).resolve()
        candidate.relative_to(self.workspace)
        return candidate

    def display_path(self, raw: str | Path) -> str:
        return Path(raw).resolve().relative_to(self.workspace).as_posix()


class _Sandbox:
    def run(self, command, *, cwd, timeout, network, environment):
        del timeout, network
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return SimpleNamespace(
            returncode=result.returncode,
            output=result.stdout,
            timed_out=False,
        )


class WorkspaceToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self._git("init", "-q")
        self._git("config", "user.name", "test")
        self._git("config", "user.email", "test@example.com")
        (self.root / "tracked.txt").write_text("before\n", encoding="utf-8")
        self._git("add", "tracked.txt")
        self._git("commit", "-qm", "initial")

        context = SimpleNamespace(
            workspace=_Project(self.root),
            sandbox=_Sandbox(),
            has_command=lambda command: command == "git",
        )
        self.tool = Workspace.__new__(Workspace)
        object.__setattr__(self.tool, "_Tool__context", context)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.root), *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

    def test_rollback_restores_staged_and_worktree_without_committing(self) -> None:
        head = self._git("rev-parse", "HEAD").stdout.strip()
        (self.root / "tracked.txt").write_text("after\n", encoding="utf-8")
        self._git("add", "tracked.txt")

        result = self.tool._execute(
            {"operation": "rollback", "paths": ["tracked.txt"]}
        )

        self.assertIn("tracked.txt", result)
        self.assertEqual(
            (self.root / "tracked.txt").read_text(encoding="utf-8"),
            "before\n",
        )
        self.assertEqual(self._git("status", "--short").stdout, "")
        self.assertEqual(self._git("rev-parse", "HEAD").stdout.strip(), head)

    def test_mixed_untracked_request_changes_nothing(self) -> None:
        (self.root / "tracked.txt").write_text("after\n", encoding="utf-8")
        (self.root / "new.txt").write_text("new\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "tracked files only"):
            self.tool._execute(
                {
                    "operation": "rollback",
                    "paths": ["tracked.txt", "new.txt"],
                }
            )

        self.assertEqual(
            (self.root / "tracked.txt").read_text(encoding="utf-8"),
            "after\n",
        )

    def test_rejects_directory_glob_and_parent_escape(self) -> None:
        for path in (".", "*.txt", "../tracked.txt"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.tool._execute(
                    {"operation": "rollback", "paths": [path]}
                )


if __name__ == "__main__":
    unittest.main()
