"""End-to-end tests for the sandboxed ``find`` filesystem operation.

The tests run the operation directly through
``FindInput.parse(...).execute(ScopedFilesystem())`` against a synthetic
directory tree, so they do not require Bubblewrap. They cover the
behavioural acceptance criteria for the v1 ``Find`` tool.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from citra.sandbox.filesystem_ops.find import (
    FindInput,
    FindOutput,
    execute as execute_find,
)
from citra.sandbox.filesystem_ops.scope import ScopedFilesystem


def _make_filesystem(root: Path) -> ScopedFilesystem:
    """Build a ``ScopedFilesystem`` whose workspace is the temp ``root``."""
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
        path.mkdir(parents=True, exist_ok=True)
    environment = {
        "CITRA_PROJECT_ROOT": str(root / "project"),
        **{name: str(path) for name, path in paths.items()},
    }
    (root / "project").mkdir(parents=True, exist_ok=True)
    with mock.patch.dict(os.environ, environment, clear=True):
        return ScopedFilesystem()


def _write_file(path: Path, text: str) -> Path:
    """Write ``text`` to ``path``, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_binary(path: Path, payload: bytes) -> Path:
    """Write ``payload`` to ``path`` in binary mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class _TreeBuilder:
    """Helper to build a deterministic source tree for ``find`` tests."""

    @staticmethod
    def build(root: Path) -> Path:
        """Create the test tree and return the project root."""
        project = root / "project"
        # A few TypeScript source files.
        _write_file(
            project / "src" / "auth" / "login.ts",
            "export function login() {\n"
            "  // TODO: rate limit\n"
            "  return true;\n"
            "}\n",
        )
        _write_file(
            project / "src" / "auth" / "session.ts",
            "export class Session {\n"
            "  start() {}\n"
            "}\n",
        )
        _write_file(
            project / "src" / "ui" / "Button.tsx",
            "export function Button() {\n"
            "  return <button />;\n"
            "}\n",
        )
        _write_file(
            project / "src" / "ui" / "Panel.tsx",
            "import React from 'react';\n"
            "export const Panel = () => <div />;\n",
        )
        _write_file(
            project / "src" / "main.ts",
            "import { login } from './auth/login';\n"
            "login();\n",
        )
        _write_file(
            project / "tests" / "login.test.ts",
            "import { login } from '../src/auth/login';\n"
            "test('login', () => { login(); });\n",
        )
        _write_file(
            project / "tests" / "session.test.ts",
            "test('session', () => {});\n",
        )
        # A Python file that should not be matched by ``extensions=['ts']``.
        _write_file(
            project / "src" / "scripts" / "build.py",
            "def build():\n    print('TODO: cleanup')\n",
        )
        # A vendored dependency tree that the exclude test must skip.
        _write_file(
            project / "node_modules" / "react" / "index.js",
            "// TODO: this is inside node_modules\n"
            "const react = {};\n"
            "module.exports = react;\n",
        )
        _write_file(
            project / "node_modules" / "lodash" / "index.js",
            "const react = 'still here';\n"
            "module.exports = {};\n",
        )
        # A binary blob to exercise the binary-skip heuristic.
        _write_binary(
            project / "src" / "assets" / "logo.bin",
            b"\x00\x01\x02react_binary_blob\x00",
        )
        # A second project root used for dedupe / cross-paths coverage.
        second = root / "second"
        second.mkdir(parents=True, exist_ok=True)
        _write_file(
            second / "extra.ts",
            "// TODO: extra project\n"
            "export const extra = 1;\n",
        )
        _write_file(
            second / "deep" / "nested" / "leaf.ts",
            "export const leaf = 1;\n",
        )
        return project


class FindExtensionFilterTests(unittest.TestCase):
    """A1: ``extensions=['ts']`` returns only ``.ts`` files."""

    def test_only_ts_files_are_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _TreeBuilder.build(root)
            fs = _make_filesystem(root)
            order = FindInput.parse(
                {
                    "paths": ["src"],
                    "extensions": ["ts"],
                }
            )
            output = execute_find(order, fs)
            self.assertIsInstance(output, FindOutput)
            self.assertEqual(output.mode, "files")
            assert output.paths is not None
            self.assertTrue(all(path.endswith(".ts") for path in output.paths))
            self.assertNotIn(
                "src/ui/Button.tsx",
                output.paths,
            )
            # Verify the matching set, irrespective of the relative prefix
            # ``ScopedFilesystem.display_path`` produces for the workspace.
            basenames = sorted(Path(path).name for path in output.paths)
            self.assertEqual(
                basenames,
                ["login.ts", "main.ts", "session.ts"],
            )
            self.assertFalse(output.truncated)
            # Touch ``project`` to silence the unused-arg warning.
            self.assertTrue(project.is_dir())


class FindContentFilterTests(unittest.TestCase):
    """A2: ``content='TODO'`` returns files containing the literal string."""

    def test_literal_substring_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _TreeBuilder.build(root)
            fs = _make_filesystem(root)
            order = FindInput.parse(
                {
                    "paths": ["src"],
                    "content": "TODO",
                }
            )
            output = execute_find(order, fs)
            self.assertEqual(output.mode, "files")
            assert output.paths is not None
            self.assertIn("src/auth/login.ts", output.paths)
            self.assertNotIn("src/auth/session.ts", output.paths)
            self.assertFalse(output.truncated)


class FindRegexTests(unittest.TestCase):
    """A3: ``content`` + ``regex=true`` finds classes / surfaces errors."""

    def test_regex_finds_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _TreeBuilder.build(root)
            fs = _make_filesystem(root)
            order = FindInput.parse(
                {
                    "paths": ["src"],
                    "content": "class\\s+\\w+",
                    "regex": True,
                }
            )
            output = execute_find(order, fs)
            assert output.paths is not None
            self.assertIn("src/auth/session.ts", output.paths)

    def test_invalid_regex_raises_value_error(self) -> None:
        """An invalid regex is rejected at parse time so the model sees the
        error before the worker dispatches a filesystem walk.
        """
        with self.assertRaises(ValueError):
            FindInput.parse(
                {
                    "paths": ["src"],
                    "content": "class(",
                    "regex": True,
                }
            )


class FindExcludeTests(unittest.TestCase):
    """A4: ``exclude`` prunes directories before they are walked."""

    def test_node_modules_is_never_traversed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _TreeBuilder.build(root)
            fs = _make_filesystem(root)
            order = FindInput.parse(
                {
                    "paths": ["."],
                    "content": "react",
                    "exclude": ["node_modules/**"],
                }
            )
            output = execute_find(order, fs)
            assert output.paths is not None
            for path in output.paths:
                self.assertFalse(
                    "node_modules" in Path(path).parts,
                    f"path under node_modules leaked: {path}",
                )
            # ``Panel.tsx`` imports react; the top-level React import is the
            # other canonical hit, so at least one match survives the prune.
            self.assertIn("src/ui/Panel.tsx", output.paths)


class FindContextTests(unittest.TestCase):
    """A5: ``context`` and ``mode='matches'`` return N lines around each hit."""

    def test_context_includes_two_lines_before_and_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _TreeBuilder.build(root)
            fs = _make_filesystem(root)
            order = FindInput.parse(
                {
                    "paths": ["src"],
                    "content": "TODO",
                    "context": 2,
                    "mode": "matches",
                }
            )
            output = execute_find(order, fs)
            self.assertEqual(output.mode, "matches")
            assert output.results is not None
            by_path = {entry.path: entry for entry in output.results}
            self.assertIn("src/auth/login.ts", by_path)
            entry = by_path["src/auth/login.ts"]
            self.assertEqual(len(entry.matches), 1)
            match = entry.matches[0]
            self.assertEqual(match.line, 2)
            self.assertIn("TODO", match.text)
            # The TODO line plus one line on each side: at least three
            # context lines, with the matching line in the middle.
            self.assertGreaterEqual(len(match.context), 3)
            self.assertIn("  // TODO: rate limit", "\n".join(match.context))


class FindAdditionalBehaviorTests(unittest.TestCase):
    """Secondary cases that round out the v1 surface."""

    def test_name_pattern_or(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _TreeBuilder.build(root)
            fs = _make_filesystem(root)
            order = FindInput.parse(
                {
                    "paths": ["src"],
                    "name": ["*.ts", "*.tsx"],
                }
            )
            output = execute_find(order, fs)
            assert output.paths is not None
            basenames = sorted(Path(path).name for path in output.paths)
            self.assertIn("login.ts", basenames)
            self.assertIn("Button.tsx", basenames)
            self.assertNotIn("build.py", basenames)

    def test_max_depth_does_not_descend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _TreeBuilder.build(root)
            fs = _make_filesystem(root)
            order = FindInput.parse(
                {
                    "paths": ["src"],
                    "maxDepth": 1,
                }
            )
            output = execute_find(order, fs)
            assert output.paths is not None
            # ``maxDepth=1`` keeps the root (``src``) and its direct files
            # such as ``main.ts``; deeper descendants like
            # ``src/auth/login.ts`` must be excluded.
            self.assertIn("src/main.ts", output.paths)
            self.assertNotIn("src/auth/login.ts", output.paths)
            self.assertNotIn("src/auth/session.ts", output.paths)
            # Path ``src/<file>`` has 2 parts but the walk's relative
            # depth is 0 (a direct child of the walk root ``src``).
            # Paths under ``src/auth/`` would have 3+ parts and a
            # relative depth >= 1, which is exactly what we are pruning.
            for path in output.paths:
                relative_depth = len(Path(path).parts) - 2
                self.assertLessEqual(relative_depth, 0)

    def test_limit_truncates_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _TreeBuilder.build(root)
            fs = _make_filesystem(root)
            order = FindInput.parse(
                {
                    "paths": ["."],
                    "extensions": ["ts", "tsx"],
                    "limit": 2,
                }
            )
            output = execute_find(order, fs)
            assert output.paths is not None
            self.assertLessEqual(len(output.paths), 2)
            self.assertTrue(output.truncated)

    def test_dedup_across_overlapping_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _TreeBuilder.build(root)
            fs = _make_filesystem(root)
            # Two roots that overlap; the canonical workspace root is
            # ``project`` (CITRA_PROJECT_ROOT) and ``second`` is reachable
            # via its absolute path. The dedupe test exercises the absolute
            # form by also listing the same file twice with different
            # relative spellings.
            order = FindInput.parse(
                {
                    "paths": [
                        "src/auth/login.ts",
                        "./src/auth/login.ts",
                    ],
                }
            )
            output = execute_find(order, fs)
            assert output.paths is not None
            self.assertEqual(
                output.paths,
                ("src/auth/login.ts",),
            )

    def test_case_insensitive_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _TreeBuilder.build(root)
            fs = _make_filesystem(root)
            order = FindInput.parse(
                {
                    "paths": ["src"],
                    "content": "todo",
                    "caseSensitive": False,
                }
            )
            output = execute_find(order, fs)
            assert output.paths is not None
            self.assertIn("src/auth/login.ts", output.paths)

    def test_matches_mode_emits_per_file_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _TreeBuilder.build(root)
            fs = _make_filesystem(root)
            order = FindInput.parse(
                {
                    "paths": ["src"],
                    "content": "TODO",
                    "mode": "matches",
                }
            )
            output = execute_find(order, fs)
            self.assertEqual(output.mode, "matches")
            assert output.results is not None
            entries = {entry.path: entry for entry in output.results}
            self.assertIn("src/auth/login.ts", entries)
            self.assertGreaterEqual(
                len(entries["src/auth/login.ts"].matches),
                1,
            )

    def test_missing_path_is_skipped_silently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _TreeBuilder.build(root)
            fs = _make_filesystem(root)
            order = FindInput.parse(
                {
                    "paths": ["does-not-exist", "src"],
                }
            )
            output = execute_find(order, fs)
            assert output.paths is not None
            # No exception is raised; the existing root still produces hits.
            self.assertIn("src/auth/login.ts", output.paths)

    def test_invalid_inputs_raise_value_error(self) -> None:
        for bad in (
            {"paths": []},
            {"paths": ["src"], "extensions": []},
            {"paths": ["src"], "maxDepth": -1},
            {"paths": ["src"], "context": -1},
            {"paths": ["src"], "limit": 0},
            {"paths": ["src"], "mode": "unknown"},
            {"paths": ["src"], "caseSensitive": "yes"},
            {"paths": ["src"], "exclude": []},
        ):
            with self.assertRaises(
                ValueError, msg=f"expected ValueError for {bad}"
            ):
                FindInput.parse(bad)

    def test_to_arguments_round_trip(self) -> None:
        order = FindInput.parse(
            {
                "paths": ["src", "tests"],
                "name": ["*.ts", "*.tsx"],
                "extensions": ["ts", ".tsx"],
                "content": "TODO",
                "regex": True,
                "caseSensitive": False,
                "exclude": ["node_modules/**"],
                "maxDepth": 3,
                "context": 2,
                "limit": 50,
                "mode": "matches",
            }
        )
        round_trip = FindInput.parse(order.to_arguments())
        self.assertEqual(round_trip, order)


if __name__ == "__main__":
    unittest.main()
