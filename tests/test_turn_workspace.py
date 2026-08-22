from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from citra.context.config_loader import WorkspaceContextConfig
from citra.context.turn_workspace import WorkspaceContext
from citra.context.workspace_changes import (
    WorkspaceChanges,
    WorkspaceConflictError,
)
from citra.utils.sandbox import WorkspaceSandbox


class _FakeProcess:
    returncode = 0
    pid = 12345

    def communicate(self, timeout=None):
        return ("ok\n", None)


class TurnWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()

        self._git("init", "--quiet")
        self._git("config", "user.name", "Test User")
        self._git("config", "user.email", "test@example.invalid")

        (self.source / "tracked.txt").write_text(
            "one\ntwo\nthree\n",
            encoding="utf-8",
        )
        (self.source / "nested").mkdir()
        (self.source / "nested" / "module.py").write_text(
            "VALUE = 1\n",
            encoding="utf-8",
        )
        (self.source / "untracked.txt").write_text(
            "not copied\n",
            encoding="utf-8",
        )

        self._git(
            "add",
            "tracked.txt",
            "nested/module.py",
        )
        self._git(
            "commit",
            "--quiet",
            "-m",
            "baseline",
        )

        self.context = WorkspaceContext.create(
            config=WorkspaceContextConfig(
                temporary_workspace=str(
                    self.base / "turns"
                ),
                permanent_workspace=None,
            ),
            workspace=self.source,
        )

    def tearDown(self) -> None:
        self.context.cleanup()
        self.temporary.cleanup()

    def _git(self, *arguments: str) -> str:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(self.source),
                *arguments,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout

    def test_initial_workspace_is_empty_except_source_mountpoint(self) -> None:
        self.assertEqual(
            [path.name for path in self.context.workspace.iterdir()],
            ["@source"],
        )

    def test_aliases_separate_source_workspace_and_home(self) -> None:
        self.assertEqual(
            self.context.resolve_path("."),
            self.context.workspace,
        )
        self.assertEqual(
            self.context.resolve_path("~/notes.txt"),
            self.context.home / "notes.txt",
        )
        self.assertEqual(
            self.context.resolve_path("@source/tracked.txt"),
            self.source / "tracked.txt",
        )
        self.assertEqual(
            self.context.resolve_path("./@source/tracked.txt"),
            self.source / "tracked.txt",
        )
        self.assertEqual(
            self.context.environment()["CITRA_SOURCE"],
            str(self.source),
        )

        with self.assertRaises(ValueError):
            self.context.require_writable_path(
                "@source/tracked.txt"
            )

    def test_materialize_copies_only_selected_scope(self) -> None:
        result = self.context.changes.materialize(
            ["nested"]
        )

        self.assertEqual(
            result.materialized,
            ("nested/module.py",),
        )
        self.assertTrue(
            (self.context.workspace / "nested" / "module.py").is_file()
        )
        self.assertFalse(
            (self.context.workspace / "tracked.txt").exists()
        )
        self.assertFalse(
            (self.context.workspace / "untracked.txt").exists()
        )
        self.assertEqual(
            self.context.changes.materialize(
                ["untracked.txt"]
            ).materialized,
            ("untracked.txt",),
        )
        self.assertEqual(
            self.context.changes.status(),
            "(clean)",
        )

    def test_materialize_can_expand_to_remaining_project(self) -> None:
        self.context.changes.materialize(
            ["nested"]
        )
        selected = self.context.workspace / "nested" / "module.py"
        selected.write_text(
            "VALUE = 2\n",
            encoding="utf-8",
        )

        expanded = self.context.changes.materialize(
            ["."]
        )

        self.assertEqual(
            expanded.materialized,
            (
                "tracked.txt",
                "untracked.txt",
            ),
        )
        self.assertEqual(
            expanded.already_materialized,
            ("nested/module.py",),
        )
        self.assertEqual(
            selected.read_text(encoding="utf-8"),
            "VALUE = 2\n",
        )
        self.assertTrue(
            (self.context.workspace / "untracked.txt").exists()
        )

    def test_non_git_source_can_materialize_and_apply_updates(self) -> None:
        source = self.base / "non-git-source"
        source.mkdir()
        (source / "utility.py").write_text(
            "VALUE = 1\n",
            encoding="utf-8",
        )
        context = WorkspaceContext.create(
            config=WorkspaceContextConfig(
                temporary_workspace=str(
                    self.base / "non-git-turns"
                ),
                permanent_workspace=None,
            ),
            workspace=source,
        )

        try:
            result = context.changes.materialize(
                ["utility.py"]
            )
            self.assertEqual(
                result.materialized,
                ("utility.py",),
            )
            (context.workspace / "utility.py").write_text(
                "VALUE = 2\n",
                encoding="utf-8",
            )
            (context.workspace / "new_utility.py").write_text(
                "NEW = True\n",
                encoding="utf-8",
            )
            context.changes.stage(
                [
                    "utility.py",
                    "new_utility.py",
                ]
            )
            context.changes.apply()

            self.assertEqual(
                (source / "utility.py").read_text(encoding="utf-8"),
                "VALUE = 2\n",
            )
            self.assertEqual(
                (source / "new_utility.py").read_text(encoding="utf-8"),
                "NEW = True\n",
            )
        finally:
            context.cleanup()

    def test_preview_reports_scope_without_copying(self) -> None:
        result = self.context.changes.materialize(
            ["."],
            preview=True,
        )

        self.assertTrue(result.preview)
        self.assertEqual(
            result.planned,
            (
                "nested/module.py",
                "tracked.txt",
                "untracked.txt",
            ),
        )
        self.assertGreater(result.total_bytes, 0)
        self.assertEqual(
            [path.name for path in self.context.workspace.iterdir()],
            ["@source"],
        )

    def test_ignored_expansion_and_explicit_override(self) -> None:
        (self.source / ".gitignore").write_text(
            "ignored/\n",
            encoding="utf-8",
        )
        (self.source / "ignored").mkdir()
        (self.source / "ignored" / "local.py").write_text(
            "LOCAL = True\n",
            encoding="utf-8",
        )

        expanded = self.context.changes.materialize(
            ["."],
            preview=True,
        )
        self.assertNotIn(
            "ignored/local.py",
            expanded.planned,
        )
        self.assertIn(
            "ignored/local.py",
            expanded.ignored,
        )

        explicit = self.context.changes.materialize(
            ["ignored/local.py"]
        )
        self.assertEqual(
            explicit.materialized,
            ("ignored/local.py",),
        )

    def test_include_ignored_expands_ignored_directory(self) -> None:
        (self.source / ".gitignore").write_text(
            "generated/\n",
            encoding="utf-8",
        )
        (self.source / "generated").mkdir()
        (self.source / "generated" / "fixture.dat").write_text(
            "fixture\n",
            encoding="utf-8",
        )

        result = self.context.changes.materialize(
            ["generated"],
            include_ignored=True,
        )
        self.assertEqual(
            result.materialized,
            ("generated/fixture.dat",),
        )

    def test_exact_file_overrides_builtin_soft_exclusion(self) -> None:
        (self.source / ".venv").mkdir()
        (self.source / ".venv" / "local_tool.py").write_text(
            "LOCAL = True\n",
            encoding="utf-8",
        )

        preview = self.context.changes.materialize(
            ["."],
            preview=True,
        )
        self.assertNotIn(
            ".venv/local_tool.py",
            preview.planned,
        )

        exact = self.context.changes.materialize(
            [".venv/local_tool.py"]
        )
        self.assertEqual(
            exact.materialized,
            (".venv/local_tool.py",),
        )

    def test_citraignore_applies_without_source_git_repository(self) -> None:
        source = self.base / "ignored-non-git-source"
        source.mkdir()
        (source / ".citraignore").write_text(
            "private/\n",
            encoding="utf-8",
        )
        (source / "visible.py").write_text(
            "VISIBLE = True\n",
            encoding="utf-8",
        )
        (source / "private").mkdir()
        (source / "private" / "local.py").write_text(
            "PRIVATE = True\n",
            encoding="utf-8",
        )
        context = WorkspaceContext.create(
            config=WorkspaceContextConfig(
                temporary_workspace=str(
                    self.base / "ignored-non-git-turns"
                ),
                permanent_workspace=None,
            ),
            workspace=source,
        )

        try:
            preview = context.changes.materialize(
                ["."],
                preview=True,
            )
            self.assertIn(
                "visible.py",
                preview.planned,
            )
            self.assertNotIn(
                "private/local.py",
                preview.planned,
            )
            self.assertIn(
                "private/local.py",
                preview.ignored,
            )
        finally:
            context.cleanup()

    def test_tracked_file_remains_eligible_when_later_ignored(self) -> None:
        (self.source / ".gitignore").write_text(
            "tracked.txt\n",
            encoding="utf-8",
        )

        result = self.context.changes.materialize(
            ["."],
            preview=True,
        )
        self.assertIn(
            "tracked.txt",
            result.planned,
        )

    def test_citraignore_can_exclude_source_git_tracked_file(self) -> None:
        (self.source / ".citraignore").write_text(
            "tracked.txt\n",
            encoding="utf-8",
        )

        result = self.context.changes.materialize(
            ["."],
            preview=True,
        )
        self.assertNotIn(
            "tracked.txt",
            result.planned,
        )
        self.assertIn(
            "tracked.txt",
            result.ignored,
        )

    def test_vcs_internals_are_hard_excluded(self) -> None:
        result = self.context.changes.materialize(
            [".git/config"],
            preview=True,
        )

        self.assertEqual(result.planned, ())
        self.assertEqual(
            result.hard_excluded,
            (".git/config",),
        )

    @unittest.skipUnless(
        hasattr(os, "mkfifo"),
        "FIFOs are not supported on this platform",
    )
    def test_special_files_are_not_materialized(self) -> None:
        os.mkfifo(self.source / "local.pipe")

        result = self.context.changes.materialize(
            ["local.pipe"],
            preview=True,
        )

        self.assertEqual(result.planned, ())
        self.assertEqual(
            result.unsupported,
            ("local.pipe",),
        )

    def test_large_scope_requires_explicit_override(self) -> None:
        with mock.patch.object(
            WorkspaceChanges,
            "MAX_MATERIALIZE_FILES",
            1,
        ):
            preview = self.context.changes.materialize(
                ["."],
                preview=True,
            )
            self.assertTrue(preview.limit_exceeded)

            with self.assertRaisesRegex(
                ValueError,
                "allow_large=true",
            ):
                self.context.changes.materialize(
                    ["."]
                )

            copied = self.context.changes.materialize(
                ["."],
                allow_large=True,
            )

        self.assertEqual(
            len(copied.materialized),
            3,
        )

    def test_stage_and_apply_update_only_source_files(self) -> None:
        self.context.changes.materialize(
            ["tracked.txt"]
        )
        agent_file = self.context.workspace / "tracked.txt"
        agent_file.write_text(
            "changed\n",
            encoding="utf-8",
        )

        self.assertEqual(
            (self.source / "tracked.txt").read_text(encoding="utf-8"),
            "one\ntwo\nthree\n",
        )

        status = self.context.changes.stage(
            ["tracked.txt"]
        )
        self.assertIn(
            "M  tracked.txt",
            status,
        )
        self.assertIn(
            "+changed",
            self.context.changes.diff(
                staged=True
            ),
        )

        result = self.context.changes.apply()

        self.assertIn(
            "Applied staged file updates",
            result,
        )
        self.assertEqual(
            (self.source / "tracked.txt").read_text(encoding="utf-8"),
            "changed\n",
        )
        self.assertEqual(
            self._git("diff", "--cached"),
            "",
        )
        self.assertEqual(
            self.context.changes.status(),
            "(clean)",
        )

    def test_apply_rejects_source_changed_after_materialization(self) -> None:
        self.context.changes.materialize(
            ["tracked.txt"]
        )
        (self.context.workspace / "tracked.txt").write_text(
            "agent change\n",
            encoding="utf-8",
        )
        self.context.changes.stage(
            ["tracked.txt"]
        )
        (self.source / "tracked.txt").write_text(
            "external change\n",
            encoding="utf-8",
        )

        with self.assertRaises(WorkspaceConflictError):
            self.context.changes.apply()

        self.assertEqual(
            (self.source / "tracked.txt").read_text(encoding="utf-8"),
            "external change\n",
        )

    def test_new_file_is_applied_without_staging_source_git(self) -> None:
        new_file = self.context.workspace / "new.py"
        new_file.write_text(
            "VALUE = 2\n",
            encoding="utf-8",
        )

        self.context.changes.stage(
            ["new.py"]
        )
        self.context.changes.apply()

        self.assertEqual(
            (self.source / "new.py").read_text(encoding="utf-8"),
            "VALUE = 2\n",
        )
        self.assertEqual(
            self._git("diff", "--cached"),
            "",
        )
        self.assertIn(
            "?? new.py",
            self._git("status", "--short"),
        )

    def test_new_file_apply_rejects_source_target_collision(self) -> None:
        (self.context.workspace / "collision.py").write_text(
            "AGENT = True\n",
            encoding="utf-8",
        )
        self.context.changes.stage(
            ["collision.py"]
        )
        (self.source / "collision.py").write_text(
            "EXTERNAL = True\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            WorkspaceConflictError,
            "target appeared",
        ):
            self.context.changes.apply()

        self.assertEqual(
            (self.source / "collision.py").read_text(encoding="utf-8"),
            "EXTERNAL = True\n",
        )

    def test_tracked_file_deletion_is_applied_without_staging_source_git(
        self,
    ) -> None:
        self.context.changes.materialize(
            ["tracked.txt"]
        )
        (self.context.workspace / "tracked.txt").unlink()

        self.context.changes.stage(
            ["tracked.txt"]
        )
        result = self.context.changes.apply()

        self.assertIn(
            "D\ttracked.txt",
            result,
        )
        self.assertFalse(
            (self.source / "tracked.txt").exists()
        )
        self.assertEqual(
            self._git("diff", "--cached"),
            "",
        )

    def test_partial_patch_stages_only_selected_hunk(self) -> None:
        self.context.changes.materialize(
            ["tracked.txt"]
        )
        (self.context.workspace / "tracked.txt").write_text(
            "ONE\ntwo\nTHREE\n",
            encoding="utf-8",
        )
        patch = """\
diff --git a/tracked.txt b/tracked.txt
--- a/tracked.txt
+++ b/tracked.txt
@@ -1,3 +1,3 @@
-one
+ONE
 two
 three
"""

        self.context.changes.stage_patch(
            patch
        )
        staged = self.context.changes.diff(
            staged=True
        )

        self.assertIn(
            "+ONE",
            staged,
        )
        self.assertNotIn(
            "+THREE",
            staged,
        )

        self.context.changes.apply()

        self.assertEqual(
            (self.source / "tracked.txt").read_text(encoding="utf-8"),
            "ONE\ntwo\nthree\n",
        )
        self.assertIn(
            "+THREE",
            self.context.changes.diff(
                staged=False
            ),
        )

    def test_cleanup_removes_complete_turn_root(self) -> None:
        root = self.context.root
        self.context.cleanup()
        self.assertFalse(root.exists())

    def test_source_workspace_may_be_git_repository_subdirectory(self) -> None:
        nested_source = self.source / "nested"
        nested_context = WorkspaceContext.create(
            config=WorkspaceContextConfig(
                temporary_workspace=str(
                    self.base / "nested-turns"
                ),
                permanent_workspace=None,
            ),
            workspace=nested_source,
        )

        try:
            result = nested_context.changes.materialize(
                ["."]
            )
            self.assertEqual(
                result.materialized,
                ("module.py",),
            )
            (nested_context.workspace / "module.py").write_text(
                "VALUE = 3\n",
                encoding="utf-8",
            )
            nested_context.changes.stage(
                ["module.py"]
            )
            nested_context.changes.apply()

            self.assertEqual(
                (nested_source / "module.py").read_text(encoding="utf-8"),
                "VALUE = 3\n",
            )
            self.assertFalse(
                (self.source / "module.py").exists()
            )
        finally:
            nested_context.cleanup()

    def test_bash_sandbox_mounts_source_read_only_and_disables_network(self) -> None:
        sandbox = WorkspaceSandbox(
            self.context
        )

        with mock.patch(
            "citra.utils.sandbox.shutil.which",
            return_value="/usr/bin/bwrap",
        ), mock.patch(
            "citra.utils.sandbox.subprocess.Popen",
            return_value=_FakeProcess(),
        ) as popen:
            result = sandbox.run(
                ["bash", "-c", "pwd"],
                timeout=5,
                network=False,
            )

        command = popen.call_args.args[0]
        environment = popen.call_args.kwargs["env"]
        inherited_descriptors = popen.call_args.kwargs["pass_fds"]

        self.assertEqual(result.output, "ok\n")
        self.assertIn("--unshare-net", command)
        source_alias = str(self.context.workspace / "@source")
        source_alias_mounts = [
            command[index:index + 3]
            for index in range(len(command) - 2)
            if command[index] == "--ro-bind-fd"
            and command[index + 2] == source_alias
        ]
        self.assertEqual(
            len(source_alias_mounts),
            1,
        )
        self.assertEqual(
            environment["HOME"],
            str(self.context.home),
        )
        self.assertEqual(
            environment["CITRA_WORKSPACE"],
            str(self.context.workspace),
        )
        self.assertEqual(
            environment["CITRA_SOURCE"],
            str(self.source),
        )
        self.assertTrue(inherited_descriptors)
        for descriptor in inherited_descriptors:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_sandbox_skips_missing_mask_destinations(self) -> None:
        sandbox = WorkspaceSandbox(
            self.context
        )
        existing_directory = self.base / "masked-directory"
        existing_directory.mkdir()
        existing_file = self.base / "masked-file"
        existing_file.write_text(
            "secret\n",
            encoding="utf-8",
        )
        missing_directory = self.base / "missing-directory"
        missing_file = self.base / "missing-file"

        with mock.patch(
            "citra.utils.sandbox.MASKED_HOST_DIRS",
            (
                str(existing_directory),
                str(missing_directory),
            ),
        ), mock.patch(
            "citra.utils.sandbox.MASKED_HOST_FILES",
            (
                str(existing_file),
                str(missing_file),
            ),
        ):
            command = sandbox._build_bwrap_command(
                bwrap="/usr/bin/bwrap",
                command=["true"],
                cwd_path=self.context.workspace,
                network=False,
                env=self.context.environment(),
                turn_dirs=sandbox._prepare_lifecycle_directories(),
            )

        self.assertIn(
            ["--tmpfs", str(existing_directory)],
            [
                command[index:index + 2]
                for index in range(len(command) - 1)
            ],
        )
        self.assertNotIn(str(missing_directory), command)
        self.assertIn(
            ["--ro-bind", "/dev/null", str(existing_file)],
            [
                command[index:index + 3]
                for index in range(len(command) - 2)
            ],
        )
        self.assertNotIn(str(missing_file), command)

    def test_sandbox_recreates_parents_for_runtime_under_mask(self) -> None:
        sandbox = WorkspaceSandbox(
            self.context
        )
        masked_home = self.base / "host-home"
        project = masked_home / "felipey" / "Code" / "Citra"
        source_runtime = project / "src"
        virtual_environment = project / ".venv"
        executable_directory = virtual_environment / "bin"
        source_runtime.mkdir(parents=True)
        executable_directory.mkdir(parents=True)
        executable = executable_directory / "python"
        executable.write_text(
            "",
            encoding="utf-8",
        )

        with mock.patch(
            "citra.utils.sandbox.MASKED_HOST_DIRS",
            (str(masked_home),),
        ), mock.patch.object(
            WorkspaceSandbox,
            "_citra_runtime_readonly_binds",
            return_value=(
                source_runtime,
                virtual_environment,
            ),
        ), mock.patch.object(
            WorkspaceSandbox,
            "_command_runtime_readonly_binds",
            return_value=(
                executable_directory,
                virtual_environment,
            ),
        ), mock.patch(
            "citra.utils.sandbox.shutil.which",
            return_value="/usr/bin/bwrap",
        ), mock.patch(
            "citra.utils.sandbox.subprocess.Popen",
            return_value=_FakeProcess(),
        ) as popen:
            sandbox.run(
                [str(executable)],
                timeout=5,
                network=False,
            )

        command = popen.call_args.args[0]

        directory_operations = [
            command[index:index + 2]
            for index in range(len(command) - 1)
        ]
        bind_operations = [
            command[index:index + 3]
            for index in range(len(command) - 2)
        ]

        self.assertIn(
            ["--dir", str(masked_home / "felipey")],
            directory_operations,
        )
        self.assertIn(
            ["--dir", str(project)],
            directory_operations,
        )
        self.assertIn(
            [
                "--ro-bind-fd",
                next(
                    command[index + 1]
                    for index in range(len(command) - 2)
                    if command[index] == "--ro-bind-fd"
                    and command[index + 2] == str(virtual_environment)
                ),
                str(virtual_environment),
            ],
            bind_operations,
        )
        self.assertNotIn(
            [
                "--ro-bind-fd",
                next(
                    (
                        command[index + 1]
                        for index in range(len(command) - 2)
                        if command[index] == "--ro-bind-fd"
                        and command[index + 2] == str(executable_directory)
                    ),
                    "not-mounted",
                ),
                str(executable_directory),
            ],
            bind_operations,
        )


if __name__ == "__main__":
    unittest.main()
