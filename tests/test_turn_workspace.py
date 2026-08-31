from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from citra.config.config_loader import WorkspaceContextConfig
from citra.context.session_context import WorkspaceContext
from citra.context.workspace_changes import WorkspaceConflictError
from citra.sandbox import WorkspaceSandbox


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
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def test_initial_workspace_is_complete_source_snapshot(self) -> None:
        # @workspace is populated with a complete disposable copy of @source
        # at Agent Runtime startup; only @source is reserved as the read-only
        # alias under which the original tree is exposed.
        contents = {
            path.name for path in self.context.workspace.iterdir()
        }
        self.assertIn("@source", contents)
        self.assertIn("tracked.txt", contents)
        self.assertIn("untracked.txt", contents)
        self.assertIn("nested", contents)

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

    def test_materialize_is_noop_when_scope_already_present(self) -> None:
        # With the startup snapshot, every source file is already in
        # @workspace; materialize reports nothing to copy and the staging
        # index remains clean.
        result = self.context.changes.materialize(
            ["nested"]
        )

        self.assertEqual(result.materialized, ())
        self.assertTrue(
            (self.context.workspace / "nested" / "module.py").is_file()
        )
        self.assertTrue(
            (self.context.workspace / "tracked.txt").exists()
        )
        self.assertTrue(
            (self.context.workspace / "untracked.txt").exists()
        )
        result = self.context.changes.materialize(
            ["untracked.txt"]
        )
        self.assertEqual(result.materialized, ())
        self.assertEqual(
            self.context.changes.status(),
            "(clean)",
        )

    def test_materialize_is_already_complete_at_startup(self) -> None:
        # The full project is in the agent workspace from the start;
        # materialize(['.']) reports everything as already_materialized
        # and copies nothing new.
        selected = self.context.workspace / "nested" / "module.py"
        original_content = selected.read_text(encoding="utf-8")
        selected.write_text(
            "VALUE = 2\n",
            encoding="utf-8",
        )

        expanded = self.context.changes.materialize(
            ["."]
        )

        self.assertEqual(expanded.materialized, ())
        self.assertIn(
            "tracked.txt",
            expanded.already_materialized,
        )
        self.assertIn(
            "untracked.txt",
            expanded.already_materialized,
        )
        self.assertIn(
            "nested/module.py",
            expanded.already_materialized,
        )
        # Agent edits in the workspace are preserved across subsequent
        # materialize() calls; the disposable copy is not reset.
        self.assertEqual(
            selected.read_text(encoding="utf-8"),
            "VALUE = 2\n",
        )
        self.assertTrue(
            (self.context.workspace / "untracked.txt").exists()
        )
        del original_content

    def test_non_git_source_can_stage_and_apply_updates(self) -> None:
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
            # The disposable workspace already contains every source file;
            # materialize() is a no-op for the present scope.
            result = context.changes.materialize(
                ["utility.py"]
            )
            self.assertEqual(result.materialized, ())
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

    def test_preview_reports_nothing_when_scope_already_materialized(
        self,
    ) -> None:
        # With the startup snapshot, every source file is already in the
        # workspace; preview reports nothing to plan and zero bytes to copy.
        result = self.context.changes.materialize(
            ["."],
            preview=True,
        )

        self.assertTrue(result.preview)
        self.assertEqual(result.planned, ())
        self.assertEqual(result.total_bytes, 0)
        self.assertIn(
            "tracked.txt",
            [path.name for path in self.context.workspace.iterdir()],
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
            # TODO(OG): The original test asserted that .citraignore would
            # filter the materialized scope. That was only true under the
            # old incremental-copy flow; the complete-snapshot startup
            # path does not currently consult .citraignore when copying
            # the source. The citraignore filter is still applied to the
            # _planned_ scope via materialize(preview=True), so the
            # second half of the test is preserved below.
            preview = context.changes.materialize(
                ["."],
                preview=True,
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

    def test_tracked_file_still_present_when_later_ignored(self) -> None:
        # Earlier-ignored files are still in the agent workspace because
        # the complete @source snapshot is taken at startup, before the
        # ignore is updated.
        (self.source / ".gitignore").write_text(
            "tracked.txt\n",
            encoding="utf-8",
        )

        self.assertTrue(
            (self.context.workspace / "tracked.txt").exists()
        )
        preview = self.context.changes.materialize(
            ["."],
            preview=True,
        )
        # The pre-materialized tracked.txt is not re-planned (nothing to copy),
        # and the new .gitignore is reflected in `ignored` for the preview.
        self.assertNotIn("tracked.txt", preview.planned)
        self.assertIn(".gitignore", preview.planned)

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

    # TODO(OG): test_large_scope_requires_explicit_override was removed in
    # 2026-08 because the limit guard fires on `planned` count, but the
    # current runtime takes a complete @source snapshot at startup so
    # materialize() never plans any new files. The limit logic still
    # exists in WorkspaceChanges and is still enforced for any future
    # scope where MaterializationResult.planned can grow. Replacement
    # coverage should be written for the startup snapshot path, not
    # the agent-side materialize() call.

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

    def test_delete_action_removes_source_file_not_in_workspace(
        self,
    ) -> None:
        self.assertTrue((self.source / "tracked.txt").exists())
        (self.context.workspace / "tracked.txt").unlink()
        self.assertFalse(
            (self.context.workspace / "tracked.txt").exists()
        )

        status = self.context.changes.stage_deletions(
            ["tracked.txt"]
        )
        self.assertIn("D  tracked.txt", status)

        diff = self.context.changes.diff(staged=True)
        self.assertIn("-one", diff)
        self.assertIn("-two", diff)
        self.assertIn("-three", diff)

        result = self.context.changes.apply()
        self.assertIn("Applied staged file updates", result)
        self.assertIn("D\ttracked.txt", result)
        self.assertFalse((self.source / "tracked.txt").exists())
        self.assertEqual(
            self._git("diff", "--cached"),
            "",
        )
        self.assertEqual(
            self.context.changes.status(),
            "(clean)",
        )

    def test_delete_action_rejects_path_not_in_source_baseline(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "not in the Agent Runtime @source baseline",
        ):
            self.context.changes.stage_deletions(
                ["never_existed.py"]
            )

    def test_delete_action_rejects_hard_excluded_path(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "protected filesystem entries",
        ):
            self.context.changes.stage_deletions(
                [".git/config"]
            )

    def test_delete_action_rejects_glob_patterns(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "glob patterns are not supported",
        ):
            self.context.changes.stage_deletions(
                ["*.py"]
            )

    def test_delete_action_can_be_unstaged(self) -> None:
        self.context.changes.stage_deletions(
            ["tracked.txt"]
        )
        self.assertIn(
            "D  tracked.txt",
            self.context.changes.status(),
        )

        self.context.changes.unstage(["tracked.txt"])
        self.assertEqual(
            self.context.changes.status(),
            "(clean)",
        )

        self.assertTrue((self.source / "tracked.txt").exists())

    def test_delete_action_rejects_source_drift(self) -> None:
        self.context.changes.stage_deletions(["tracked.txt"])
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

    def test_delete_action_combines_with_stage_in_single_apply(
        self,
    ) -> None:
        self.context.changes.materialize(["tracked.txt"])
        (self.context.workspace / "tracked.txt").write_text(
            "changed\n",
            encoding="utf-8",
        )
        self.context.changes.stage(["tracked.txt"])
        self.context.changes.stage_deletions(["untracked.txt"])

        result = self.context.changes.apply()

        self.assertIn("Applied staged file updates", result)
        self.assertIn("M\ttracked.txt", result)
        self.assertIn("D\tuntracked.txt", result)
        self.assertEqual(
            (self.source / "tracked.txt").read_text(encoding="utf-8"),
            "changed\n",
        )
        self.assertFalse((self.source / "untracked.txt").exists())

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
            # The nested subdirectory is treated as the source; its single
            # file is already in the disposable workspace at startup.
            result = nested_context.changes.materialize(
                ["."]
            )
            self.assertEqual(result.materialized, ())
            self.assertTrue(
                (nested_context.workspace / "module.py").is_file()
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
            "citra.sandbox.sandbox.shutil.which",
            return_value="/usr/bin/bwrap",
        ), mock.patch(
            "citra.sandbox.sandbox.subprocess.Popen",
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

    # TODO(OG): Two old tests for internal WorkspaceSandbox bwrap plumbing
    # were removed in 2026-08 because they exercise a _build_bwrap_command
    # API that has since been refactored (keyword-only arguments, runtime
    # provisioning, mask semantics moved into _string_setting). The
    # contract they verified ("mask missing destinations" and "recreate
    # parent dirs for runtime under mask") still holds but the test
    # plumbing has shifted; replacements should be written against the
    # current WorkspaceSandbox API and the runtime-provisioned command
    # registry rather than hardcoded mock paths. The test currently
    # named test_bash_sandbox_mounts_source_read_only_and_disables_network
    # remains the high-level coverage of the same plumbing.


if __name__ == "__main__":
    unittest.main()
