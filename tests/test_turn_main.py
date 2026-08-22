from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from citra.agent import AgentSession
from citra.application import CitraApplication
from citra.context import CitraConfig, ExecutionContext, WorkspaceContext
from citra.context.libraries import Libraries
from citra.tools.default_registry import TOOL_REGISTRY
from citra.tools.session_memory import TodoTool
from citra.tools.skills.skill_registry import SkillRegistry


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()
        subprocess.run(["git", "init", "--quiet", str(self.source)], check=True)
        self.config_path = self.base / "config.toml"
        self.config_path.write_text(
            f'''\
[model]
host = "https://example.invalid/v1"
api_key = "test"
id = "test-model"
max_tokens = 128

[web-search]
host_url = "http://example.invalid"

[message-context]
uncompressed_messages = 1

[workspace]
temporary_workspace = "{self.base / 'agent'}"
permanent_workspace = "{self.source}"
''',
            encoding="utf-8",
        )
        self.previous = {
            "CITRA_CONFIG_PATH": os.environ.get("CITRA_CONFIG_PATH"),
            "CITRA_ROOT": os.environ.get("CITRA_ROOT"),
        }
        os.environ["CITRA_CONFIG_PATH"] = str(self.config_path)
        os.environ["CITRA_ROOT"] = str(self.base / ".citra")
        self.config = CitraConfig.load()

    def tearDown(self) -> None:
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.temporary.cleanup()

    def test_workspace_persists_across_turns_and_closes_with_process(self) -> None:
        roots: list[Path] = []
        calls = 0

        def fake_api(*, context, messages, tools):
            nonlocal calls
            calls += 1
            roots.append(context.workspace.root)
            marker = context.workspace.workspace / "between-turns.txt"
            if calls == 1:
                marker.write_text("still here\n", encoding="utf-8")
            else:
                self.assertEqual(marker.read_text(encoding="utf-8"), "still here\n")
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        app = CitraApplication(
            config=self.config,
            source_workspace=self.source,
            api_call=fake_api,
        )
        root = app.workspace.root
        try:
            app.session.add_user_message("first")
            app.run_agent_turn()
            app.session.add_user_message("second")
            app.run_agent_turn()
            self.assertEqual(roots, [root, root])
            self.assertTrue(root.exists())
        finally:
            app.close()
        self.assertFalse(root.exists())

    def test_turn_failure_does_not_destroy_lifecycle_workspace(self) -> None:
        def fail(**_):
            raise RuntimeError("provider failed")

        app = CitraApplication(
            config=self.config,
            source_workspace=self.source,
            api_call=fail,
        )
        root = app.workspace.root
        try:
            app.session.add_user_message("work")
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                app.run_agent_turn()
            self.assertTrue(root.exists())
        finally:
            app.close()
        self.assertFalse(root.exists())

    def test_memory_is_owned_by_conversation_and_survives_tool_refresh(self) -> None:
        workspace = WorkspaceContext.create(self.config.workspace_context, self.source)
        session = AgentSession()
        try:
            context = ExecutionContext(
                workspace,
                libraries=Libraries(),
                provided_config=self.config,
                skills=SkillRegistry(agent_session=session, skills_root=None)
            )
            first = TOOL_REGISTRY.instantiate(context, session)
            todo = first["todo"]
            self.assertIsInstance(todo, TodoTool)
            todo.execute({"action": "add", "content": "survive history trimming"})
            checkpoint = first["checkpoint"]
            checkpoint.execute(
                {
                    "action": "set",
                    "content": "implementation is partially complete",
                    "next_step": "resume verification",
                }
            )
            session.add_user_message("old message")
            session.add_user_message("new message")
            self.assertEqual(len(session.get_last_n_messages(1)), 1)
            second = TOOL_REGISTRY.instantiate(context, session)
            self.assertIs(second["todo"], todo)
            self.assertIs(second["checkpoint"], checkpoint)
            self.assertIn("survive history trimming", todo.format_for_llm())
            self.assertIn("resume verification", checkpoint.format_for_llm())
        finally:
            workspace.cleanup()

    def test_workspace_and_lsp_tools_are_registered(self) -> None:
        for tool_id in ("materialize", "commit", "lsp", "checkpoint"):
            self.assertTrue(TOOL_REGISTRY.contains(tool_id))


if __name__ == "__main__":
    unittest.main()
