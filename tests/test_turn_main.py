from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from citra import main as citra_main
from citra.agent import AgentSession
from citra.context.config_loader import WorkspaceContextConfig
from citra.tools.default_registry import TOOL_REGISTRY
from citra.tools.session_memory import TodoTool


class TurnMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.turns = self.base / "turns"
        self.config = self.base / "config.toml"

        self._git("init", "--quiet")
        self._git("config", "user.name", "Test User")
        self._git("config", "user.email", "test@example.invalid")
        (self.source / "tracked.py").write_text(
            "VALUE = 1\n",
            encoding="utf-8",
        )
        self._git("add", "tracked.py")
        self._git("commit", "--quiet", "-m", "baseline")

        self.config.write_text(
            """\
[model]
host = "https://example.invalid/v1"
api_key = "test"
id = "test-model"
max_tokens = 128

[web-search]
host_url = "http://example.invalid"

[message-context]
uncompressed_messages = 20

[workspace]
temporary_workspace = "{temporary_workspace}"
permanent_workspace = "{permanent_workspace}"
""".format(
                temporary_workspace=self.turns,
                permanent_workspace=self.source,
            ),
            encoding="utf-8",
        )
        self.previous_config = os.environ.get(
            "CITRA_CONFIG_PATH"
        )
        os.environ["CITRA_CONFIG_PATH"] = str(
            self.config
        )

    def tearDown(self) -> None:
        if self.previous_config is None:
            os.environ.pop(
                "CITRA_CONFIG_PATH",
                None,
            )
        else:
            os.environ["CITRA_CONFIG_PATH"] = self.previous_config

        self.temporary.cleanup()

    def _git(self, *arguments: str) -> str:
        return subprocess.run(
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
        ).stdout

    def test_repl_creates_and_removes_workspace_for_agent_turn(self) -> None:
        inputs = iter(
            [
                "inspect the project",
                "/q",
            ]
        )
        roots: list[Path] = []

        def fake_call_api(*, context, messages, tools):
            roots.append(
                context.workspace.root
            )
            self.assertEqual(
                [
                    path.name
                    for path in context.workspace.workspace.iterdir()
                ],
                ["@source"],
            )
            self.assertEqual(
                context.workspace.source_workspace,
                self.source,
            )
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        }
                    }
                ]
            }

        with mock.patch.object(
            citra_main.terminal_input,
            "prompt",
            side_effect=lambda *args, **kwargs: next(inputs),
        ), mock.patch.object(
            citra_main,
            "call_api",
            side_effect=fake_call_api,
        ), mock.patch(
            "builtins.print"
        ):
            citra_main.main()

        self.assertEqual(
            len(roots),
            1,
        )
        self.assertFalse(
            roots[0].exists()
        )
        self.assertEqual(
            list(self.turns.iterdir()),
            [],
        )

    def test_run_agent_turn_cleans_workspace_after_failure(self) -> None:
        config = WorkspaceContextConfig(
            temporary_workspace=str(self.turns),
            permanent_workspace=str(self.source),
        )

        with mock.patch.object(
            citra_main,
            "_run_agent_turn_in_workspace",
            side_effect=RuntimeError("test failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "test failure",
            ):
                citra_main.run_agent_turn(
                    session=AgentSession(),
                    source_workspace=self.source,
                    workspace_config=config,
                )

        self.assertEqual(
            list(self.turns.iterdir()),
            [],
        )

    def test_workspace_tools_are_registered(self) -> None:
        self.assertTrue(
            TOOL_REGISTRY.contains("materialize")
        )
        self.assertTrue(
            TOOL_REGISTRY.contains("commit")
        )

    def test_materialize_tool_supports_preview_action(self) -> None:
        workspace = citra_main.WorkspaceContext.create(
            config=WorkspaceContextConfig(
                temporary_workspace=str(self.turns),
                permanent_workspace=str(self.source),
            ),
            workspace=self.source,
        )
        session = AgentSession()

        try:
            tools = TOOL_REGISTRY.instantiate(
                citra_main.ExecutionContext(workspace),
                session,
            )
            result = tools["materialize"].execute(
                {
                    "action": "preview",
                    "paths": ["."],
                }
            )

            self.assertIn(
                "Would materialize 1 file(s)",
                result,
            )
            self.assertFalse(
                (workspace.workspace / "tracked.py").exists()
            )
        finally:
            TOOL_REGISTRY.release_session(session)
            workspace.cleanup()

    def test_memory_tools_survive_turns_with_current_context(self) -> None:
        first_workspace = citra_main.WorkspaceContext.create(
            config=WorkspaceContextConfig(
                temporary_workspace=str(self.turns),
                permanent_workspace=str(self.source),
            ),
            workspace=self.source,
        )
        second_workspace = citra_main.WorkspaceContext.create(
            config=WorkspaceContextConfig(
                temporary_workspace=str(self.turns),
                permanent_workspace=str(self.source),
            ),
            workspace=self.source,
        )
        session = AgentSession()

        try:
            first_tools = TOOL_REGISTRY.instantiate(
                citra_main.ExecutionContext(first_workspace),
                session,
            )
            todo = first_tools["todo"]
            self.assertIsInstance(todo, TodoTool)
            todo.execute(
                {
                    "action": "add",
                    "content": "survive the turn boundary",
                }
            )

            second_context = citra_main.ExecutionContext(
                second_workspace
            )
            second_tools = TOOL_REGISTRY.instantiate(
                second_context,
                session,
            )

            self.assertIs(
                second_tools["todo"],
                todo,
            )
            self.assertIs(
                todo.context,
                second_context,
            )
            self.assertIn(
                "survive the turn boundary",
                todo.format_for_llm(),
            )
        finally:
            TOOL_REGISTRY.release_session(session)
            first_workspace.cleanup()
            second_workspace.cleanup()


if __name__ == "__main__":
    unittest.main()
