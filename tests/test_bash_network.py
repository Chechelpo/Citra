from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from citra.tools.transient.bash import Bash
from citra.sandbox import SandboxResult


class BashNetworkTests(unittest.TestCase):
    def _tool(self, *, always_allow: bool = False) -> tuple[Bash, object]:
        sandbox = mock.Mock()
        sandbox.run.return_value = SandboxResult(0, "ok", False)
        workspace = SimpleNamespace(
            workspace=Path("/agent/workspace"),
            resolve_path=lambda value: Path(value),
            display_path=lambda value: str(value),
        )
        context = SimpleNamespace(
            has_command=lambda command: command == "bash",
            workspace=workspace,
            sandbox=sandbox,
            config=SimpleNamespace(
                bash=SimpleNamespace(
                    always_allow_network=always_allow,
                    permission_timeout=30,
                )
            ),
        )
        return Bash(context), context

    def test_local_command_remains_network_isolated(self) -> None:
        tool, context = self._tool()
        with mock.patch.object(Path, "is_dir", return_value=True):
            result = tool._execute({"cmd": "python -m unittest"})
        self.assertEqual(result, "ok")
        self.assertFalse(context.sandbox.run.call_args.kwargs["network"])

    def test_network_requires_reason(self) -> None:
        tool, context = self._tool()
        with self.assertRaisesRegex(ValueError, "reason"):
            tool._execute({"cmd": "python -m pip index versions example", "network": True})
        context.sandbox.run.assert_not_called()

    def test_denial_prevents_execution(self) -> None:
        tool, context = self._tool()
        with mock.patch.object(Path, "is_dir", return_value=True), mock.patch(
            "citra.tools.transient.bash.PromptUser._execute",
            return_value="Deny",
        ) as prompt:
            result = tool._execute(
                {
                    "cmd": "python -m pip index versions example",
                    "network": True,
                    "reason": "Download public test data.",
                }
            )
        question = prompt.call_args.args[0]["question"]
        self.assertIn("python -m pip index", question)
        self.assertIn("Download public test data.", question)
        self.assertIn("permission-denied", result)
        context.sandbox.run.assert_not_called()

    def test_approval_enables_network(self) -> None:
        tool, context = self._tool()
        with mock.patch.object(Path, "is_dir", return_value=True), mock.patch(
            "citra.tools.transient.bash.PromptUser._execute",
            return_value="Allow once",
        ):
            tool._execute(
                {
                    "cmd": "python -m pip index versions example",
                    "network": True,
                    "reason": "Obtain the repository requested by the user.",
                }
            )
        self.assertTrue(context.sandbox.run.call_args.kwargs["network"])

    def test_always_allow_skips_prompt(self) -> None:
        tool, context = self._tool(always_allow=True)
        with mock.patch.object(Path, "is_dir", return_value=True), mock.patch(
            "citra.tools.transient.bash.PromptUser._execute"
        ) as prompt:
            tool._execute(
                {
                    "cmd": "python -m pip index versions example",
                    "network": True,
                    "reason": "Check endpoint availability.",
                }
            )
        prompt.assert_not_called()
        self.assertTrue(context.sandbox.run.call_args.kwargs["network"])

    def test_git_mutation_is_reserved_for_constrained_tools(self) -> None:
        tool, context = self._tool()
        for command in ("git add .", "git commit -m bad", "git restore file.py"):
            with self.subTest(command=command), self.assertRaisesRegex(
                ValueError,
                "Git commands are not available",
            ):
                tool._execute({"cmd": command})
        context.sandbox.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
