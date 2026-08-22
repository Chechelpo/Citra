from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from citra.tools.transient.browser import Browser
from citra.tools.transient.subprocess import Subprocess


class BrowserToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        workspace = SimpleNamespace(
            require_writable_path=lambda value: (root / str(value)).resolve(),
            display_path=lambda value: str(Path(value).resolve().relative_to(root)),
        )
        self.manager = mock.Mock()
        self.manager.request.return_value = {
            "ok": True,
            "result": {"url": "http://127.0.0.1:5173", "title": "App"},
        }
        self.context = SimpleNamespace(
            workspace=workspace,
            browser=self.manager,
            config=SimpleNamespace(
                browser=SimpleNamespace(
                    always_allow_network=False,
                    permission_timeout=30,
                )
            ),
        )
        self.tool = Browser(self.context)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_open_denial_is_fail_closed(self) -> None:
        with mock.patch(
            "citra.tools.transient.browser.PromptUser._execute",
            return_value="Deny",
        ):
            result = self.tool._execute(
                {
                    "action": "open",
                    "url": "http://127.0.0.1:5173",
                    "reason": "Test the local application.",
                }
            )
        self.assertIn("permission-denied", result)
        self.manager.request.assert_not_called()

    def test_open_approval_reaches_worker(self) -> None:
        with mock.patch(
            "citra.tools.transient.browser.PromptUser._execute",
            return_value="Allow once",
        ) as prompt:
            self.tool._execute(
                {
                    "action": "open",
                    "url": "http://127.0.0.1:5173",
                    "reason": "Exercise the UI.",
                }
            )
        question = prompt.call_args.args[0]["question"]
        self.assertIn("http://127.0.0.1:5173", question)
        self.assertIn("Exercise the UI.", question)
        self.manager.request.assert_called_once()


class SubprocessToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = mock.Mock()
        self.manager.start.return_value = 7
        workspace = SimpleNamespace(
            workspace=Path("/agent/workspace"),
            resolve_path=lambda value: Path(value),
            display_path=lambda value: str(value),
        )
        self.context = SimpleNamespace(
            workspace=workspace,
            subprocesses=self.manager,
            config=SimpleNamespace(
                subprocess=SimpleNamespace(
                    always_allow_network=False,
                    permission_timeout=30,
                )
            ),
        )
        self.tool = Subprocess(self.context)

    def test_network_start_shows_command_and_reason(self) -> None:
        with mock.patch.object(Path, "is_dir", return_value=True), mock.patch(
            "citra.tools.transient.subprocess.PromptUser._execute",
            return_value="Allow once",
        ) as prompt:
            result = self.tool._execute(
                {
                    "action": "start",
                    "cmd": "npm run dev -- --host 127.0.0.1",
                    "network": True,
                    "reason": "Expose the local test server to Chromium.",
                }
            )
        question = prompt.call_args.args[0]["question"]
        self.assertIn("npm run dev", question)
        self.assertIn("Expose the local test server", question)
        self.assertEqual(result, "Started subprocess 7.")
        self.assertTrue(self.manager.start.call_args.kwargs["network"])

    def test_network_start_requires_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "reason"):
            self.tool._execute(
                {"action": "start", "cmd": "npm run dev", "network": True}
            )
        self.manager.start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
