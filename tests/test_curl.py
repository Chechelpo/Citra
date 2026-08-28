from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from citra.tools.transient.curl import Curl
from citra.sandbox import SandboxResult


class CurlTests(unittest.TestCase):
    def _tool(self, *, always_allow: bool = False) -> tuple[Curl, object]:
        sandbox = mock.Mock()
        sandbox.run.return_value = SandboxResult(0, "response", False)
        temporary = tempfile.TemporaryDirectory()
        workspace_root = Path(temporary.name)

        def require_writable_path(value: str) -> Path:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = workspace_root / candidate
            resolved = candidate.resolve()
            resolved.relative_to(workspace_root)
            return resolved

        context = SimpleNamespace(
            has_command=lambda command: command == "curl",
            config=SimpleNamespace(
                curl=SimpleNamespace(
                    always_allow_network=always_allow,
                    permission_timeout=30,
                    default_timeout=30,
                    max_timeout=300,
                    max_output_length=100_000,
                )
            ),
            workspace=SimpleNamespace(
                home=workspace_root / "home",
                require_writable_path=require_writable_path,
                display_path=lambda path: str(
                    Path(path).resolve().relative_to(workspace_root)
                ),
            ),
            sandbox=sandbox,
            _temporary=temporary,
        )
        return Curl(context), context

    def test_denial_does_not_execute_curl(self) -> None:
        tool, context = self._tool()
        with mock.patch(
            "citra.tools.transient.curl.PromptUser._execute",
            return_value="Deny",
        ):
            result = tool._execute({"url": "https://example.com"})
        self.assertIn("permission-denied", result)
        context.sandbox.run.assert_not_called()

    def test_explicit_permission_enables_network_for_one_request(self) -> None:
        tool, context = self._tool()
        with mock.patch(
            "citra.tools.transient.curl.PromptUser._execute",
            return_value="Allow once",
        ):
            result = tool._execute(
                {
                    "url": "https://example.com/api",
                    "method": "POST",
                    "headers": ["Content-Type: application/json"],
                    "data": "{}",
                }
            )
        self.assertEqual(result, "response")
        _, kwargs = context.sandbox.run.call_args
        self.assertTrue(kwargs["network"])
        command = context.sandbox.run.call_args.args[0]
        self.assertIn("--data-raw", command)
        self.assertEqual(command[-1], "https://example.com/api")

    def test_always_allow_skips_prompt(self) -> None:
        tool, context = self._tool(always_allow=True)
        with mock.patch(
            "citra.tools.transient.curl.PromptUser._execute"
        ) as prompt:
            tool._execute({"url": "http://localhost:8080/health"})
        prompt.assert_not_called()
        context.sandbox.run.assert_called_once()

    def test_rejects_credentials_and_header_newlines(self) -> None:
        tool, _ = self._tool(always_allow=True)
        with self.assertRaisesRegex(ValueError, "Credentials"):
            tool._execute({"url": "https://user:secret@example.com"})
        with self.assertRaisesRegex(ValueError, "newlines"):
            tool._execute(
                {
                    "url": "https://example.com",
                    "headers": ["X-Test: safe\r\nX-Injected: yes"],
                }
            )

    def test_download_uses_sandboxed_destination(self) -> None:
        tool, context = self._tool(always_allow=True)
        result = tool._execute(
            {
                "url": "https://example.com/archive.zip",
                "download_to": "downloads/archive.zip",
            }
        )
        command = context.sandbox.run.call_args.args[0]
        self.assertIn("--create-dirs", command)
        self.assertIn("--remove-on-error", command)
        output_index = command.index("--output")
        self.assertTrue(command[output_index + 1].endswith("downloads/archive.zip"))
        self.assertEqual(result, "Downloaded to downloads/archive.zip")

    def test_download_does_not_overwrite_without_permission(self) -> None:
        tool, context = self._tool(always_allow=True)
        destination = Path(context._temporary.name) / "existing.bin"
        destination.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            tool._execute(
                {
                    "url": "https://example.com/file",
                    "download_to": "existing.bin",
                }
            )
        context.sandbox.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
