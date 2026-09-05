from __future__ import annotations

import logging
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from citra.logging import Logger
from citra.sandbox import SandboxResult, SandboxedFilesystem
from citra.sandbox.filesystem_ops import (
    EditInput,
    FindInput,
    GlobInput,
    ReadBinaryInput,
    ReadInput,
    ReadRawInput,
    TreeInput,
    WriteInput,
)
from citra.utils.process_logging import process_log


class _FilesystemSandbox:
    """Return valid worker responses without invoking Bubblewrap."""

    _RESULTS = {
        "read": {"entries": [], "single_literal": True},
        "read_raw": {"content": ""},
        "read_binary": {
            "content_b64": "",
            "size": 0,
            "mime_type": None,
        },
        "write": {"status": "ok"},
        "edit": {"status": "ok"},
        "glob": {"paths": []},
        "tree": {
            "root": ".",
            "lines": [],
            "directories": 0,
            "files": 0,
            "skipped": 0,
            "directories_only": False,
        },
        "find": {
            "mode": "files",
            "paths": [],
            "truncated": False,
        },
    }

    @staticmethod
    def resolve_command(command: str) -> Path | None:
        if command == "citra-filesystem-python":
            return Path("/runtime/bin/citra-filesystem-python")
        return None

    @staticmethod
    def filesystem_environment() -> dict[str, str]:
        return {}

    def run(self, _command, **kwargs) -> SandboxResult:
        request = json.loads(kwargs["input_text"])
        response = {
            "ok": True,
            "result": self._RESULTS[request["operation"]],
        }
        return SandboxResult(0, json.dumps(response), False)


class ProcessLoggingTests(unittest.TestCase):
    def test_process_log_replaces_latest_log_and_captures_citra_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_directory = root / "logs"
            log_directory.mkdir()
            stale_log = log_directory / "latest.log"
            stale_log.write_text("stale process\n", encoding="utf-8")

            with process_log(log_directory) as log_path:
                logging.getLogger("citra.test").info("diagnostic marker")
                logging.getLogger("dependency.test").error("excluded marker")

            contents = log_path.read_text(encoding="utf-8")
            self.assertNotIn("stale process", contents)
            self.assertIn("Citra process started", contents)
            self.assertIn("diagnostic marker", contents)
            self.assertIn("[citra.test:", contents)
            self.assertNotIn("excluded marker", contents)
            self.assertIn("Citra process stopped normally", contents)
            self.assertEqual(stat.S_IMODE(log_directory.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(log_path.stat().st_mode), 0o600)
            self.assertFalse((log_directory / ".gitignore").exists())

    def test_process_log_records_unhandled_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_directory = Path(temporary) / "logs"
            with self.assertRaisesRegex(RuntimeError, "failure marker"):
                with process_log(log_directory) as log_path:
                    raise RuntimeError("failure marker")

            contents = log_path.read_text(encoding="utf-8")
            self.assertIn("Citra process terminated unexpectedly", contents)
            self.assertIn("RuntimeError: failure marker", contents)
            self.assertIn("Traceback (most recent call last)", contents)

    def test_installation_logs_only_supply_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            citra_root = root / "install" / ".citra"
            config_logs = citra_root / "logs"
            config_logs.mkdir(parents=True)
            (config_logs / "config.toml").write_text(
                'level = "trace"\n',
                encoding="utf-8",
            )
            runtime_logs = root / "citra-process-1-testxx" / "logs"

            with mock.patch.dict(
                "os.environ",
                {"CITRA_ROOT": str(citra_root)},
            ):
                logger = Logger("test")
                with process_log(runtime_logs):
                    logger.info("runtime-only marker")

            self.assertIn(
                "runtime-only marker",
                (runtime_logs / "latest.log").read_text(encoding="utf-8"),
            )
            self.assertFalse((config_logs / "latest.log").exists())

    def test_all_filesystem_operations_use_the_process_log(self) -> None:
        operations = (
            ReadInput.parse({"path": "a.txt"}),
            ReadRawInput.parse({"path": "a.txt"}),
            ReadBinaryInput.parse({"path": "a.png"}),
            WriteInput.parse({"path": "a.txt", "content": "x"}),
            EditInput.parse({"path": "a.txt", "old": "x", "new": "y"}),
            GlobInput.parse({"pat": "*.txt"}),
            TreeInput.parse({"path": "."}),
            FindInput.parse({"paths": ["src"]}),
        )

        with tempfile.TemporaryDirectory() as temporary:
            log_directory = Path(temporary) / "logs"
            filesystem = SandboxedFilesystem(_FilesystemSandbox())  # type: ignore[arg-type]
            with process_log(log_directory):
                for operation in operations:
                    filesystem.execute(operation)

            contents = (log_directory / "latest.log").read_text(
                encoding="utf-8"
            )
            for operation in operations:
                self.assertIn(
                    f"filesystem operation '{operation.operation}' completed",
                    contents,
                )


if __name__ == "__main__":
    unittest.main()
