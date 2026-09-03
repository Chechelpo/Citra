from __future__ import annotations

import logging
from pathlib import Path
import stat
import tempfile
import unittest

from citra.utils.process_logging import process_log


class ProcessLoggingTests(unittest.TestCase):
    def test_process_log_replaces_last_log_and_captures_citra_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_directory = root / ".citra.logs"
            log_directory.mkdir()
            stale_log = log_directory / "last.log"
            stale_log.write_text("stale process\n", encoding="utf-8")

            with process_log(root) as log_path:
                logging.getLogger("citra.test").info("diagnostic marker")
                logging.getLogger("dependency.test").error("excluded marker")

            contents = log_path.read_text(encoding="utf-8")
            self.assertNotIn("stale process", contents)
            self.assertIn("Citra process started", contents)
            self.assertIn("diagnostic marker", contents)
            self.assertNotIn("excluded marker", contents)
            self.assertIn("Citra process stopped normally", contents)
            self.assertEqual(stat.S_IMODE(log_directory.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(log_path.stat().st_mode), 0o600)
            self.assertEqual(
                (log_directory / ".gitignore").read_text(encoding="utf-8"),
                "*\n",
            )

    def test_process_log_records_unhandled_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "failure marker"):
                with process_log(root) as log_path:
                    raise RuntimeError("failure marker")

            contents = log_path.read_text(encoding="utf-8")
            self.assertIn("Citra process terminated unexpectedly", contents)
            self.assertIn("RuntimeError: failure marker", contents)
            self.assertIn("Traceback (most recent call last)", contents)


if __name__ == "__main__":
    unittest.main()
