from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from citra.context.workspace_context import _private_source_exclusions


class ProcessLogWorkspaceExclusionTests(unittest.TestCase):
    def test_process_logs_are_controller_private(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "project"
            source.mkdir()
            logs = source / ".citra.logs"
            logs.mkdir()
            library = root / "library"

            exclusions = _private_source_exclusions(source, library)

            self.assertIn(logs.resolve(), exclusions)


if __name__ == "__main__":
    unittest.main()
