from __future__ import annotations

from io import StringIO
import unittest

from citra.utils.terminal import terminal_bell


class PromptNotificationTests(unittest.TestCase):
    def test_terminal_bell_writes_and_flushes_once(self) -> None:
        class Stream(StringIO):
            flushed = False

            def flush(self) -> None:
                self.flushed = True
                super().flush()

        stream = Stream()
        terminal_bell(stream)
        self.assertEqual(stream.getvalue(), "\a")
        self.assertTrue(stream.flushed)


if __name__ == "__main__":
    unittest.main()
