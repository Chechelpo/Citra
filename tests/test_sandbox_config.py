from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from citra.sandbox import WorkspaceSandbox


class SandboxConfigTests(unittest.TestCase):
    def test_extra_readonly_bind_comes_from_config_and_expands_home(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "home"
            uv_root = home / ".local" / "share" / "uv"
            uv_root.mkdir(parents=True)

            sandbox = WorkspaceSandbox(
                SimpleNamespace(),
                config=SimpleNamespace(
                    extra_readonly_binds=("~/.local/share/uv",),
                ),
            )

            with mock.patch.dict(os.environ, {"HOME": str(home)}), mock.patch.object(
                sandbox,
                "_compatibility_readonly_binds",
                return_value=[],
            ), mock.patch.object(
                sandbox,
                "_citra_runtime_readonly_binds",
                return_value=(),
            ), mock.patch.object(
                sandbox,
                "_command_runtime_readonly_binds",
                return_value=(),
            ):
                mounts = sandbox._open_readonly_mounts(("bash",), {})

            try:
                self.assertEqual(
                    tuple(mount.target for mount in mounts),
                    (uv_root,),
                )
            finally:
                sandbox._close_mounts(mounts)


if __name__ == "__main__":
    unittest.main()
