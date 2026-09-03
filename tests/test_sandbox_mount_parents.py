from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from citra.config import SandboxPolicy
from citra.sandbox.sandbox import WorkspaceSandbox, _mount_parent_arguments


def _option_index(
    arguments: list[str],
    option: str,
    *values: str,
) -> int:
    width = 1 + len(values)
    expected = (option, *values)
    for index in range(len(arguments) - width + 1):
        if tuple(arguments[index : index + width]) == expected:
            return index
    raise AssertionError(f"Missing Bubblewrap arguments: {expected!r}")


class SandboxMountParentTests(unittest.TestCase):
    def test_build_bwrap_arguments_creates_bind_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "runtime" / "workspace"
            workspace.mkdir(parents=True)
            executable = Path("/usr/bin/bash")
            self.assertTrue(executable.is_file())

            policy = SandboxPolicy(
                runtime_results=[],
                base_readonly_binds=[executable],
                masked_host_dirs=[Path("/tmp")],
            )
            sandbox = WorkspaceSandbox(workspace, policy)

            arguments = sandbox.build_bwrap_arguments(
                command=(str(executable), "-c", "true"),
                cwd=workspace,
                network=False,
            )

            usr = _option_index(arguments, "--dir", "/usr")
            usr_bin = _option_index(arguments, "--dir", "/usr/bin")
            executable_bind = _option_index(
                arguments,
                "--ro-bind",
                str(executable),
                str(executable),
            )
            self.assertLess(usr, usr_bin)
            self.assertLess(usr_bin, executable_bind)

            tmp_mask = _option_index(arguments, "--tmpfs", "/tmp")
            workspace_parent = _option_index(
                arguments,
                "--dir",
                str(workspace.parent),
            )
            workspace_bind = _option_index(
                arguments,
                "--bind",
                str(workspace),
                str(workspace),
            )
            self.assertLess(tmp_mask, workspace_parent)
            self.assertLess(workspace_parent, workspace_bind)

            broad_bind = any(
                tuple(arguments[index : index + 3])
                == ("--ro-bind", "/usr/bin", "/usr/bin")
                for index in range(len(arguments) - 2)
            )
            self.assertFalse(broad_bind)

    def test_mount_parent_arguments_reject_relative_bind_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            _mount_parent_arguments((Path("relative/tool"),))


if __name__ == "__main__":
    unittest.main()
