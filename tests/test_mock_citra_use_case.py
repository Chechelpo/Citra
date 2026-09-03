from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from citra.config.config_loader import WorkspaceContextConfig
from citra.context.workspace_context import WorkspaceContext
from citra.sandbox import WorkspaceSandbox
from citra.sandbox import SandboxedFilesystem


class MockCitraFilesystemUseCaseTests(unittest.TestCase):
    """Exercise the real filesystem-worker path with a Citra-shaped layout."""

    def test_read_write_worker_survives_masked_install_and_source_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_raw:
            temporary = Path(temporary_raw)
            host_home = temporary / "home"
            host_mnt = temporary / "mnt"
            host_tmp = temporary / "tmp"
            missing_media = temporary / "media"

            citra_root = host_home / "felipey" / "Code" / "Citra"
            citra_src = citra_root / "src"
            virtual_environment = citra_root / ".venv"
            virtual_environment_bin = virtual_environment / "bin"
            virtual_environment_bin.mkdir(parents=True)
            citra_src.mkdir(parents=True)

            mock_python = virtual_environment_bin / "python"
            mock_python.symlink_to(Path(sys.executable).resolve())

            source = (
                host_mnt
                / "1tb"
                / "AI"
                / "SillyTavern"
                / "SillyTavern-Launcher"
                / "SillyTavern"
                / "public"
                / "scripts"
                / "extensions"
                / "third-party"
                / "ComfyInject-custom"
            )
            source.mkdir(parents=True)
            specification = source / "SPEC.md"
            specification.write_text(
                "Implement the extension specification.\n",
                encoding="utf-8",
            )
            host_tmp.mkdir()

            bwrap_log = temporary / "bwrap.jsonl"
            fake_bwrap = temporary / "bwrap"
            fake_bwrap.write_text(
                self._fake_bwrap_source(bwrap_log),
                encoding="utf-8",
            )
            fake_bwrap.chmod(0o755)

            workspace = WorkspaceContext.create(
                config=WorkspaceContextConfig(
                    temporary_workspace=str(host_tmp),
                    permanent_workspace=None,
                ),
                workspace=source,
            )

            try:
                sandbox = WorkspaceSandbox(workspace)
                filesystem = SandboxedFilesystem(sandbox)

                with mock.patch(
                    "citra.sandbox.sandbox.MASKED_HOST_DIRS",
                    (
                        str(host_home),
                        str(host_mnt),
                        str(host_tmp),
                        str(missing_media),
                    ),
                ), mock.patch(
                    "citra.sandbox.sandbox.shutil.which",
                    return_value=str(fake_bwrap),
                ), mock.patch.object(
                    WorkspaceSandbox,
                    "_citra_runtime_readonly_binds",
                    return_value=(
                        citra_src,
                        virtual_environment,
                    ),
                ), mock.patch(
                    "citra.sandbox.sandboxed_filesystem.sys.executable",
                    str(mock_python),
                ):
                    read_result = filesystem.execute(
                        "read",
                        {
                            "path": "@source/SPEC.md",
                        },
                    )
                    write_result = filesystem.execute(
                        "write",
                        {
                            "path": "notes/plan.txt",
                            "content": "Implementation plan\n",
                        },
                    )
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "read-only",
                    ):
                        filesystem.execute(
                            "write",
                            {
                                "path": "@source/SPEC.md",
                                "content": "forbidden\n",
                            },
                        )

                self.assertIn(
                    "Implement the extension specification.",
                    read_result,
                )
                self.assertEqual(write_result, "ok")
                self.assertEqual(
                    (workspace.workspace / "notes" / "plan.txt").read_text(
                        encoding="utf-8"
                    ),
                    "Implementation plan\n",
                )
                self.assertEqual(
                    specification.read_text(encoding="utf-8"),
                    "Implement the extension specification.\n",
                )

                invocations = [
                    json.loads(line)
                    for line in bwrap_log.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(len(invocations), 3)

                for arguments in invocations:
                    self.assertIn("--unshare-net", arguments)
                    self.assertNotIn(str(missing_media), arguments)
                    self._assert_fd_mount(arguments, virtual_environment)
                    self._assert_fd_mount(arguments, source)
                    self._assert_fd_mount(arguments, workspace.workspace)
                    self.assertEqual(
                        arguments[arguments.index("--") + 1],
                        str(mock_python),
                    )

                first = invocations[0]
                self.assertIn(
                    ["--dir", str(host_home / "felipey")],
                    self._windows(first, 2),
                )
                self.assertIn(
                    ["--dir", str(citra_root)],
                    self._windows(first, 2),
                )
                self.assertIn(
                    ["--dir", str(host_mnt / "1tb")],
                    self._windows(first, 2),
                )
            finally:
                workspace.cleanup()

    @staticmethod
    def _assert_fd_mount(arguments: list[str], target: Path) -> None:
        matches = [
            window
            for window in MockCitraFilesystemUseCaseTests._windows(
                arguments,
                3,
            )
            if window[0] in {"--ro-bind-fd", "--bind-fd"}
            and window[2] == str(target)
        ]
        if not matches:
            raise AssertionError(f"No descriptor mount found for {target}")

    @staticmethod
    def _windows(values: list[str], size: int) -> list[list[str]]:
        return [
            values[index:index + size]
            for index in range(len(values) - size + 1)
        ]

    @staticmethod
    def _fake_bwrap_source(log: Path) -> str:
        return textwrap.dedent(
            f"""\
            #!{Path(sys.executable).resolve()}
            import json
            import os
            from pathlib import Path
            import sys

            arguments = sys.argv[1:]
            for index, argument in enumerate(arguments[:-2]):
                if argument in {{"--ro-bind-fd", "--bind-fd"}}:
                    os.fstat(int(arguments[index + 1]))

            with Path({str(log)!r}).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(arguments) + "\\n")

            separator = arguments.index("--")
            command = arguments[separator + 1:]
            chdir = arguments[arguments.index("--chdir") + 1]
            os.chdir(chdir)
            os.execvpe(command[0], command, os.environ)
            """
        )


if __name__ == "__main__":
    unittest.main()
