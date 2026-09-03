from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from citra.commands.default_registry import COMMAND_REGISTRY
from citra.commands.workspace import WorkspaceCommand


def _command(project: Path) -> WorkspaceCommand:
    context = SimpleNamespace(
        workspace=SimpleNamespace(workspace=project),
    )
    return WorkspaceCommand(context)


def test_workspace_command_is_registered() -> None:
    assert COMMAND_REGISTRY.contains("workspace")


def test_workspace_command_shows_path_and_commit_instructions(
    tmp_path: Path,
) -> None:
    result = _command(tmp_path).run("")

    assert f"Project checkout: {tmp_path.resolve()}" in result.output
    assert "/workspace shell" in result.output
    assert "/apply" in result.output


def test_workspace_shell_opens_in_checkout(tmp_path: Path) -> None:
    completed = SimpleNamespace(returncode=0)
    with mock.patch.dict("os.environ", {"SHELL": "/bin/test-shell"}), mock.patch(
        "citra.commands.workspace.subprocess.run",
        return_value=completed,
    ) as run:
        result = _command(tmp_path).run("shell")

    run.assert_called_once()
    assert run.call_args.args[0] == ["/bin/test-shell", "-i"]
    assert run.call_args.kwargs["cwd"] == tmp_path.resolve()
    assert (
        run.call_args.kwargs["env"]["CITRA_PROJECT_ROOT"]
        == str(tmp_path.resolve())
    )
    assert "exit code 0" in result.output


def test_workspace_command_rejects_unknown_action(tmp_path: Path) -> None:
    assert "Usage:" in _command(tmp_path).run("unknown").output
