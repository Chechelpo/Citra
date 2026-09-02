from __future__ import annotations

from pathlib import Path

from citra.config import LintContextConfig, LintRuleConfig
from citra.sandbox import SandboxResult
from citra.tools.linting import LintRunner


class FakeWorkspace:
    def __init__(self, project: Path) -> None:
        self.workspace = project

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.workspace / path).resolve()


class FakeSandbox:
    def __init__(self, result: SandboxResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def run(self, command, *, cwd, timeout, network):
        self.calls.append(
            {
                "command": tuple(command),
                "cwd": Path(cwd),
                "timeout": timeout,
                "network": network,
            }
        )
        return self.result


def test_global_lint_rule_uses_current_project_placeholders(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "module.py"
    target.write_text("x = 1\n", encoding="utf-8")
    workspace = FakeWorkspace(project)
    sandbox = FakeSandbox(
        SandboxResult(returncode=1, output="E001\n", timed_out=False)
    )
    config = LintContextConfig(
        enabled=True,
        rules=(
            LintRuleConfig(
                name="check",
                command=("checker", "{project}", "{relative_path}", "{path}"),
                include=("**/*.py",),
                cwd=".",
            ),
        ),
    )

    result = LintRunner(workspace, sandbox, config).lint_for_path("module.py")

    assert result is not None and "E001" in result
    assert sandbox.calls == [
        {
            "command": (
                "checker",
                str(project),
                "module.py",
                str(target),
            ),
            "cwd": project,
            "timeout": 30,
            "network": False,
        }
    ]


def test_lint_does_not_run_for_nonproject_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    workspace = FakeWorkspace(project)
    sandbox = FakeSandbox(SandboxResult(0, "", False))
    config = LintContextConfig(
        enabled=True,
        rules=(LintRuleConfig(name="check", command=("checker", "{path}")),),
    )

    assert LintRunner(workspace, sandbox, config).lint_for_path(tmp_path / "other.py") is None
    assert sandbox.calls == []


def test_project_ruff_policy_uses_copied_pyproject(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("x=1\n", encoding="utf-8")
    (project / "pyproject.toml").write_text(
        "[tool.ruff.lint]\nselect = ['E']\n",
        encoding="utf-8",
    )
    workspace = FakeWorkspace(project)
    sandbox = FakeSandbox(SandboxResult(0, "", False))

    result = LintRunner(
        workspace,
        sandbox,
        LintContextConfig(enabled=True),
    ).lint_for_path("module.py")

    assert result is None
    command = sandbox.calls[0]["command"]
    assert str(project / "pyproject.toml") in command
    assert "@source" not in " ".join(command)
