from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from citra.config import LintContextConfig, LintRuleConfig
from citra.config._analysis import load_lint_config
from citra.sandbox import SandboxResult
from citra.tools.editing import Edit, Write
from citra.tools.editing._post_edit import post_edit_result
from citra.tools.linting import LintRunner


class FakeWorkspace:
    def __init__(self, project: Path) -> None:
        self.workspace = project

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return (
            path.resolve() if path.is_absolute() else (self.workspace / path).resolve()
        )


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


def test_lint_config_loads_auto_fix_command() -> None:
    config = load_lint_config(
        {
            "lint": {
                "auto_fix": False,
                "rules": [
                    {
                        "name": "ruff",
                        "command": ["ruff", "check", "{path}"],
                        "fix_command": ["ruff", "check", "--fix", "{path}"],
                    }
                ],
            }
        }
    )

    assert config.auto_fix is False
    assert config.rules[0].fix_command == (
        "ruff",
        "check",
        "--fix",
        "{path}",
    )


def test_lint_config_rejects_empty_fix_command() -> None:
    with pytest.raises(ValueError, match=r"lint\.rules\[0\]\.fix_command"):
        load_lint_config(
            {
                "lint": {
                    "rules": [
                        {
                            "name": "ruff",
                            "command": ["ruff", "check", "{path}"],
                            "fix_command": [],
                        }
                    ]
                }
            }
        )


def test_fix_command_runs_before_check(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "module.py"
    target.write_text("x = 1\n", encoding="utf-8")
    sandbox = FakeSandbox(SandboxResult(0, "", False))
    config = LintContextConfig(
        rules=(
            LintRuleConfig(
                name="pyrefly-custom",
                command=("pyrefly", "check", "{path}"),
                fix_command=("fixer", "{relative_path}"),
                include=("**/*.py",),
            ),
        ),
    )

    result = LintRunner(FakeWorkspace(project), sandbox, config).lint_for_path(
        "module.py"
    )

    assert result is None
    assert [call["command"] for call in sandbox.calls] == [
        ("fixer", "module.py"),
        ("pyrefly", "check", str(target)),
    ]


def test_auto_fix_can_be_disabled_without_disabling_checks(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "module.py"
    target.write_text("x = 1\n", encoding="utf-8")
    sandbox = FakeSandbox(SandboxResult(0, "", False))
    config = LintContextConfig(
        auto_fix=False,
        rules=(
            LintRuleConfig(
                name="pyrefly-custom",
                command=("pyrefly", "check", "{path}"),
                fix_command=("fixer", "{path}"),
                include=("**/*.py",),
            ),
        ),
    )

    LintRunner(FakeWorkspace(project), sandbox, config).lint_for_path("module.py")

    assert [call["command"] for call in sandbox.calls] == [
        ("pyrefly", "check", str(target)),
    ]


@pytest.mark.parametrize(
    ("configured", "requested", "expected_commands"),
    (
        (True, False, ("checker",)),
        (False, True, ("fixer", "checker")),
    ),
)
def test_auto_fix_can_be_overridden_for_one_tool_call(
    tmp_path: Path,
    configured: bool,
    requested: bool,
    expected_commands: tuple[str, ...],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("x = 1\n", encoding="utf-8")
    sandbox = FakeSandbox(SandboxResult(0, "", False))
    config = LintContextConfig(
        auto_fix=configured,
        rules=(
            LintRuleConfig(
                name="custom",
                command=("checker", "{path}"),
                fix_command=("fixer", "{path}"),
                include=("**/*.py",),
            ),
        ),
    )

    LintRunner(FakeWorkspace(project), sandbox, config).lint_for_path(
        "module.py",
        auto_fix=requested,
    )

    commands = tuple(call["command"][0] for call in sandbox.calls)
    assert tuple(command for command in commands if command != "pyrefly") == (
        expected_commands
    )


def test_edit_and_write_expose_auto_fix_override() -> None:
    for tool_type in (Edit, Write):
        parameters = tool_type.CITRA_DEFINITION.function.parameters.to_dict()

        assert parameters["properties"]["auto_fix"]["type"] == "boolean"
        assert "auto_fix" not in parameters["required"]


def test_post_edit_fixes_before_collecting_diagnostics() -> None:
    calls: list[str] = []
    context = SimpleNamespace(
        lint_for_path=lambda path: calls.append(f"lint:{path}") or None,
        diagnostics_for_path=lambda path: calls.append(f"lsp:{path}") or None,
    )

    assert post_edit_result(context, "module.py") == "ok"
    assert calls == ["lint:module.py", "lsp:module.py"]


def test_post_edit_forwards_auto_fix_override() -> None:
    calls: list[str] = []

    def lint_for_path(path: str, *, auto_fix: bool | None = None) -> None:
        calls.append(f"lint:{path}:{auto_fix}")

    context = SimpleNamespace(
        lint_for_path=lint_for_path,
        diagnostics_for_path=lambda path: calls.append(f"lsp:{path}") or None,
    )

    assert post_edit_result(context, "module.py", auto_fix=False) == "ok"
    assert calls == ["lint:module.py:False", "lsp:module.py"]


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
    assert sandbox.calls[0] == {
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
    assert sandbox.calls[1]["command"] == (
        "pyrefly",
        "check",
        "--preset",
        "default",
        "--search-path",
        str(project),
        str(target),
    )
    assert sandbox.calls[1]["cwd"] == project
    assert sandbox.calls[1]["timeout"] == 30
    assert sandbox.calls[1]["network"] is False


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
    commands = [call["command"] for call in sandbox.calls]
    assert commands[0][:3] == ("ruff", "check", "--fix")
    assert commands[1][:2] == ("ruff", "check")
    assert str(project / "pyproject.toml") in commands[0]
    assert str(project / "pyproject.toml") in commands[1]
    assert "@source" not in " ".join(commands[0])


def test_lint_workspace_placeholder_in_command_expands_to_project(
    tmp_path: Path,
) -> None:
    """``{workspace}`` in a command expands to the project root. Author: adamant-disaster."""
    project = tmp_path / "project"
    project.mkdir()
    target = project / "module.py"
    target.write_text("x = 1\n", encoding="utf-8")
    workspace = FakeWorkspace(project)
    sandbox = FakeSandbox(SandboxResult(0, "", False))
    config = LintContextConfig(
        enabled=True,
        rules=(
            LintRuleConfig(
                name="ruff-format",
                command=(
                    "ruff",
                    "format",
                    "--check",
                    "--force-exclude",
                    "--config",
                    "{workspace}/pyproject.toml",
                    "{path}",
                ),
                include=("**/*.py",),
                cwd=".",
            ),
        ),
    )

    result = LintRunner(workspace, sandbox, config).lint_for_path("module.py")

    assert result is None
    assert "unsupported placeholder" not in (result or "")
    assert sandbox.calls[0] == {
        "command": (
            "ruff",
            "format",
            "--check",
            "--force-exclude",
            "--config",
            f"{project}/pyproject.toml",
            str(target),
        ),
        "cwd": project,
        "timeout": 30,
        "network": False,
    }
    assert sandbox.calls[1]["command"] == (
        "pyrefly",
        "check",
        "--preset",
        "default",
        "--search-path",
        str(project),
        str(target),
    )


def test_lint_workspace_placeholder_in_cwd_expands_to_project(
    tmp_path: Path,
) -> None:
    """``{workspace}`` in ``cwd`` expands to the project root. Author: adamant-disaster."""
    project = tmp_path / "project"
    project.mkdir()
    target = project / "module.py"
    target.write_text("x = 1\n", encoding="utf-8")
    workspace = FakeWorkspace(project)
    sandbox = FakeSandbox(SandboxResult(0, "", False))
    config = LintContextConfig(
        enabled=True,
        rules=(
            LintRuleConfig(
                name="pyrefly",
                command=("pyrefly", "check", "{path}"),
                include=("**/*.py",),
                cwd="{workspace}",
            ),
        ),
    )

    result = LintRunner(workspace, sandbox, config).lint_for_path("module.py")

    assert result is None
    assert sandbox.calls == [
        {
            "command": ("pyrefly", "check", str(target)),
            "cwd": project,
            "timeout": 30,
            "network": False,
        }
    ]


def test_lint_workspace_alias_matches_project_expansion(tmp_path: Path) -> None:
    """``{workspace}`` and ``{project}`` expand identically. Author: adamant-disaster."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("x = 1\n", encoding="utf-8")
    workspace = FakeWorkspace(project)
    runner = LintRunner(
        workspace,
        FakeSandbox(SandboxResult(0, "", False)),
        LintContextConfig(enabled=True),
    )
    path = workspace.resolve_path("module.py")

    assert (
        runner._expand("{workspace}/pyproject.toml", path, "module.py")
        == runner._expand("{project}/pyproject.toml", path, "module.py")
        == f"{project}/pyproject.toml"
    )
    assert (
        runner._expand("{workspace}", path, "module.py")
        == runner._expand("{project}", path, "module.py")
        == str(project)
    )


def test_lint_rule_with_unsupported_placeholder_in_command_fails_fast(
    tmp_path: Path,
) -> None:
    """An unrecognized ``{...}`` token in a command fails fast and never invokes the sandbox. Author: adamant-disaster."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("x = 1\n", encoding="utf-8")
    workspace = FakeWorkspace(project)
    sandbox = FakeSandbox(SandboxResult(0, "", False))
    config = LintContextConfig(
        enabled=True,
        rules=(
            LintRuleConfig(
                name="ruff-format",
                command=(
                    "ruff",
                    "format",
                    "--check",
                    "--force-exclude",
                    "--config",
                    "{unknown}/pyproject.toml",
                    "{path}",
                ),
                include=("**/*.py",),
                cwd=".",
            ),
        ),
    )

    result = LintRunner(workspace, sandbox, config).lint_for_path("module.py")

    assert result is not None
    assert "unsupported placeholder '{unknown}'" in result
    assert "rule 'ruff-format'" in result
    assert "{path}, {relative_path}, {project}, {workspace}" in result
    assert "Sandbox working directory does not exist" not in result
    assert len(sandbox.calls) == 1
    assert "pyrefly" in str(sandbox.calls[0].get("command"))
    assert "ruff" not in str(sandbox.calls[0].get("command"))


def test_lint_rule_with_unsupported_placeholder_in_cwd_fails_fast(
    tmp_path: Path,
) -> None:
    """An unrecognized ``{...}`` token in ``cwd`` fails fast and never invokes the sandbox. Author: adamant-disaster."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("x = 1\n", encoding="utf-8")
    workspace = FakeWorkspace(project)
    sandbox = FakeSandbox(SandboxResult(0, "", False))
    config = LintContextConfig(
        enabled=True,
        rules=(
            LintRuleConfig(
                name="pyrefly",
                command=("pyrefly", "check", "{path}"),
                include=("**/*.py",),
                cwd="{unknown}",
            ),
        ),
    )

    result = LintRunner(workspace, sandbox, config).lint_for_path("module.py")

    assert result is not None
    assert "unsupported placeholder '{unknown}'" in result
    assert "rule 'pyrefly'" in result
    assert "in cwd" in result
    assert "Sandbox working directory does not exist" not in result
    assert sandbox.calls == []


def test_hardcoded_pyrefly_runs_for_python_with_empty_config(
    tmp_path: Path,
) -> None:
    """Empty operator config still runs hardcoded pyrefly. Author: liberating-potato."""
    project = tmp_path / "project"
    project.mkdir()
    target = project / "module.py"
    target.write_text("x = 1\n", encoding="utf-8")
    workspace = FakeWorkspace(project)
    sandbox = FakeSandbox(SandboxResult(0, "", False))

    result = LintRunner(
        workspace, sandbox, LintContextConfig(enabled=True)
    ).lint_for_path("module.py")

    assert result is None
    assert len(sandbox.calls) == 1
    assert sandbox.calls[0] == {
        "command": (
            "pyrefly",
            "check",
            "--preset",
            "default",
            "--search-path",
            str(project),
            str(target),
        ),
        "cwd": project,
        "timeout": 30,
        "network": False,
    }


def test_hardcoded_pyrefly_skipped_when_disabled(tmp_path: Path) -> None:
    """Master switch disables hardcoded pyrefly. Author: liberating-potato."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("x = 1\n", encoding="utf-8")
    workspace = FakeWorkspace(project)
    sandbox = FakeSandbox(SandboxResult(0, "", False))

    result = LintRunner(
        workspace, sandbox, LintContextConfig(enabled=False)
    ).lint_for_path("module.py")

    assert result is None
    assert sandbox.calls == []


def test_hardcoded_pyrefly_skipped_for_non_python(tmp_path: Path) -> None:
    """Hardcoded pyrefly is python-only. Author: liberating-potato."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "notes.txt").write_text("hello\n", encoding="utf-8")
    workspace = FakeWorkspace(project)
    sandbox = FakeSandbox(SandboxResult(0, "", False))

    result = LintRunner(
        workspace, sandbox, LintContextConfig(enabled=True)
    ).lint_for_path("notes.txt")

    assert result is None
    assert sandbox.calls == []


def test_hardcoded_pyrefly_skipped_when_unavailable(tmp_path: Path) -> None:
    """Missing pyrefly binary skips silently. Author: liberating-potato."""

    class NoPyreflyWorkspace(FakeWorkspace):
        def resolve_command(self, command: str) -> str | None:
            return None

    class NoPyreflySandbox(FakeSandbox):
        def resolve_command(self, command: str) -> str | None:
            return None

    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("x = 1\n", encoding="utf-8")
    workspace = NoPyreflyWorkspace(project)
    sandbox = NoPyreflySandbox(SandboxResult(0, "", False))

    result = LintRunner(
        workspace, sandbox, LintContextConfig(enabled=True)
    ).lint_for_path("module.py")

    assert result is None
    assert sandbox.calls == []


def test_hardcoded_pyrefly_skips_missing_binary_output(
    tmp_path: Path,
) -> None:
    """Missing-binary diagnostics skip instead of failing. Author: liberating-potato."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("x = 1\n", encoding="utf-8")
    workspace = FakeWorkspace(project)
    sandbox = FakeSandbox(
        SandboxResult(
            returncode=1,
            output="pyrefly: No such file or directory\n",
            timed_out=False,
        )
    )

    result = LintRunner(
        workspace, sandbox, LintContextConfig(enabled=True)
    ).lint_for_path("module.py")

    assert result is None
    assert len(sandbox.calls) == 1


def test_hardcoded_pyrefly_reports_type_errors(tmp_path: Path) -> None:
    """Default preset surfaces bad-assignment. Author: liberating-potato."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("x: int = 'hi'\n", encoding="utf-8")
    workspace = FakeWorkspace(project)
    sandbox = FakeSandbox(
        SandboxResult(
            returncode=1,
            output="ERROR bad-assignment [bad-assignment]\n",
            timed_out=False,
        )
    )

    result = LintRunner(
        workspace, sandbox, LintContextConfig(enabled=True)
    ).lint_for_path("module.py")

    assert result is not None
    assert "bad-assignment" in result
    assert "pyrefly" in result
