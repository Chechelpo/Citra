from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from citra.context import CitraConfig, LintContextConfig, LintRuleConfig
from citra.tools.linting import LintRunner
from citra.tools.transient import Edit, Write
from citra.sandbox import SandboxResult


class FakeWorkspace:
    def __init__(self, root: Path) -> None:
        self.workspace = root / "workspace"
        self.source_workspace = root / "source"
        self.workspace.mkdir()
        self.source_workspace.mkdir()

    def resolve_path(self, raw: str | Path) -> Path:
        value = str(raw)
        if value == "@source":
            return self.source_workspace
        if value.startswith("@source/"):
            return self.source_workspace / value[len("@source/"):]
        path = Path(value)
        if path.is_absolute():
            return path.resolve()
        return (self.workspace / path).resolve()


class FakeSandbox:
    def __init__(self, results: list[SandboxResult]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, object]] = []

    def run(self, command, *, cwd, timeout, network, **kwargs):
        self.calls.append(
            {
                "command": tuple(command),
                "cwd": Path(cwd),
                "timeout": timeout,
                "network": network,
            }
        )
        return self.results.pop(0)


class MutatingFilesystem:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute(self, operation: str, arguments: dict[str, object]) -> str:
        self.calls.append((operation, dict(arguments)))
        return "ok"


def _rule(**overrides) -> LintRuleConfig:
    values = {
        "name": "python-style",
        "command": (
            "ruff",
            "check",
            "--config",
            "{source}/pyproject.toml",
            "{path}",
        ),
        "include": ("**/*.py",),
        "exclude": ("generated/**",),
        "cwd": "@source",
    }
    values.update(overrides)
    return LintRuleConfig(**values)


def test_lint_runner_is_disabled_until_rules_are_configured(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path)
    sandbox = FakeSandbox([])
    runner = LintRunner(workspace, sandbox, LintContextConfig())

    assert runner.lint_for_path("module.py") is None
    assert sandbox.calls == []


def test_lint_runner_matches_file_and_expands_project_placeholders(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path)
    sandbox = FakeSandbox(
        [SandboxResult(returncode=1, output="E501 too long\n", timed_out=False)]
    )
    runner = LintRunner(
        workspace,
        sandbox,
        LintContextConfig(enabled=True, rules=(_rule(),)),
    )

    result = runner.lint_for_path("module.py")

    assert result is not None
    assert "Lint violations for module.py" in result
    assert "[python-style]" in result
    assert "E501 too long" in result
    assert "exit code 1" in result
    assert sandbox.calls == [
        {
            "command": (
                "ruff",
                "check",
                "--config",
                str(workspace.source_workspace / "pyproject.toml"),
                str(workspace.workspace / "module.py"),
            ),
            "cwd": workspace.source_workspace,
            "timeout": 30,
            "network": False,
        }
    ]


def test_lint_runner_skips_excluded_nonmatching_and_nonproject_paths(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path)
    sandbox = FakeSandbox([])
    runner = LintRunner(
        workspace,
        sandbox,
        LintContextConfig(enabled=True, rules=(_rule(),)),
    )

    assert runner.lint_for_path("notes.txt") is None
    assert runner.lint_for_path("generated/schema.py") is None
    assert runner.lint_for_path("generated/deep/schema.py") is None
    assert runner.lint_for_path("@source/module.py") is None
    assert sandbox.calls == []


def test_lint_runner_clean_result_is_silent_and_timeout_is_reported(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path)
    sandbox = FakeSandbox(
        [
            SandboxResult(returncode=0, output="", timed_out=False),
            SandboxResult(returncode=-9, output="partial", timed_out=True),
        ]
    )
    runner = LintRunner(
        workspace,
        sandbox,
        LintContextConfig(enabled=True, timeout=7, rules=(_rule(),)),
    )

    assert runner.lint_for_path("first.py") is None
    timeout = runner.lint_for_path("second.py")
    assert timeout is not None
    assert "partial" in timeout
    assert "timed out after 7s" in timeout


def test_lint_runner_reports_execution_failures_and_truncates_output(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path)

    class ExplodingSandbox:
        def run(self, *args, **kwargs):
            raise RuntimeError("configured executable is unavailable")

    runner = LintRunner(
        workspace,
        ExplodingSandbox(),
        LintContextConfig(
            enabled=True,
            max_output_length=80,
            rules=(_rule(),),
        ),
    )

    result = runner.lint_for_path("module.py")
    assert result is not None
    assert len(result) > 80
    assert result.startswith("Lint violations for module.py")
    assert "<truncated" in result


def test_lint_runner_executes_each_matching_rule_without_network(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path)
    sandbox = FakeSandbox(
        [
            SandboxResult(returncode=0, output="clean", timed_out=False),
            SandboxResult(returncode=2, output="style violation", timed_out=False),
        ]
    )
    runner = LintRunner(
        workspace,
        sandbox,
        LintContextConfig(
            enabled=True,
            rules=(
                _rule(name="ruff"),
                _rule(name="project-policy", command=("policy-check", "{relative_path}")),
            ),
        ),
    )

    result = runner.lint_for_path("src/module.py")
    assert result is not None
    assert "[ruff]" not in result
    assert "[project-policy]" in result
    assert "style violation" in result
    assert len(sandbox.calls) == 2
    assert all(call["network"] is False for call in sandbox.calls)


def test_edit_and_write_always_run_diagnostics_and_lint_after_success() -> None:
    filesystem = MutatingFilesystem()
    diagnostics_calls: list[str] = []
    lint_calls: list[str] = []
    context = SimpleNamespace(
        filesystem=filesystem,
        diagnostics_for_path=lambda path: diagnostics_calls.append(path) or "type error",
        lint_for_path=lambda path: lint_calls.append(path) or "style error",
    )

    edited = Edit(context)._execute(
        {
            "path": "module.py",
            "old": "before",
            "new": "after",
            "diagnostics": False,
        }
    )
    written = Write(context)._execute(
        {
            "path": "module.py",
            "content": "after\n",
        }
    )

    for result in (edited, written):
        assert result.startswith("ok\n\n")
        assert "LSP diagnostics after edit:\ntype error" in result
        assert "Lint checks after edit:\nstyle error" in result
    assert diagnostics_calls == ["module.py", "module.py"]
    assert lint_calls == ["module.py", "module.py"]
    assert filesystem.calls[0][1] == {
        "path": "module.py",
        "old": "before",
        "new": "after",
    }


def test_post_edit_checks_do_not_run_when_filesystem_mutation_fails() -> None:
    class FailedFilesystem:
        def execute(self, operation, arguments):
            return "error: mutation failed"

    diagnostics = []
    lint = []
    context = SimpleNamespace(
        filesystem=FailedFilesystem(),
        diagnostics_for_path=lambda path: diagnostics.append(path),
        lint_for_path=lambda path: lint.append(path),
    )

    assert Edit(context)._execute(
        {"path": "a.py", "old": "x", "new": "y"}
    ) == "error: mutation failed"
    assert diagnostics == []
    assert lint == []


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lint: str) -> CitraConfig:
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(
        f'''\
[model]
host = "https://model.invalid/v1"
api_key = "secret"
id = "test-model"
max_tokens = 128

[web-search]
host_url = "http://search.invalid"

[workspace]
temporary_workspace = "{tmp_path / 'agent'}"
permanent_workspace = "{source}"

{lint}
''',
        encoding="utf-8",
    )
    monkeypatch.setenv("CITRA_CONFIG_PATH", str(config))
    return CitraConfig.load()


def test_configured_lint_rules_enable_linting_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(
        tmp_path,
        monkeypatch,
        '''\
[lint]
timeout = 12
max_output_length = 4096

[[lint.rules]]
name = "ruff"
command = ["ruff", "check", "--config", "{source}/pyproject.toml", "{path}"]
include = ["**/*.py"]
exclude = ["generated/**"]
cwd = "@source"
''',
    )

    assert config.lint.enabled is True
    assert config.lint.timeout == 12
    assert config.lint.max_output_length == 4096
    assert config.lint.rules == (
        LintRuleConfig(
            name="ruff",
            command=("ruff", "check", "--config", "{source}/pyproject.toml", "{path}"),
            include=("**/*.py",),
            exclude=("generated/**",),
            cwd="@source",
        ),
    )


def test_lint_config_rejects_invalid_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match=r"lint.rules\[0\]\.command"):
        _write_config(
            tmp_path,
            monkeypatch,
            '''\
[lint]

[[lint.rules]]
name = "broken"
command = []
''',
        )


def test_lint_config_allows_enabled_without_global_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(
        tmp_path,
        monkeypatch,
        "[lint]\nenabled = true\n",
    )

    assert config.lint.enabled is True
    assert config.lint.rules == ()


def test_lint_config_can_keep_rules_explicitly_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(
        tmp_path,
        monkeypatch,
        '''\
[lint]
enabled = false

[[lint.rules]]
name = "ruff"
command = ["ruff", "check", "{path}"]
include = ["**/*.py"]
''',
    )
    assert config.lint.enabled is False
    assert len(config.lint.rules) == 1


def test_source_pyproject_ruff_lint_precedes_global_policy(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path)
    (workspace.source_workspace / "pyproject.toml").write_text(
        '''\
[project]
name = "example"
version = "0.1.0"

[tool.ruff]
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "B"]
''',
        encoding="utf-8",
    )
    sandbox = FakeSandbox(
        [SandboxResult(returncode=1, output="F401 unused import", timed_out=False)]
    )
    runner = LintRunner(
        workspace,
        sandbox,
        LintContextConfig(
            enabled=True,
            rules=(
                _rule(
                    name="global-policy",
                    command=("global-lint", "{path}"),
                ),
            ),
        ),
    )

    result = runner.lint_for_path("src/module.py")

    assert result is not None
    assert "[ruff-project]" in result
    assert "[global-policy]" not in result
    assert sandbox.calls == [
        {
            "command": (
                "ruff",
                "check",
                "--force-exclude",
                "--config",
                str(workspace.source_workspace / "pyproject.toml"),
                str(workspace.workspace / "src/module.py"),
            ),
            "cwd": workspace.workspace,
            "timeout": 30,
            "network": False,
        }
    ]


def test_source_ruff_project_policy_works_without_global_rules(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path)
    (workspace.source_workspace / "pyproject.toml").write_text(
        "[tool.ruff.lint]\nselect = ['F']\n",
        encoding="utf-8",
    )
    sandbox = FakeSandbox(
        [SandboxResult(returncode=0, output="", timed_out=False)]
    )
    runner = LintRunner(workspace, sandbox, LintContextConfig())

    assert runner.lint_for_path("module.py") is None
    assert len(sandbox.calls) == 1
    assert sandbox.calls[0]["command"][0:2] == ("ruff", "check")


def test_lint_master_switch_disables_project_and_global_rules(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path)
    (workspace.source_workspace / "pyproject.toml").write_text(
        "[tool.ruff.lint]\nselect = ['F']\n",
        encoding="utf-8",
    )
    sandbox = FakeSandbox([])
    runner = LintRunner(
        workspace,
        sandbox,
        LintContextConfig(
            enabled=False,
            rules=(
                _rule(name="global-policy", command=("global-lint", "{path}")),
            ),
        ),
    )

    assert runner.lint_for_path("module.py") is None
    assert sandbox.calls == []


def test_source_ruff_format_is_checked_only_when_project_declares_format(
    tmp_path: Path,
) -> None:
    workspace = FakeWorkspace(tmp_path)
    pyproject = workspace.source_workspace / "pyproject.toml"
    pyproject.write_text(
        '''\
[tool.ruff.lint]
select = ["F"]

[tool.ruff.format]
quote-style = "single"
''',
        encoding="utf-8",
    )
    sandbox = FakeSandbox(
        [
            SandboxResult(returncode=0, output="", timed_out=False),
            SandboxResult(returncode=1, output="Would reformat: module.py", timed_out=False),
        ]
    )
    runner = LintRunner(workspace, sandbox, LintContextConfig())

    result = runner.lint_for_path("module.py")

    assert result is not None
    assert "[ruff-format-project]" in result
    assert len(sandbox.calls) == 2
    assert sandbox.calls[0]["command"][0:2] == ("ruff", "check")
    assert sandbox.calls[1]["command"][0:3] == ("ruff", "format", "--check")


def test_source_pyproject_without_declared_linter_falls_back_to_global(
    tmp_path: Path,
) -> None:
    workspace = FakeWorkspace(tmp_path)
    (workspace.source_workspace / "pyproject.toml").write_text(
        '''\
[project]
name = "example"
version = "0.1.0"

[tool.ruff]
line-length = 100
''',
        encoding="utf-8",
    )
    sandbox = FakeSandbox(
        [SandboxResult(returncode=0, output="", timed_out=False)]
    )
    runner = LintRunner(
        workspace,
        sandbox,
        LintContextConfig(
            enabled=True,
            rules=(
                _rule(
                    name="global-policy",
                    command=("global-lint", "{path}"),
                ),
            ),
        ),
    )

    assert runner.lint_for_path("module.py") is None
    assert sandbox.calls[0]["command"][0] == "global-lint"


def test_source_project_policy_is_not_merged_with_global_rules(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path)
    (workspace.source_workspace / "pyproject.toml").write_text(
        "[tool.ruff.lint]\nselect = ['F']\n",
        encoding="utf-8",
    )
    sandbox = FakeSandbox(
        [SandboxResult(returncode=0, output="", timed_out=False)]
    )
    runner = LintRunner(
        workspace,
        sandbox,
        LintContextConfig(
            enabled=True,
            rules=(
                _rule(name="global-a", command=("global-a", "{path}")),
                _rule(name="global-b", command=("global-b", "{path}")),
            ),
        ),
    )

    assert runner.lint_for_path("module.py") is None
    assert len(sandbox.calls) == 1
    assert sandbox.calls[0]["command"][0] == "ruff"


def test_malformed_source_pyproject_falls_back_to_global_policy(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path)
    (workspace.source_workspace / "pyproject.toml").write_text(
        "[tool.ruff.lint\n",
        encoding="utf-8",
    )
    sandbox = FakeSandbox(
        [SandboxResult(returncode=0, output="", timed_out=False)]
    )
    runner = LintRunner(
        workspace,
        sandbox,
        LintContextConfig(
            enabled=True,
            rules=(
                _rule(name="global-policy", command=("global-lint", "{path}")),
            ),
        ),
    )

    assert runner.lint_for_path("module.py") is None
    assert sandbox.calls[0]["command"][0] == "global-lint"


def test_nearest_nested_source_pyproject_controls_only_its_project(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path)
    nested_source = workspace.source_workspace / "clients" / "alpha"
    nested_source.mkdir(parents=True)
    (nested_source / "pyproject.toml").write_text(
        "[tool.ruff.lint]\nselect = ['F']\n",
        encoding="utf-8",
    )
    nested_workspace = workspace.workspace / "clients" / "alpha" / "src"
    nested_workspace.mkdir(parents=True)
    sandbox = FakeSandbox(
        [SandboxResult(returncode=0, output="", timed_out=False)]
    )
    runner = LintRunner(
        workspace,
        sandbox,
        LintContextConfig(
            enabled=True,
            rules=(
                _rule(name="global-policy", command=("global-lint", "{path}")),
            ),
        ),
    )

    assert runner.lint_for_path("clients/alpha/src/module.py") is None

    assert sandbox.calls == [
        {
            "command": (
                "ruff",
                "check",
                "--force-exclude",
                "--config",
                str(nested_source / "pyproject.toml"),
                str(workspace.workspace / "clients/alpha/src/module.py"),
            ),
            "cwd": workspace.workspace / "clients" / "alpha",
            "timeout": 30,
            "network": False,
        }
    ]
