from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from citra.agent import AgentSession
from citra.agent import runner as runner_module
from citra.agent.runner import AgentRunner
from citra.cli import repl as repl_module
from citra.cli.repl import select_startup_mode, select_startup_workflow
from citra.modes import ModeRegistry, SandboxConfig, TaskSteeringConfig, UserMode
from citra.config import SandboxPolicy
from citra.utils.prompt import build_system_prompt
from citra.sandbox import SandboxMode, WorkspaceSandbox
from citra.workflows import WorkflowRegistry


class _Input:
    def __init__(self, *responses: str) -> None:
        self._responses = iter(responses)

    def prompt(self, _message: str) -> str:
        return next(self._responses)


def _mode(
    name: str,
    *,
    sandbox: SandboxConfig | None = None,
    steering: TaskSteeringConfig | None = None,
) -> UserMode:
    return UserMode(
        name=name,
        description=f"{name} mode",
        system_prompt=f"prompt:{name}",
        sandbox_config=sandbox,
        task_steering=steering,
    )


def test_registry_reads_default_and_accepts_empty_number_or_name(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "mode.toml").write_text(
        'default = "second"\n',
        encoding="utf-8",
    )
    registry = ModeRegistry(
        config_path=config,
        modes=(_mode("first"), _mode("second")),
    )

    assert registry.select().name == "second"
    assert registry.select("1").name == "first"
    assert registry.select("second").name == "second"


def test_startup_selector_uses_configured_default_on_enter(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "mode.toml").write_text(
        'default = "second"\n',
        encoding="utf-8",
    )
    registry = ModeRegistry(
        config_path=config,
        modes=(_mode("first"), _mode("second")),
    )

    selected = select_startup_mode(
        registry,
        input_service=_Input(""),
    )

    assert selected.name == "second"


def test_startup_workflow_selector_defaults_to_simple() -> None:
    registry = WorkflowRegistry(
        mode_registry=ModeRegistry(
            modes=(_mode("first"),),
            default_mode="first",
        )
    )

    selected = select_startup_workflow(
        registry,
        input_service=_Input(""),
    )

    assert selected.name == "simple"


def test_repl_selects_mode_before_application_runtime_is_created(
    monkeypatch,
) -> None:
    registry = ModeRegistry(
        modes=(_mode("first"), _mode("second")),
        default_mode="second",
    )
    events: list[str] = []

    class _StartupInput:
        calls = 0

        def prompt(self, _message: str) -> str:
            self.calls += 1
            if self.calls == 1:
                events.append("workflow selected")
                return ""
            if self.calls == 2:
                events.append("mode selected")
                return ""
            raise EOFError

    class _Application:
        config = object()
        source_workspace = Path(".")
        workspace = SimpleNamespace(workspace=Path("."))
        hard_shutdown_requested = False

        def close(self, *, force: bool = False) -> None:
            del force

    def create_application(**kwargs) -> _Application:
        assert kwargs["workflow"].name == "simple"
        assert kwargs["workflow"].initial_mode.name == "second"
        events.append("runtime created")
        return _Application()

    monkeypatch.setattr(
        repl_module,
        "ModeRegistry",
        lambda **_kwargs: registry,
    )
    monkeypatch.setattr(
        repl_module,
        "WorkflowRegistry",
        lambda **_kwargs: WorkflowRegistry(mode_registry=registry),
    )
    monkeypatch.setattr(
        repl_module.CitraApplication,
        "create",
        staticmethod(create_application),
    )
    monkeypatch.setattr(repl_module, "print_header", lambda *_args: None)

    repl_module.main(
        input_service=_StartupInput(),
        interactive_mode_selection=True,
    )

    assert events == [
        "workflow selected",
        "mode selected",
        "runtime created",
    ]


def test_mode_sandbox_policy_is_authoritative_and_operator_policy_adds(
    tmp_path: Path,
) -> None:
    mode_ro = tmp_path / "mode-ro"
    mode_rw = tmp_path / "mode-rw"
    operator_ro = tmp_path / "operator-ro"
    operator_rw = tmp_path / "operator-rw"
    for path in (mode_ro, mode_rw, operator_ro, operator_rw):
        path.mkdir()
    policy = SandboxPolicy(
        extra_ro_binds=[operator_ro],
        extra_w_binds=[operator_rw],
    )
    policy.apply_mode_config(
        SandboxConfig(
            mode=SandboxMode.FULL_SANDBOX,
            additional_ro_binds=(mode_ro,),
            additional_w_binds=(mode_rw,),
            global_network_disallow=True,
        )
    )
    sandbox = WorkspaceSandbox(tmp_path, policy, base_environment={})

    assert sandbox.mode is SandboxMode.FULL_SANDBOX
    assert sandbox.readonly_binds() == (
        mode_ro,
        operator_ro,
    )
    assert sandbox.writable_binds() == (
        tmp_path,
        mode_rw,
        operator_rw,
    )
    assert sandbox.allows_network(True) is False


def test_operator_can_further_restrict_mode_network() -> None:
    policy = SandboxPolicy(global_disallow_network=True)
    policy.apply_mode_config(SandboxConfig(global_network_disallow=False))
    sandbox = WorkspaceSandbox(Path.cwd(), policy, base_environment={})

    assert sandbox.allows_network(True) is False
    assert sandbox.allows_network(False) is False


def test_system_prompt_and_task_steering_are_owned_by_mode() -> None:
    mode = _mode(
        "custom",
        steering=TaskSteeringConfig(
            every_n_turns=2,
            content="re-check constraints",
            include_first=True,
        ),
    )
    context = SimpleNamespace(mode=mode)

    assert build_system_prompt(context) == "prompt:custom"
    assert mode.get_task_steering(0, context) == "re-check constraints"
    assert mode.get_task_steering(1, context) is None
    assert mode.get_task_steering(2, context) == "re-check constraints"


def test_runner_injects_mode_steering_before_first_request(monkeypatch) -> None:
    mode = _mode(
        "custom",
        steering=TaskSteeringConfig(
            every_n_turns=3,
            content="mode steering",
            include_first=True,
        ),
    )
    context = SimpleNamespace(
        mode=mode,
        workspace=SimpleNamespace(
            is_closing=False,
            disabled_tool_ids=(),
        ),
        config=SimpleNamespace(
            model=lambda: SimpleNamespace(
                id="test-model",
                max_input_tokens=10_000,
                reasoning_effort=None,
            ),
        ),
    )
    session = AgentSession(memory_enabled=False)
    session.add_user_message("original request")
    requests: list[dict] = []

    class _Registry:
        def __init__(self, **_kwargs) -> None:
            pass

        core_tool_ids: tuple[str, ...] = ()

        @staticmethod
        def deferred_catalog(_context) -> dict[str, str]:
            return {}

        @staticmethod
        def instantiate(*_args, **_kwargs) -> dict:
            return {}

        @staticmethod
        def index_by_model_name(_tools) -> dict:
            return {}

    monkeypatch.setattr(runner_module, "ToolRegistry", _Registry)

    def api_call(**kwargs) -> dict:
        requests.append(kwargs)
        return {
            "choices": [
                {"message": {"role": "assistant", "content": None}}
            ]
        }

    AgentRunner(context, session, api_call=api_call).run_turn()

    assert requests[0]["sys_prompt"] == "prompt:custom"
    assert requests[0]["messages"][-1] == {
        "role": "user",
        "content": "mode steering",
    }


def test_builtin_modes_use_the_sandboxed_project_view() -> None:
    from citra.modes import ChatMode, SimpleTask

    assert ChatMode().sandbox_config.mode is SandboxMode.FULL_SANDBOX
    assert SimpleTask().sandbox_config.mode is SandboxMode.FULL_SANDBOX
