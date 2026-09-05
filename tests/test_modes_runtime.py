from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from citra.agent import AgentSession
from citra.agent import runner as runner_module
from citra.agent.runner import AgentRunner
from citra.cli import repl as repl_module
from citra.cli.repl import select_startup_workflow
from citra.config import SandboxPolicy
from citra.sandbox import SandboxMode, WorkspaceSandbox
from citra.tools.session_memory import RequirementTool
from citra.utils.prompt import build_system_prompt
from citra.workflows import (
    ChatWorkflow,
    SandboxConfig,
    SingleModeWorkflow,
    TaskSteeringConfig,
    TaskWorkflow,
    UserWorkflow,
    WorkflowRun,
    WorkflowRegistry,
    WorkflowStep,
)


class _Input:
    def __init__(self, *responses: str) -> None:
        self._responses = iter(responses)

    def prompt(self, _message: str) -> str:
        return next(self._responses)


def _workflow(
    name: str,
    *,
    sandbox: SandboxConfig = SandboxConfig(),
    steering: TaskSteeringConfig = TaskSteeringConfig(),
) -> UserWorkflow:
    return UserWorkflow(
        name=name,
        description=f"{name} workflow",
        system_prompt=f"prompt:{name}",
        sandbox_config=sandbox,
        task_steering=steering,
    )


def test_registry_reads_default_and_accepts_empty_number_or_name(
    tmp_path: Path,
) -> None:
    (tmp_path / "workflow.toml").write_text(
        'default = "second"\n',
        encoding="utf-8",
    )
    registry = WorkflowRegistry(
        config_path=tmp_path,
        workflows=(_workflow("first"), _workflow("second")),
    )

    assert registry.select().name == "second"
    assert registry.select("1").name == "first"
    assert registry.select("second").name == "second"


def test_startup_selector_uses_configured_default_on_enter(tmp_path: Path) -> None:
    (tmp_path / "workflow.toml").write_text(
        'default = "second"\n',
        encoding="utf-8",
    )
    registry = WorkflowRegistry(
        config_path=tmp_path,
        workflows=(_workflow("first"), _workflow("second")),
    )

    selected = select_startup_workflow(registry, input_service=_Input(""))

    assert selected.name == "second"


def test_registry_treats_ordinary_modes_as_single_mode_workflows() -> None:
    workflows = WorkflowRegistry().workflows

    assert tuple(item.name for item in workflows) == (
        "chat",
        "task",
        "serial_roles",
        "serial_roles_assured",
        "architect",
    )
    assert isinstance(workflows[0], SingleModeWorkflow)
    assert isinstance(workflows[1], SingleModeWorkflow)


def test_repl_selects_one_workflow_before_runtime_creation(monkeypatch) -> None:
    registry = WorkflowRegistry(
        workflows=(_workflow("first"), _workflow("second")),
        default_workflow="second",
    )
    events: list[str] = []

    class _StartupInput:
        def prompt(self, _message: str) -> str:
            if not events:
                events.append("workflow selected")
                return ""
            raise EOFError

    class _Application:
        config = object()
        workspace = SimpleNamespace(workspace=Path("."))
        hard_shutdown_requested = False

        def close(self, *, force: bool = False) -> None:
            del force

    def create_application(**kwargs) -> _Application:
        assert kwargs["workflow"].name == "second"
        events.append("runtime created")
        return _Application()

    monkeypatch.setattr(repl_module, "WorkflowRegistry", lambda **_kwargs: registry)
    monkeypatch.setattr(
        repl_module.CitraApplication,
        "create",
        staticmethod(create_application),
    )
    monkeypatch.setattr(repl_module, "print_header", lambda *_args: None)

    repl_module.main(
        input_service=_StartupInput(),
        interactive_workflow_selection=True,
    )

    assert events == ["workflow selected", "runtime created"]


def test_workflow_sandbox_policy_is_authoritative_and_additive(
    tmp_path: Path,
) -> None:
    workflow_ro = tmp_path / "workflow-ro"
    workflow_rw = tmp_path / "workflow-rw"
    operator_ro = tmp_path / "operator-ro"
    operator_rw = tmp_path / "operator-rw"
    for path in (workflow_ro, workflow_rw, operator_ro, operator_rw):
        path.mkdir()
    policy = SandboxPolicy(
        extra_ro_binds=[operator_ro],
        extra_w_binds=[operator_rw],
    )
    policy.apply_workflow_config(
        SandboxConfig(
            mode=SandboxMode.FULL_SANDBOX,
            additional_ro_binds=(workflow_ro,),
            additional_w_binds=(workflow_rw,),
            global_network_disallow=True,
        )
    )
    sandbox = WorkspaceSandbox(tmp_path, policy, base_environment={})

    assert sandbox.mode is SandboxMode.FULL_SANDBOX
    assert sandbox.readonly_binds()[:2] == (workflow_ro, operator_ro)
    assert sandbox.writable_binds()[:3] == (tmp_path, workflow_rw, operator_rw)
    assert sandbox.allows_network(True) is False


def test_operator_can_further_restrict_workflow_network() -> None:
    policy = SandboxPolicy(global_disallow_network=True)
    policy.apply_workflow_config(SandboxConfig(global_network_disallow=False))
    sandbox = WorkspaceSandbox(Path.cwd(), policy, base_environment={})

    assert sandbox.allows_network(True) is False
    assert sandbox.allows_network(False) is False


def test_system_prompt_and_task_steering_are_owned_by_workflow() -> None:
    workflow = _workflow(
        "custom",
        steering=TaskSteeringConfig(
            every_n_turns=2,
            content="re-check constraints",
            include_first=True,
        ),
    )
    context = SimpleNamespace(
        workflow=workflow,
        config=SimpleNamespace(memory=SimpleNamespace(enabled=False)),
        workspace=SimpleNamespace(disabled_tool_ids=()),
    )

    assert build_system_prompt(context) == "prompt:custom"
    assert workflow.get_task_steering(0, context) == "re-check constraints"
    assert workflow.get_task_steering(1, context) is None
    assert workflow.get_task_steering(2, context) == "re-check constraints"


def test_runner_injects_workflow_steering_before_first_request(monkeypatch) -> None:
    workflow = _workflow(
        "custom",
        steering=TaskSteeringConfig(
            every_n_turns=3,
            content="workflow steering",
            include_first=True,
        ),
    )
    model = SimpleNamespace(
        id="test-model",
        max_input_tokens=10_000,
        reasoning_effort=None,
    )
    context = SimpleNamespace(
        workflow=workflow,
        workspace=SimpleNamespace(is_closing=False, disabled_tool_ids=()),
        config=SimpleNamespace(
            memory=SimpleNamespace(enabled=False),
            model=lambda: model,
        ),
        ensure_active=lambda: None,
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
    monkeypatch.setattr("citra.agent.session.tokenize", lambda *_args, **_kwargs: 1)

    def api_call(**kwargs) -> dict:
        requests.append(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": None}}]}

    AgentRunner(context, session, api_call=api_call).run_turn()

    assert requests[0]["sys_prompt"] == "prompt:custom"
    assert requests[0]["messages"][-1] == {
        "role": "user",
        "content": "workflow steering",
    }


def test_builtin_request_receives_every_retained_memory_service(monkeypatch) -> None:
    """Keep read-only prior-role memory visible beside active role tools."""
    workflow = _workflow("custom")
    model = SimpleNamespace(
        id="test-model",
        max_input_tokens=10_000,
        reasoning_effort=None,
    )
    context = SimpleNamespace(
        workflow=workflow,
        workspace=SimpleNamespace(is_closing=False, disabled_tool_ids=()),
        config=SimpleNamespace(
            memory=SimpleNamespace(enabled=True),
            model=lambda: model,
        ),
        ensure_active=lambda: None,
    )
    session = AgentSession(memory_enabled=True)
    session.add_user_message("original request")
    retained = object()
    session.memory.get_or_create("prior-role-record", lambda: retained)
    requests: list[dict] = []

    class _Registry:
        """Provide an empty active tool set for request-boundary testing."""

        def __init__(self, **_kwargs) -> None:
            """Accept the production registry constructor shape."""

        core_tool_ids: tuple[str, ...] = ()

        @staticmethod
        def deferred_catalog(_context) -> dict[str, str]:
            """Return no deferred tools."""
            return {}

        @staticmethod
        def instantiate(*_args, **_kwargs) -> dict:
            """Return no active tools."""
            return {}

        @staticmethod
        def index_by_model_name(_tools) -> dict:
            """Return no model-facing tools."""
            return {}

    def built_in_api(**kwargs) -> dict:
        """Capture the request prepared for the built-in API boundary."""
        requests.append(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": None}}]}

    monkeypatch.setattr(runner_module, "ToolRegistry", _Registry)
    monkeypatch.setattr(runner_module, "call_api", built_in_api)
    monkeypatch.setattr("citra.agent.session.tokenize", lambda *_args, **_kwargs: 1)

    AgentRunner(context, session, api_call=built_in_api).run_turn()

    assert requests[0]["memory_services"] == (retained,)


def test_custom_request_prompt_includes_read_only_retained_memory(monkeypatch) -> None:
    """Expose prior-role records even when the current role lacks their tool."""
    workflow = _workflow("custom")
    model = SimpleNamespace(
        id="test-model",
        max_input_tokens=10_000,
        reasoning_effort=None,
    )
    context = SimpleNamespace(
        workflow=workflow,
        workspace=SimpleNamespace(is_closing=False, disabled_tool_ids=()),
        config=SimpleNamespace(
            memory=SimpleNamespace(enabled=True),
            model=lambda: model,
        ),
        ensure_active=lambda: None,
    )
    session = AgentSession(memory_enabled=True)
    session.add_user_message("continue the serial workflow")
    retained = RequirementTool(
        context=context,
        session=session,
    )
    session.memory.get_or_create(RequirementTool.TOOL_ID, lambda: retained)
    retained.execute(
        {"action": "add", "content": "Preserve recorded state"}
    )
    requests: list[dict] = []

    class _Registry:
        """Expose no tools to emulate a read-only downstream role."""

        def __init__(self, **_kwargs) -> None:
            """Accept the production registry constructor shape."""

        core_tool_ids: tuple[str, ...] = ()

        @staticmethod
        def deferred_catalog(_context) -> dict[str, str]:
            """Return no deferred tools."""
            return {}

        @staticmethod
        def instantiate(*_args, **_kwargs) -> dict:
            """Return no active tools."""
            return {}

        @staticmethod
        def index_by_model_name(_tools) -> dict:
            """Return no model-facing tools."""
            return {}

    def custom_api(**kwargs) -> dict:
        """Capture the custom request without accepting memory services."""
        requests.append(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": None}}]}

    monkeypatch.setattr(runner_module, "ToolRegistry", _Registry)
    monkeypatch.setattr("citra.agent.session.tokenize", lambda *_args, **_kwargs: 1)

    AgentRunner(context, session, api_call=custom_api).run_turn()

    assert "memory_services" not in requests[0]
    assert "[R1] Preserve recorded state" in requests[0]["sys_prompt"]


def test_builtin_single_mode_workflows_use_full_sandbox() -> None:
    assert ChatWorkflow().sandbox_config.mode is SandboxMode.FULL_SANDBOX
    assert TaskWorkflow().sandbox_config.mode is SandboxMode.FULL_SANDBOX


def test_workflow_constructors_validate_exact_inputs() -> None:
    with pytest.raises(ValueError, match="name"):
        UserWorkflow(name="", system_prompt="prompt")
    with pytest.raises(TypeError, match="core_tools"):
        UserWorkflow(name="invalid", system_prompt="prompt", core_tools=[])
    with pytest.raises(TypeError, match="additional_ro_binds"):
        SandboxConfig(additional_ro_binds=[])
    with pytest.raises(ValueError, match="default_workflow"):
        WorkflowRegistry(default_workflow="")


def test_composite_steps_must_share_the_root_sandbox_config() -> None:
    root = _workflow(
        "root",
        sandbox=SandboxConfig(mode=SandboxMode.PARTIAL_SANDBOX),
    )
    step_workflow = _workflow(
        "step",
        sandbox=SandboxConfig(mode=SandboxMode.FULL_SANDBOX),
    )

    with pytest.raises(ValueError, match="root workflow"):
        WorkflowRun(
            workflow=root,
            task="test",
            steps=(WorkflowStep("step", step_workflow, ("complete",)),),
        )
