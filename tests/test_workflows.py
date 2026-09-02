from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from citra import application as application_module
from citra.agent import AgentSession
from citra.application import CitraApplication
from citra.modes import SandboxConfig, UserMode
from citra.sandbox import SandboxMode
from citra.tools.default_registry import ToolSet, memory_tools
from citra.tools.session_memory import CheckpointTool, RequirementTool
from citra.tools.subagent.supervisor import _paths_overlap
from citra.tools.tool_registry import ToolRegistry
from citra.workflows import (
    SerialRolesWorkflow,
    SingleModeWorkflow,
    WorkflowRegistry,
    WorkflowRuntime,
    simple_workflow,
)
from citra.workflows.workflow import WorkflowRun, WorkflowStep


def _mode(name: str, sandbox_mode: SandboxMode) -> UserMode:
    return UserMode(
        name=name,
        system_prompt=f"prompt:{name}",
        sandbox_config=SandboxConfig(mode=sandbox_mode),
    )


def test_workflow_policy_optionally_overrides_mode_policy() -> None:
    mode = _mode("mode", SandboxMode.PARTIAL_SANDBOX)
    inherited = simple_workflow(mode)
    override = SingleModeWorkflow(
        name="override",
        description="test",
        mode=mode,
        sandbox_config=SandboxConfig(mode=SandboxMode.FULL_SANDBOX),
    )

    assert inherited.sandbox_config is None
    assert inherited.resolved_sandbox_config is mode.sandbox_config
    assert override.resolved_sandbox_config.mode is SandboxMode.FULL_SANDBOX


def test_workflow_runtime_owns_the_concrete_sandbox() -> None:
    workflow = SerialRolesWorkflow()
    sandbox = SimpleNamespace(mode=SandboxMode.FULL_SANDBOX)

    runtime = WorkflowRuntime(
        workflow=workflow,
        workspace=SimpleNamespace(),
        sandbox=sandbox,
    )

    assert runtime.sandbox is sandbox
    assert runtime.workflow is workflow
    assert runtime.sandbox_config is workflow.sandbox_config

    first = runtime.start_run("first task")
    with pytest.raises(RuntimeError, match="already active"):
        runtime.start_run("second task")
    assert runtime.cancel_run()
    assert first.snapshot().cancelled
    assert runtime.start_run("second task").task == "second task"


def test_workflow_registry_defaults_to_simple(tmp_path: Path) -> None:
    registry = WorkflowRegistry(config_path=tmp_path)

    assert tuple(item.name for item in registry.workflows) == (
        "simple",
        "serial_roles",
        "architect",
    )
    assert registry.select().name == "simple"


def test_workflow_registry_reads_configured_default(tmp_path: Path) -> None:
    (tmp_path / "workflow.toml").write_text(
        'default = "serial_roles"\n',
        encoding="utf-8",
    )

    registry = WorkflowRegistry(config_path=tmp_path)

    assert registry.select().name == "serial_roles"


def test_serial_run_validates_and_applies_loop_transitions() -> None:
    run = SerialRolesWorkflow().create_run("Implement the feature")

    assert run.begin_step().step_id == "explore"
    run.submit_handoff(summary="Need another evidence pass.", next_step="explore")
    run.advance()
    assert run.current_step.step_id == "explore"

    run.begin_step()
    run.submit_handoff(summary="Relevant paths and constraints.", next_step="plan")
    run.advance()
    assert run.current_step.step_id == "plan"
    assert "Relevant paths and constraints." in run.phase_input()

    with pytest.raises(ValueError, match="cannot transition"):
        run.submit_handoff(summary="Invalid jump.", next_step="complete")


def test_serial_roles_use_standard_memory_tools_for_handoff() -> None:
    workflow = SerialRolesWorkflow()
    run = workflow.create_run("Implement the feature")
    expected_memory_ids = {tool.TOOL_ID for tool in memory_tools()}

    for phase, next_step in (
        ("explore", "plan"),
        ("plan", "implement"),
        ("implement", "test"),
        ("test", "review"),
        ("review", "complete"),
    ):
        assert run.current_step.step_id == phase
        tool_ids = run.current_step.mode.tool_set.core_tool_ids
        assert expected_memory_ids <= tool_ids
        assert "workflow_handoff" not in tool_ids
        run.submit_handoff(
            summary=f"Assistant message from {phase}",
            next_step=next_step,
        )
        run.advance()


def test_serial_role_sessions_reuse_memory_but_not_conversation() -> None:
    workflow = SerialRolesWorkflow()
    run = workflow.create_run("Implement the feature")
    context = SimpleNamespace(
        config=SimpleNamespace(
            memory=SimpleNamespace(enabled=False),
            model=lambda: SimpleNamespace(id="test-model"),
        ),
        workflow=workflow,
    )
    registry = ToolRegistry(
        ToolSet(core_tools=(CheckpointTool,), deferred_tools=())
    )
    first_session = AgentSession(memory=run.memory, memory_enabled=True)
    first = registry.instantiate(context, first_session)[CheckpointTool.TOOL_ID]
    first.execute(
        {
            "action": "set",
            "content": "Exploration complete",
            "next_step": "plan",
        }
    )
    first_session.add_user_message("private exploration history")

    second_session = AgentSession(memory=run.memory, memory_enabled=True)
    second = registry.instantiate(context, second_session)[CheckpointTool.TOOL_ID]

    assert second is first
    assert second.session is second_session
    current = second.current_checkpoint
    assert current is not None
    assert current.content == "Exploration complete"
    assert second_session.get_messages() == []


def test_requirement_memory_tracks_verified_acceptance() -> None:
    context = SimpleNamespace(
        config=SimpleNamespace(
            model=lambda: SimpleNamespace(id="test-model"),
        )
    )
    session = AgentSession()
    requirements = RequirementTool(context, session)

    requirements.execute(
        {
            "action": "add",
            "contents": ["Preserve API", "Pass contract tests"],
        }
    )
    assert requirements.has_unsatisfied_requirements()
    requirements.execute(
        {
            "action": "satisfy",
            "id": 1,
            "evidence": "Compatibility test passed",
        }
    )
    first = requirements.get_extracts()[0]
    assert first.satisfied
    assert first.evidence == "Compatibility test passed"

    requirements.execute({"action": "satisfy", "id": 2})
    assert not requirements.has_unsatisfied_requirements()
    requirements.execute({"action": "reopen", "id": 1})
    assert requirements.has_unsatisfied_requirements()


def test_serial_loop_has_a_controller_execution_bound() -> None:
    workflow = SerialRolesWorkflow()
    run = WorkflowRun(
        workflow=workflow,
        task="bounded loop",
        steps=(
            WorkflowStep(
                "explore",
                workflow.initial_mode,
                ("explore",),
            ),
        ),
        max_executions=1,
    )

    run.begin_step()
    run.submit_handoff(summary="repeat", next_step="explore")
    run.advance()

    with pytest.raises(RuntimeError, match="maximum serial step executions"):
        run.begin_step()
    assert run.snapshot().cancelled


def test_application_uses_fresh_session_per_serial_role(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workflow = SerialRolesWorkflow()
    activated: list[tuple[str, object, object]] = []
    executed: list[tuple[str, object]] = []
    role_inputs: list[str] = []
    shared_sandbox = object()

    class FakeCheckpoint:
        TOOL_ID = "checkpoint"

        def __init__(self) -> None:
            self.revision = 0
            self.current_checkpoint = None

    class FakeContext:
        mode = workflow.initial_mode
        workflow_run = None
        sandbox = shared_sandbox

        def activate_mode(self, mode, *, skills, workflow_run) -> None:
            del skills
            assert self.sandbox is shared_sandbox
            self.mode = mode
            self.workflow_run = workflow_run
            activated.append((mode.name, application.session, self.sandbox))

    transitions = {
        "explore": "plan",
        "plan": "implement",
        "implement": "test",
        "test": "review",
        "review": "complete",
    }

    class FakeRunner:
        def __init__(self, context, session, *, api_call) -> None:
            del api_call
            self.context = context
            self.session = session

        def run_turn(self) -> None:
            phase = self.context.workflow_run.current_step.step_id
            executed.append((phase, self.session))
            role_inputs.append(str(self.session.get_messages()[0]["content"]))
            checkpoint = self.session.memory.get_or_create(
                FakeCheckpoint.TOOL_ID,
                FakeCheckpoint,
            )
            checkpoint.revision += 1
            checkpoint.current_checkpoint = SimpleNamespace(
                next_step=transitions[phase],
            )
            self.session.add_assistant_message(
                {
                    "role": "assistant",
                    "content": f"Assistant message from {phase}",
                }
            )

    class FakeWorkflowRuntime:
        active_run = None

        def start_run(self, task: str):
            self.active_run = workflow.create_run(task)
            return self.active_run

    monkeypatch.setattr(application_module, "AgentRunner", FakeRunner)
    monkeypatch.setattr(application_module, "CheckpointTool", FakeCheckpoint)
    monkeypatch.setattr(
        application_module,
        "SkillRegistry",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    application = CitraApplication.__new__(CitraApplication)
    application.config = SimpleNamespace(memory=SimpleNamespace(enabled=True))
    application.workflow = workflow
    application.workspace = SimpleNamespace(ensure_active=lambda: None)
    application.context = FakeContext()
    application._api_call = object()
    application.workflow_runtime = FakeWorkflowRuntime()
    application._skills_root = lambda: tmp_path

    application.prepare_user_turn("Implement the feature")
    application.run_agent_turn()

    assert [phase for phase, _ in executed] == [
        "explore",
        "plan",
        "implement",
        "test",
        "review",
    ]
    assert len({id(session) for _, session in executed}) == 5
    assert len({id(session.memory) for _, session in executed}) == 1
    assert len({id(session) for _, session, _ in activated}) == 5
    assert {id(sandbox) for _, _, sandbox in activated} == {
        id(shared_sandbox)
    }
    assert "Assistant message from explore" in role_inputs[1]
    assert "Assistant message from explore" not in role_inputs[2]
    assert "Assistant message from plan" in role_inputs[2]
    assert application.workflow_run.snapshot().completed


def test_subagent_ownership_paths_cannot_overlap(tmp_path: Path) -> None:
    component = tmp_path / "component"
    nested = component / "nested"
    sibling = tmp_path / "sibling"

    assert _paths_overlap(component, nested)
    assert _paths_overlap(nested, component)
    assert not _paths_overlap(component, sibling)
