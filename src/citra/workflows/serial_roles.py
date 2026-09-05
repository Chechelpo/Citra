"""General serial-role workflows with durable, typed handoff state."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from citra.logging import Logger
from citra.sandbox import SandboxMode
from citra.tools.capabilities import ToolCapabilities
from citra.tools.default_registry import ToolConfiguration, ToolSet
from citra.tools.session_memory import (
    AcceptanceCriteriaTool,
    ChangeTool,
    CheckpointTool,
    ConstraintTool,
    DecisionTool,
    FactTool,
    IssueTool,
    RequirementTool,
    ScopeTool,
    TodoTool,
    VerificationTool,
    WorkingStateTool,
)
from citra.tools.skills.skill import Skill
from citra.tools.subagent.tool import SubagentTool
from citra.tools.transient import (
    Bash,
    Browser,
    Edit,
    Glob,
    Lsp,
    PromptUser,
    Read,
    SkillTool,
    Subprocess,
    Tree,
    Workspace,
    Write,
)
from citra.utils.prompt import EnvironmentInfo, collect_environment, format_skills, basic_coding_conventions

from .workflow import (
    SandboxConfig,
    SingleModeWorkflow,
    TaskSteeringConfig,
    Workflow,
    WorkflowRun,
    WorkflowStep,
)

if TYPE_CHECKING:
    from citra.context import ExecutionContext
    from citra.tools.tool import Tool


_logger = Logger(__name__)
_SERIAL_SANDBOX = SandboxConfig(mode=SandboxMode.PARTIAL_SANDBOX)


def _restricted(
    tool_type: type[Tool],
    *actions: str,
) -> ToolConfiguration:
    """Create one action-restricted tool entry for a role boundary."""
    return ToolConfiguration(
        type=tool_type,
        capabilities=ToolCapabilities(include=tuple(actions)),
    )


_MEMORY_PROTOCOL = """
# Shared workflow memory protocol

Structured memory is the durable handoff between isolated roles. Use it as the
single source of truth for workflow state.

Never copy conversation history into memory. Store only information another
isolated role needs to continue correctly.

Memory ownership:

- `requirement` (R):
  User-visible behavior or need that must be satisfied.
  Explorer establishes the wording. Reviewer decides satisfaction.

- `acceptance_criteria` (A):
  Observable conditions proving requirements are fulfilled.
  Explorer defines them. Reviewer decides satisfaction.

- `scope`:
  Explicit boundaries of the request.
  Explorer establishes included and excluded work.

- `constraint`:
  Mandatory restrictions, invariants, or limitations.
  Explorer establishes them. All roles must respect them.

- `fact`:
  Verified evidence from the repository or environment.
  Explorer discovers them. Other roles may add only newly verified facts.

- `todo`:
  Executable implementation plan.
  Planner owns creation. Implementer owns completion.

- `decision`:
  A consequential choice that affects future work.
  Planner owns architectural/design decisions. Do not record ordinary coding
  choices.

- `change` (CH):
  Actual implementation changes with affected paths and behavior.
  Implementer owns creation and updates.

- `verification` (V):
  Reproducible evidence from testing or inspection.
  Tester owns creation and invalidation.

- `issue` (I):
  A routed problem requiring correction:
  defect, requirement gap, plan gap, or verification gap.
  Create only when action is required.

- `working_state`:
  Temporary reasoning, hypotheses, and intermediate exploration.
  Resolve, promote, or discard before handoff.

- `checkpoint`:
  Controller routing state only.

Always preserve relationships:
R -> A -> TODO -> CH -> V

If evidence invalidates previous state:
- update the owned memory if your role has authority;
- otherwise create an issue and route to the responsible phase.

Do not create unnecessary records. Small tasks should remain small.
""".strip()


class _RoleWorkflow(SingleModeWorkflow):
    """Provide one stateless role in a controller-managed serial workflow."""

    ROLE: ClassVar[str]
    DESCRIPTION: ClassVar[str]
    INSTRUCTIONS: ClassVar[str]
    TOOLS: ClassVar[ToolSet]
    TASK_STEERING: ClassVar[TaskSteeringConfig]
    WORKFLOW_PREFIX: ClassVar[str] = "serial"
    ASSURANCE_INSTRUCTIONS: ClassVar[str] = ""
    _AVAILABLE_SKILLS: ClassVar[tuple[Skill, ...]] = ()

    @property
    def name(self) -> str:
        """Return the stable role workflow name."""
        return f"{self.WORKFLOW_PREFIX}:{self.ROLE}"

    @property
    def description(self) -> str:
        """Return the role's selection description."""
        return self.DESCRIPTION

    @property
    def tool_set(self) -> ToolSet:
        """Return tools and action capabilities owned by this role."""
        return self.TOOLS

    @property
    def skills(self) -> tuple[Skill, ...]:
        """Return skills available to this isolated role."""
        return self._AVAILABLE_SKILLS

    @property
    def sandbox_config(self) -> SandboxConfig:
        """Return the sandbox policy shared by every serial phase."""
        return _SERIAL_SANDBOX

    @property
    def task_steering(self) -> TaskSteeringConfig:
        """Return periodic role-specific self-review guidance."""
        return self.TASK_STEERING

    @property
    def initial_working_states(self) -> tuple[str, ...]:
        """Return no speculative state for a fresh isolated role."""
        return ()

    @override
    def get_system_prompt(self, context: ExecutionContext) -> str:
        """Build the role prompt with route, environment, and memory contracts."""
        run = context.workflow_runtime.require_active_run()
        step = run.current_step
        checkpoint_name = CheckpointTool.resolve_definition_for_context(
            context
        ).function.name
        allowed = ", ".join(f"`{item}`" for item in step.allowed_next)
        environment: EnvironmentInfo = collect_environment(context)
        assurance = (
            f"\n\n# Additional assurance discipline\n\n{self.ASSURANCE_INSTRUCTIONS}"
            if self.ASSURANCE_INSTRUCTIONS
            else ""
        )
        _logger.debug(
            "Built serial role prompt",
            workflow=self.WORKFLOW_PREFIX,
            role=self.ROLE,
            allowed_next=step.allowed_next,
        )

        return f"""
# Serial workflow role: {self.ROLE}

Work only on this phase. You share the sandbox, filesystem, and structured
memory with other roles, but this conversation is fresh. Treat the previous
role's message and retained memory as evidence to verify, not hidden reasoning.

{self.INSTRUCTIONS}

{_MEMORY_PROTOCOL}{assurance}

# Environment

{environment.as_prompt_section()}

Use the available inspection, editing, execution, and documentation tools only
when they materially help this phase. Avoid environment-specific coupling when
a reasonable portable implementation is available.

# Available skills

{format_skills(self._AVAILABLE_SKILLS)}

Call a skill only when relevant.

# Coding

{basic_coding_conventions()}

# Required handoff

Before ending this phase:

1. Reconcile every memory record your role owns with the current filesystem.
2. Ensure provisional working states are resolved or discarded.
3. Call `{checkpoint_name}` with `action="set"`, a compact delta-oriented
   `content`, and `next_step` equal to one of {allowed}.
4. Return a self-contained final message for the next role: state the outcome,
   material evidence, blockers, and immediate next action. Refer to durable
   record IDs instead of repeating all stored content.

The controller validates the checkpoint and starts a new isolated role. Do not
simulate that role in this turn.
""".strip()


class ExplorerWorkflow(_RoleWorkflow):
    """Ground the request, boundaries, and observable success conditions."""

    ROLE = "explore"
    DESCRIPTION = "Clarify the task and inspect relevant repository evidence."
    INSTRUCTIONS = INSTRUCTIONS = """
Transform the user's request into a grounded problem definition.

You are the only role allowed to ask the user questions. Ask only when a
material ambiguity cannot be resolved through repository evidence.

First understand:
- user goal;
- expected outcome;
- requested behavior;
- constraints;
- boundaries.

Record:
- requirements as user-visible needs;
- acceptance criteria as observable success conditions;
- scope as included and excluded work;
- constraints as mandatory limitations;
- facts as verified repository evidence.

Then inspect:
- source code;
- tests;
- configuration;
- dependencies;
- entry points;
- current behavior.

Do not design the solution. Do not choose components or implementation
approaches.

Your output should allow a planner to answer:
"What problem are we solving, what must remain true, and how will we know it is
complete?"

Advance to plan only when:
- requirements are clear;
- scope is bounded;
- acceptance criteria exist for meaningful behavior;
- relevant facts and constraints are recorded.
""".strip()
    TASK_STEERING = TaskSteeringConfig(
        include_first=False,
        every_n_turns=4,
        content="""
        Re-evaluate whether the request is ready for planning.

        Check:
            1. Requirements describe user needs, not implementation ideas.
            2. Scope explicitly prevents accidental expansion.
            3. Acceptance criteria describe observable outcomes.
            4. Constraints are separated from facts.
            5. Facts are supported by repository evidence.
            6. Unknowns that affect correctness are resolved or routed.

        Do not design a solution. If the problem definition is complete, hand off to
        plan.
        """
    )
    TOOLS = ToolSet(
        core_tools=(
            PromptUser,
            SkillTool,
            Read,
            Glob,
            Tree,
            _restricted(RequirementTool, "add", "update", "remove"),
            _restricted(AcceptanceCriteriaTool, "add", "update", "remove"),
            ScopeTool,
            ConstraintTool,
            FactTool,
            _restricted(IssueTool, "add", "update", "resolve", "remove"),
            WorkingStateTool,
            _restricted(CheckpointTool, "set"),
        ),
        deferred_tools=(Lsp,),
    )


class PlannerWorkflow(_RoleWorkflow):
    """Produce the smallest executable plan supported by repository evidence."""

    ROLE = "plan"
    DESCRIPTION = "Create a concise executable implementation plan."
    INSTRUCTIONS = """
Transform the explored problem into the smallest coherent solution.

Validate the exploration state against repository evidence before planning.

Determine the appropriate solution level:
- local modification;
- refactoring existing code;
- changing module boundaries;
- introducing or modifying components;
- changing interfaces;
- introducing architectural decisions.

Do not create architecture for its own sake. Planning effort must match task
complexity.

Create TODOs that describe:
- what changes;
- where it changes;
- why it changes;
- which requirements/acceptance criteria it covers;
- how it will be verified.

Record decisions only when a choice materially affects:
- structure;
- compatibility;
- quality attributes;
- future evolution.

Consider alternatives only when the trade-off affects correctness, risk, cost,
or maintainability.

Do not edit files.
""".strip()
    TASK_STEERING = TaskSteeringConfig(
        include_first=False,
        every_n_turns=4,
        content="""
        Check that the plan is executable.

        Verify:
        1. Every requirement has implementation coverage.
        2. Every acceptance criterion has a verification path.
        3. TODOs are concrete and ordered.
        4. Components or boundaries are introduced only when justified.
        5. Decisions exist only for consequential choices.
        6. The implementer can execute without rediscovering intent.

        If the approach is wrong, return to exploration or revise the plan.
        """
    )
    TOOLS = ToolSet(
        core_tools=(
            SkillTool,
            Read,
            Glob,
            Tree,
            _restricted(TodoTool, "add", "insert", "promote", "remove"),
            DecisionTool,
            FactTool,
            _restricted(IssueTool, "add", "update", "resolve", "remove"),
            WorkingStateTool,
            _restricted(CheckpointTool, "set"),
        ),
        deferred_tools=(Lsp,),
    )


class ImplementerWorkflow(_RoleWorkflow):
    """Implement planned work and record the actual change set."""

    ROLE = "implement"
    DESCRIPTION = "Implement the approved TODOs and record changed behavior."
    INSTRUCTIONS = """
Execute the approved TODOs.

Before editing:
- inspect affected files;
- confirm assumptions;
- preserve unrelated changes.

During implementation:
- make the smallest coherent change;
- maintain repository conventions;
- update change records with actual paths and behavior;
- record important discoveries.

Do not redesign the solution. If the plan is invalid, route back to plan.

Complete TODOs only when the described outcome exists and can be verified.

""".strip()
    TASK_STEERING = TaskSteeringConfig(
        include_first=False,
        every_n_turns=5,
        content=(
            "Reconcile files with TODOs and CH records, preserve scope and "
            "constraints, classify discoveries instead of silently redesigning, "
            "and route to independent test when implementation is coherent."
        ),
    )
    TOOLS = ToolSet(
        core_tools=(
            SkillTool,
            Read,
            Glob,
            Tree,
            Edit,
            Write,
            Bash,
            Workspace,
            Lsp,
            _restricted(TodoTool, "check"),
            ChangeTool,
            FactTool,
            _restricted(IssueTool, "add", "update", "resolve", "reopen"),
            WorkingStateTool,
            _restricted(CheckpointTool, "set"),
        ),
        deferred_tools=(SubagentTool,),
    )


class TesterWorkflow(_RoleWorkflow):
    """Collect independent executable evidence without repairing defects."""

    ROLE = "test"
    DESCRIPTION = "Verify the implementation and report reproducible failures."
    INSTRUCTIONS = """
Produce independent evidence that the requested behavior works.

Use:
- requirements;
- acceptance criteria;
- change records;
- existing tests;
- regression risks.

Record verification commands and outcomes.

Do not modify files.
Do not mark acceptance criteria satisfied.

Failures become issues routed to the phase capable of correction.
""".strip()
    TASK_STEERING = TaskSteeringConfig(
        include_first=False,
        every_n_turns=3,
        content=(
            "Check that V records cover requested behavior and regressions, "
            "commands and outcomes are exact, failures have routed I records, "
            "and no project file was edited during verification."
        ),
    )
    TOOLS = ToolSet(
        core_tools=(
            SkillTool,
            Read,
            Glob,
            Tree,
            Bash,
            Lsp,
            VerificationTool,
            _restricted(IssueTool, "add", "update", "resolve", "reopen"),
            WorkingStateTool,
            _restricted(CheckpointTool, "set"),
        ),
        deferred_tools=(Browser, Subprocess),
    )


class ReviewerWorkflow(_RoleWorkflow):
    """Adjudicate task completion from repository state and current evidence."""

    ROLE = "review"
    DESCRIPTION = "Review scope, correctness, evidence, and task satisfaction."
    INSTRUCTIONS = """
Review the repository state independently: diff, behavior, error handling,
compatibility, tests, scope, constraints, TODOs, CH records, V evidence, and
open issues. Do not trust a handoff claim without checking its evidence and do
not repair defects in review.

You are the only role that marks requirements and acceptance criteria satisfied
or reopens them. Cite active verification or precise inspection evidence for
each satisfaction action. If anything fails, add or reopen a routed issue and
send the workflow to the earliest phase capable of correction.

Complete only when every valid R and A is satisfied, every valid TODO is
checked, no active failed/blocked/stale-change V remains, no blocking I remains
open, and no working state remains provisional.
""".strip()
    TASK_STEERING = TaskSteeringConfig(
        include_first=False,
        every_n_turns=3,
        content=(
            "Reassert independence: inspect actual state, validate V coverage, "
            "adjudicate every R/A with evidence, route each defect to its "
            "earliest owner, and complete only when all gates are clear."
        ),
    )
    TOOLS = ToolSet(
        core_tools=(
            SkillTool,
            Read,
            Glob,
            Tree,
            Bash,
            Lsp,
            _restricted(RequirementTool, "satisfy", "reopen"),
            _restricted(AcceptanceCriteriaTool, "satisfy", "reopen"),
            _restricted(VerificationTool, "record", "invalidate"),
            _restricted(IssueTool, "add", "update", "reopen"),
            WorkingStateTool,
            _restricted(CheckpointTool, "set"),
        ),
        deferred_tools=(),
    )


class AssuredExplorerWorkflow(ExplorerWorkflow):
    """Apply stricter traceability while grounding higher-risk work."""

    WORKFLOW_PREFIX = "serial_assured"
    ASSURANCE_INSTRUCTIONS = (
        "Link each acceptance criterion to its requirement IDs. Capture every "
        "material boundary or risk explicitly; do not manufacture records for "
        "irrelevant categories."
    )


class AssuredPlannerWorkflow(PlannerWorkflow):
    """Apply stricter coverage discipline to a still-proportional plan."""

    WORKFLOW_PREFIX = "serial_assured"
    ASSURANCE_INSTRUCTIONS = (
        "Ensure every requirement and criterion is covered by at least one "
        "TODO, using links on the TODO itself. This is a coverage check, not a "
        "request for extra planning documents or speculative design."
    )


class AssuredImplementerWorkflow(ImplementerWorkflow):
    """Require explicit change records for higher-assurance implementation."""

    WORKFLOW_PREFIX = "serial_assured"
    ASSURANCE_INSTRUCTIONS = (
        "Before test, every checked TODO must be represented by a current CH "
        "record with exact paths and applicable R/A links."
    )


class AssuredTesterWorkflow(TesterWorkflow):
    """Require explicit evidence coverage for higher-assurance verification."""

    WORKFLOW_PREFIX = "serial_assured"
    ASSURANCE_INSTRUCTIONS = (
        "Give every acceptance criterion active V evidence or an open I record "
        "that explains the coverage gap and routes correction."
    )


class AssuredReviewerWorkflow(ReviewerWorkflow):
    """Require end-to-end evidence links before higher-assurance completion."""

    WORKFLOW_PREFIX = "serial_assured"
    ASSURANCE_INSTRUCTIONS = (
        "Trace every satisfaction judgment through current TODO, CH, and V/I "
        "records. The identifiers are the audit trail; do not duplicate them in "
        "a separate report."
    )


def _steps(
    explorer: _RoleWorkflow,
    planner: _RoleWorkflow,
    implementer: _RoleWorkflow,
    tester: _RoleWorkflow,
    reviewer: _RoleWorkflow,
) -> tuple[WorkflowStep, ...]:
    """Build the shared feedback graph for one serial-role variant."""
    return (
        WorkflowStep("explore", explorer, ("explore", "plan")),
        WorkflowStep("plan", planner, ("explore", "plan", "implement")),
        WorkflowStep(
            "implement",
            implementer,
            ("explore", "plan", "implement", "test"),
        ),
        WorkflowStep(
            "test",
            tester,
            ("explore", "plan", "implement", "test", "review"),
        ),
        WorkflowStep(
            "review",
            reviewer,
            ("explore", "plan", "implement", "test", "review", "complete"),
        ),
    )


class _SerialRolesWorkflowBase(Workflow):
    """Share lifecycle behavior between serial-role workflow variants."""

    _NAME: ClassVar[str]
    _DESCRIPTION: ClassVar[str]
    _STEPS: ClassVar[tuple[WorkflowStep, ...]]
    _MAX_EXECUTIONS: ClassVar[int] = 32

    def __init__(self) -> None:
        """Validate phase definitions and their shared sandbox policy."""
        self.validate()
        for step in self._STEPS:
            if step.workflow.sandbox_config != self.sandbox_config:
                _logger.error(
                    "Serial phase sandbox mismatch",
                    workflow=self._NAME,
                    step=step.step_id,
                )
                raise ValueError(
                    "Serial workflow phases must share the root workflow's "
                    "sandbox configuration"
                )
        _logger.info(
            "Initialized serial-role workflow",
            workflow=self._NAME,
            steps=tuple(step.step_id for step in self._STEPS),
        )

    @property
    def name(self) -> str:
        """Return the selectable workflow name."""
        return self._NAME

    @property
    def description(self) -> str:
        """Return the selectable workflow description."""
        return self._DESCRIPTION

    @property
    def sandbox_config(self) -> SandboxConfig:
        """Return the sandbox policy shared by all phases."""
        return _SERIAL_SANDBOX

    @property
    def initial_workflow(self) -> SingleModeWorkflow:
        """Return the exploration role that starts each task."""
        return self._STEPS[0].workflow

    @property
    def is_serial(self) -> bool:
        """Declare isolated sequential role execution."""
        return True

    @property
    def requires_memory(self) -> bool:
        """Require structured memory even when global memory is disabled."""
        return True

    def create_run(self, task: str) -> WorkflowRun:
        """Create one bounded run over this workflow's feedback graph."""
        _logger.info("Created serial workflow run", workflow=self._NAME)
        return WorkflowRun(
            workflow=self,
            task=task,
            steps=self._STEPS,
            max_executions=self._MAX_EXECUTIONS,
        )


class SerialRolesWorkflow(_SerialRolesWorkflowBase):
    """Run the balanced general-purpose five-role workflow."""

    _NAME = "serial_roles"
    _DESCRIPTION = (
        "Balanced isolated explore, plan, implement, test, and review roles."
    )
    _STEPS = _steps(
        ExplorerWorkflow(),
        PlannerWorkflow(),
        ImplementerWorkflow(),
        TesterWorkflow(),
        ReviewerWorkflow(),
    )


class AssuredSerialRolesWorkflow(_SerialRolesWorkflowBase):
    """Run the general five-role workflow with stronger evidence traceability."""

    _NAME = "serial_roles_assured"
    _DESCRIPTION = (
        "Higher-assurance serial roles with explicit coverage and evidence."
    )
    _MAX_EXECUTIONS = 48
    _STEPS = _steps(
        AssuredExplorerWorkflow(),
        AssuredPlannerWorkflow(),
        AssuredImplementerWorkflow(),
        AssuredTesterWorkflow(),
        AssuredReviewerWorkflow(),
    )


__all__ = ["AssuredSerialRolesWorkflow", "SerialRolesWorkflow"]
