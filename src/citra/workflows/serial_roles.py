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
from citra.utils.prompt import EnvironmentInfo, collect_environment, format_skills

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
# Shared workflow memory

The filesystem and structured memory survive role changes; conversation and
hidden reasoning do not. Use only the smallest applicable record type:

- `requirement` (R): user-visible need. Exploration owns wording; review owns
  satisfaction.
- `acceptance_criteria` (A): observable proof of one or more requirements.
  Link it to R IDs when known; review owns satisfaction.
- `scope`, `constraint`, `fact`: boundaries, mandatory invariants, and verified
  repository evidence. Do not interchange them.
- `todo`: the executable plan. R/A links are useful but must not inflate a
  small plan.
- `change` (CH): what implementation actually changed, including exact paths.
- `verification` (V): reproducible pass/fail/blocked evidence linked to the CH
  revision it exercised. It does not adjudicate acceptance.
- `issue` (I): a risk, defect, requirement gap, plan gap, or test gap routed
  to the earliest phase able to correct it. Resolve it only with evidence.
- `decision`: a consequential choice another role must respect. Ordinary code
  details do not need a decision entry.
- `working_state`: provisional reasoning only; promote useful consequences and
  resolve it, or discard it.
- `checkpoint`: the sole controller route for the current role.

Never copy a transcript into memory. Keep identifiers and evidence stable so a
later isolated role can follow R -> A -> TODO -> CH -> V/I without guessing.
If new evidence invalidates an accepted item, reopen it where your capabilities
permit and route to the earliest responsible phase.
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
    INSTRUCTIONS = """
Understand the user's goal before designing a solution. You are the only role
allowed to ask the user questions; ask only when repository evidence cannot
resolve a material ambiguity.

Record confirmed needs immediately as requirements. Define concise observable
acceptance criteria and link them to the requirements they prove. Record scope,
constraints, and verified facts in their own memory types. Inspect relevant
code, tests, configuration, entry points, dependencies, and current behavior,
but do not modify project files.

Use issues for material risks or gaps, not for ordinary unknowns you can inspect
now. Do not propose a solution design. Advance to plan when a fresh planner can
act without conversation history; repeat explore when important ambiguity
remains.
""".strip()
    TASK_STEERING = TaskSteeringConfig(
        include_first=False,
        every_n_turns=4,
        content=(
            "Recheck the user goal, R/A links, scope boundary, constraints, "
            "verified current behavior, and unresolved gaps. Ask only if a "
            "material ambiguity remains; otherwise hand off to plan."
        ),
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
Validate the explored state against the repository, then choose the smallest
coherent implementation approach. Planning effort must scale with the task: a
localized change may need one TODO; a refactor may need a short ordered tree.

Each TODO should say what changes, where, and how it will be verified. Attach
R/A IDs while creating it when that information already exists; do not create a
separate mapping exercise. Record a decision only when a consequential choice
must survive the handoff. Compare alternatives only when they could materially
change correctness, compatibility, risk, or effort.

Do not edit project files and do not perform system-architecture design. Route
to explore for an unclear goal or boundary, remain in plan for a flawed or
incomplete approach, and advance to implement as soon as the TODOs are safely
executable.
""".strip()
    TASK_STEERING = TaskSteeringConfig(
        include_first=False,
        every_n_turns=4,
        content=(
            "Keep the plan proportional. Verify paths and invariants, ensure "
            "TODO order and checks are executable, record only consequential "
            "decisions, then advance without extra ceremony."
        ),
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
Execute the active TODOs with the smallest coherent change. Inspect files before
editing, preserve unrelated work, follow repository conventions, and run useful
focused checks while implementing. Do not create Git commits.

Check TODOs only after their outcomes are complete. Record each coherent change
with exact paths and relevant TODO/R/A links; update an existing CH record when
the same change evolves. Resolve an issue only after applying its correction
and cite the concrete result.

Route to explore if implementation exposes a goal or scope ambiguity, to plan
if the approach is invalid, repeat implement if code work remains, and advance
to test when the recorded change set is ready for independent verification.
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
Derive checks from requirements, acceptance criteria, recorded changes, open
issues, and likely regressions. Run the strongest relevant focused and broader
verification available. Record meaningful results as V entries with exact
commands and relevant CH IDs when applicable; supersede stale results after a
retest.

For every failure or blockage, create or update an issue with reproduction
evidence, affected R/A/CH IDs, severity, and the earliest correction phase. Do
not modify project files to make checks pass and do not mark requirements or
criteria satisfied.

Route requirement/scope gaps to explore, approach flaws to plan, code defects to
implement, incomplete or flaky verification to test, and advance to review only
when current evidence is sufficient for independent adjudication.
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
