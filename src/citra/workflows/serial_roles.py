from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from citra.sandbox import SandboxMode
from citra.tools.default_registry import ToolSet, memory_tools
from citra.tools.session_memory import CheckpointTool
from citra.tools.subagent import SubagentTool
from citra.tools.transient import *
from citra.tools.transient import SkillTool
from citra.tools.skills.skill import Skill
from citra.utils.prompt import format_skills
from citra.utils.prompt import collect_environment
from citra.utils.prompt import EnvironmentInfo
from citra.tools.transient import Subprocess
from citra.tools.transient import Browser
from citra.tools.session_memory import *

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


_SERIAL_SANDBOX = SandboxConfig(mode=SandboxMode.PARTIAL_SANDBOX)


class _RoleWorkflow(SingleModeWorkflow):
    """A stateless role used for exactly one isolated workflow phase."""

    ROLE: ClassVar[str]
    DESCRIPTION: ClassVar[str]
    INSTRUCTIONS: ClassVar[str]
    TOOLS: ClassVar[ToolSet]
    TASK_STEERING: ClassVar[TaskSteeringConfig]
    _AVAILABLE_SKILLS: ClassVar[tuple[Skill, ...]] = ()

    @property
    def name(self) -> str:
        """Handle name."""
        return f"serial:{self.ROLE}"

    @property
    def description(self) -> str:
        """Handle description."""
        return self.DESCRIPTION

    @property
    def tool_set(self) -> ToolSet:
        """Handle tool set."""
        return self.TOOLS

    @property
    def skills(self) -> tuple[Skill, ...]:
        """Handle skills."""
        return self._AVAILABLE_SKILLS

    @property
    def sandbox_config(self) -> SandboxConfig:
        """Handle sandbox config."""
        return _SERIAL_SANDBOX

    @property
    def task_steering(self) -> TaskSteeringConfig:
        """Handle task steering."""
        return self.TASK_STEERING

    @property
    def initial_working_states(self) -> tuple[str, ...]:
        """Handle initial working states."""
        return ()

    @override
    def get_system_prompt(self, context: ExecutionContext) -> str:
        """Return get system prompt."""
        run = context.workflow_runtime.require_active_run()
        step = run.current_step
        checkpoint_name = CheckpointTool.resolve_definition_for_context(
            context
        ).function.name
        allowed = ", ".join(f"`{item}`" for item in step.allowed_next)
        environment: EnvironmentInfo = collect_environment(context)
        
        return f"""
# Serial workflow role: {self.ROLE}

You are the {self.ROLE} role in a controller-managed software workflow. Work
only on this phase. You share the same sandbox and filesystem with the other
roles, but you have a fresh conversation. Structured memory tools persist
across roles; use them as the primary source of durable workflow state. Treat
the previous role's message and retained memory as evidence to verify, not as
hidden reasoning or unquestionable truth.

{self.INSTRUCTIONS}

# Environment

{environment.as_prompt_section()}

Treat this environment as the current execution target. Avoid unnecessary
environment-specific coupling when a reasonable abstraction can keep the
implementation portable.

You operate inside a permissive sandbox. Use the available development,
inspection, planning, documentation, diagramming, web, and execution tools
when they materially improve the result.

# Available skills

{format_skills(self._AVAILABLE_SKILLS)}

Call them if relevant.

# Memory

Use facts, constraints, requirements, todo and working state appropiately. Do not, for example, register constraints in facts. Use each for its scoped purpose.

# Memory and message handoff

Keep requirements, facts, decisions, constraints, TODOs, working state, and
the checkpoint concise and synchronized with the shared filesystem. Do not
copy a transcript into memory; retain only state that another isolated role
must survive with.

Before ending this phase:

1. Reconcile the durable memory tools with the work performed.
2. Call `{checkpoint_name}` with `action="set"`, a compact current-state
   `content`, and `next_step` equal to one of {allowed}.
3. After the checkpoint call completes, return a self-contained final
   assistant message containing the next steps (avoid parroting information of the rest of the memory system)

The controller validates the checkpoint transition and sends that final
assistant text to a new isolated role as its user-message handoff. Do not
simulate the next role in this turn.
""".strip()


class ExplorerWorkflow(_RoleWorkflow):
    """Represent ExplorerWorkflow."""
    ROLE = "explore"
    DESCRIPTION = "Inspect the repository and establish grounded constraints."
    
    INSTRUCTIONS = """
Your responsibility is to transform the user's request into a grounded,
implementation-ready problem definition.

You are the only workflow role allowed to communicate directly with the user.
Use that ability to remove ambiguity before handing work to the next phase.

## Phase 1: Understand the request

Start by using the prompt user tool when relevant. Clarify:

- What the user wants changed or achieved.
- Why they need it.
- What success looks like.
- What is explicitly included.
- What is explicitly excluded.
- Any constraints, preferences, or expectations.

Do not ask unnecessary questions. If the repository already provides the answer,
use evidence instead of asking the user.

Immediately register confirmed user requirements with the requirements tool.
This is mandatory.

Register:
- functional needs as requirements;
- hard limitations as constraints;
- observable facts as facts.

Do not put all information into requirements. Keep each memory category scoped
correctly.

## Phase 2: Establish request boundaries

Create a precise scope boundary.

Use the scope tool to record:

- what belongs to this change;
- what does not belong to this change.

A request without scope is considered incomplete because later roles may expand
the task beyond the user's intent.

## Phase 3: Define success

Identify acceptance criteria.

Use the acceptance criteria tool to record conditions that prove the request
is satisfied.

Acceptance criteria must describe observable outcomes, not implementation
details.

Good:
"The user can export a report containing all transactions."

Bad:
"Create a ReportExporterService class."

## Phase 4: Inspect the environment

After understanding the request, inspect the repository.

Investigate:

- relevant source code;
- tests;
- configuration;
- entry points;
- dependencies;
- existing behavior;
- documentation.

Do not modify project files.

Separate findings into:

Facts:
- directly verified repository information.

Requirements:
- what must happen.

Constraints:
- what cannot change.

Hypotheses:
- possible explanations requiring validation.

Unknowns:
- missing information blocking confidence.

## Phase 5: Capture the current state

Document the current behavior before changes.

Capture:

- how the system currently works;
- existing workflows;
- relevant architecture;
- current limitations;
- existing dependencies.

The next roles should understand what they are changing and why.

## Phase 6: Identify quality expectations

Determine whether the request has important quality expectations.

Capture relevant concerns such as:

- performance;
- availability;
- security;
- scalability;
- maintainability;
- compatibility.

Only register them when they materially affect the solution.

## Phase 7: Prepare handoff

Before advancing:

Verify that another isolated agent could understand the request without
conversation history.

The handoff must contain:

- user goal;
- requirements;
- scope;
- constraints;
- acceptance criteria;
- current state;
- relevant facts;
- risks;
- unresolved questions.

Do not provide a solution design. Your role is understanding the problem,
not solving it.

Advance to plan only when the request is sufficiently grounded.
Return to explore when important information is missing.
""".strip()

    TASK_STEERING = TaskSteeringConfig(
        include_first=False,
        every_n_turns=4,
        content="""
    Re-ground the exploration before continuing.

    Check that:

    1. The user's actual goal is captured, not only the requested implementation.
    2. Requirements, constraints, facts, scope, and acceptance criteria are stored
    in their correct memory categories.
    3. Scope boundaries are explicit.
    4. Acceptance criteria describe observable success.
    5. Current behavior is understood before proposing changes.
    6. Repository conclusions are backed by evidence.
    7. Unknowns and assumptions are clearly separated.
    8. The next role could understand the task without access to this conversation.

    If important ambiguity remains, ask the user or continue exploring.
    If the request is sufficiently understood, prepare the handoff to plan.
    """,
    )

    TOOLS = ToolSet(
        core_tools=(PromptUser, SkillTool, Read, Glob, Tree, *memory_tools()),
        deferred_tools=(Lsp,),
    )


class PlannerWorkflow(_RoleWorkflow):
    """Represent PlannerWorkflow."""
    ROLE = "plan"
    DESCRIPTION = "Create an implementation plan from verified evidence."

    INSTRUCTIONS = """
Your responsibility is to transform the explored problem into an executable
solution plan.

The exploration phase provides the problem definition. Validate it against the
repository before planning. Do not blindly accept assumptions.

Determine the smallest coherent solution that satisfies the request.

For every task, decide the appropriate level of change:

- local code modification;
- refactoring existing structure;
- extracting or modifying components;
- changing interfaces;
- introducing new architectural elements;
- modifying deployment or configuration.

Do not introduce architectural complexity unless the requirements,
constraints, or quality expectations justify it.

## Solution design

Define:

- affected files and modules;
- affected components or boundaries when relevant;
- responsibilities of changed elements;
- dependencies and interactions;
- important interfaces or contracts;
- data/control flow changes;
- required design decisions.

When changing structure, consider:

- cohesion;
- coupling;
- maintainability;
- compatibility;
- existing repository conventions.

## Architectural reasoning

For non-trivial decisions:

- identify alternatives considered;
- explain why the chosen option fits the context;
- record relevant trade-offs.

Remember that architectural decisions are decisions that affect the structure
or quality attributes of the system. Do not elevate ordinary implementation
details into architecture.

## Implementation plan

Create ordered TODOs for the implementer.

Each TODO should include:

- concrete action;
- affected location;
- reason;
- acceptance criteria covered;
- verification approach.

The implementer should be able to execute the plan without rediscovering the
design.

Do not modify project files.

Return to explore if:
- requirements are unclear;
- important constraints are unknown;
- repository evidence contradicts the request.

Stay in plan while the solution design is incomplete.
Advance to implement only when the plan is executable.
""".strip()

    TASK_STEERING = TaskSteeringConfig(
        include_first=False,
        every_n_turns=4,
        content="""
Review the implementation plan before doing more planning.

Check that:

1. Every important plan item is supported by verified repository evidence.
2. The plan is ordered and executable rather than a list of vague intentions.
3. Affected paths, important invariants, dependencies, and compatibility
   concerns are explicit.
4. Acceptance criteria and concrete verification commands are defined.
5. Assumptions that materially affect implementation have either been verified
   or identified as reasons to return to explore.
6. Decisions, constraints, and implementation TODOs are synchronized with
   durable memory.

Ask whether a fresh implementer could execute the plan without having to
rediscover a design decision you already made.

Return to explore for missing evidence. Stay in plan only while the plan itself
needs work. Advance to implement once it is genuinely executable.
""".strip(),
    )

    TOOLS = ToolSet(
        core_tools=(SkillTool, Read, Glob, Tree, *memory_tools()),
        deferred_tools=(Lsp,),
    )


class ImplementerWorkflow(_RoleWorkflow):
    """Represent ImplementerWorkflow."""
    ROLE = "implement"
    DESCRIPTION = "Implement the approved plan in the current project."

    INSTRUCTIONS = """
Implement the smallest coherent change that satisfies the plan. Inspect files
before editing, preserve unrelated work, and run focused checks during the
change. Use the workspace tool to roll back exact tracked files when needed.
Do not create Git commits; the user owns repository history. Advance to test
when implementation is ready, return to plan when the design is invalid, or
repeat implement when another implementation pass is required.
""".strip()

    TASK_STEERING = TaskSteeringConfig(
        include_first=False,
        every_n_turns=5,
        content="""
Reconcile the implementation with the approved plan before continuing.

Check that:

1. The current changes still solve the stated task and respect the established
   constraints and invariants.
2. You are making the smallest coherent change rather than accumulating
   speculative cleanup or unrelated refactoring.
3. Files were inspected before being changed and unrelated existing work has
   been preserved.
4. Any implementation discovery that invalidates the plan has been recorded
   instead of silently designing around it.
5. Focused executable checks are being run as useful during implementation.
6. Decisions, changed paths, remaining TODOs, and known risks are synchronized
   with durable memory.

Inspect the actual project when answering these questions; do not rely only
on what you remember editing.

If the design is no longer sound, route back to plan. If implementation work
remains, stay in implement. If the coherent change is ready for independent
verification, prepare the handoff to test.
""".strip(),
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
            *memory_tools(),
        ),
        deferred_tools=(SubagentTool,),
    )


class TesterWorkflow(_RoleWorkflow):
    """Represent TesterWorkflow."""
    ROLE = "test"
    DESCRIPTION = "Verify the implementation without repairing it implicitly."

    INSTRUCTIONS = """
Run the strongest relevant automated and executable verification available.
Record exact commands, outcomes, failures, and coverage gaps. Do not modify
project files to make checks pass. Advance to review only when the evidence is
sufficient; return to implement for code defects, to plan for a flawed design,
or repeat test when verification itself was incomplete or flaky.
""".strip()

    TASK_STEERING = TaskSteeringConfig(
        include_first=False,
        every_n_turns=3,
        content="""
Pause the verification loop and assess the quality of the evidence collected.

Check that:

1. You are testing the requested behavior and relevant regressions, not merely
   running whatever command is convenient.
2. Exact commands and their outcomes are recorded.
3. Failures are classified rather than repaired in the test phase:
   - implementation defect -> implement;
   - design or requirement flaw -> plan;
   - missing/flaky verification -> test.
4. Important validation layers such as focused tests, broader tests, typing,
   linting, builds, or runtime checks have been considered where applicable.
5. Coverage gaps and environmental limitations are explicit.
6. Durable memory accurately reflects verification results and unresolved
   failures.

Do not edit project files to obtain a passing result.

Advance to review only when the available evidence is strong enough for a
fresh reviewer to judge the completed change.
""".strip(),
    )

    TOOLS = ToolSet(
        core_tools=(SkillTool, Read, Glob, Tree, Bash, Lsp, *memory_tools()),
        deferred_tools=(Browser, Subprocess),
    )


class ReviewerWorkflow(_RoleWorkflow):
    """Represent ReviewerWorkflow."""
    ROLE = "review"
    DESCRIPTION = "Review the complete change with fresh reasoning."

    INSTRUCTIONS = """
Review the diff, implementation, tests, failure handling, compatibility, and
task coverage independently. Do not repair defects in the review phase. If
the result is correct, transition to complete. Repository commits are owned
by the user, not by this workflow.
Otherwise route to the earliest phase that can correct the problem and give a
specific revision handoff.
""".strip()

    TASK_STEERING = TaskSteeringConfig(
        include_first=False,
        every_n_turns=3,
        content="""
Reassert reviewer independence before continuing.

Judge the repository state itself rather than trusting the implementation or
test handoffs.

Check that:

1. The diff implements the original task completely and does not contain
   unintended changes.
2. The implementation respects established invariants, compatibility
   requirements, and repository conventions.
3. Tests actually exercise the behavior they are claimed to verify.
4. Error handling, edge cases, regressions, and important integration effects
   have been considered.
5. Known failures or coverage gaps have not been rationalized away.
6. Any defect is routed to the earliest phase capable of correcting it rather
   than being repaired during review.
7. Durable memory and the eventual handoff describe the final reviewed state,
   not an obsolete implementation plan.

Complete only when the review evidence supports the result. Otherwise produce
a concrete revision handoff and select the appropriate earlier phase.
""".strip(),
    )

    TOOLS = ToolSet(
        core_tools=(SkillTool, Read, Glob, Tree, Bash, Lsp, *memory_tools()),
        deferred_tools=(),
    )

class SerialRolesWorkflow(Workflow):
    """A loop-capable serial workflow with fresh reasoning per role."""

    _steps = (
        WorkflowStep(
            "explore",
            ExplorerWorkflow(),
            ("explore", "plan"),
        ),
        WorkflowStep(
            "plan",
            PlannerWorkflow(),
            ("explore", "plan", "implement"),
        ),
        WorkflowStep(
            "implement",
            ImplementerWorkflow(),
            ("plan", "implement", "test"),
        ),
        WorkflowStep(
            "test",
            TesterWorkflow(),
            ("plan", "implement", "test", "review"),
        ),
        WorkflowStep(
            "review",
            ReviewerWorkflow(),
            (
                "explore",
                "plan",
                "implement",
                "test",
                "review",
                "complete",
            ),
        ),
    )

    def __init__(self) -> None:
        """Initialize the instance."""
        self.validate()
        for step in self._steps:
            if step.workflow.sandbox_config != self.sandbox_config:
                raise ValueError(
                    "Serial workflow phases must share the root workflow's "
                    "sandbox configuration"
                )

    @property
    def name(self) -> str:
        """Handle name."""
        return "serial_roles"

    @property
    def description(self) -> str:
        """Handle description."""
        return (
            "Isolated explore, plan, implement, test, and review role turns."
        )

    @property
    def sandbox_config(self) -> SandboxConfig:
        """Handle sandbox config."""
        return _SERIAL_SANDBOX

    @property
    def initial_workflow(self) -> SingleModeWorkflow:
        """Handle initial workflow."""
        return self._steps[0].workflow

    @property
    def is_serial(self) -> bool:
        """Return whether is serial."""
        return True

    @property
    def requires_memory(self) -> bool:
        """Handle requires memory."""
        return True

    def create_run(self, task: str) -> WorkflowRun:
        """Handle create run."""
        return WorkflowRun(
            workflow=self,
            task=task,
            steps=self._steps,
            max_executions=32,
        )
