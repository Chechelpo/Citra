from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from citra.modes import Mode, SandboxConfig, TaskSteeringConfig
from citra.sandbox import SandboxMode
from citra.tools.default_registry import ToolSet, memory_tools
from citra.tools.session_memory import CheckpointTool
from citra.tools.subagent import SubagentTool
from citra.tools.transient import *
from citra.tools.transient import SkillTool
from citra.tools.skills.skill import Skill
from citra.utils.prompt import format_skills
from citra.utils.prompt import collect_environment
from citra.utils.prompt import PromptEnvironment
from citra.tools.transient import Subprocess
from citra.tools.transient import Browser

from .workflow import Workflow, WorkflowRun, WorkflowStep

if TYPE_CHECKING:
    from citra.context import ExecutionContext


_SERIAL_SANDBOX = SandboxConfig(mode=SandboxMode.FULL_SANDBOX)


class _RoleMode(Mode):
    """A stateless role used for exactly one isolated workflow phase."""

    ROLE: ClassVar[str]
    DESCRIPTION: ClassVar[str]
    INSTRUCTIONS: ClassVar[str]
    TOOLS: ClassVar[ToolSet]
    TASK_STEERING: ClassVar[TaskSteeringConfig]
    _AVAILABLE_SKILLS: ClassVar[tuple[Skill]] = tuple()

    @property
    def name(self) -> str:
        return f"serial:{self.ROLE}"

    @property
    def description(self) -> str:
        return self.DESCRIPTION

    @property
    def tool_set(self) -> ToolSet:
        return self.TOOLS

    @property
    def skills(self) -> tuple:
        return ()

    @property
    def sandbox_config(self) -> SandboxConfig:
        # Compatibility only. The SerialRolesWorkflow owns this policy.
        return _SERIAL_SANDBOX

    @property
    def task_steering(self) -> TaskSteeringConfig:
        return self.TASK_STEERING

    @property
    def initial_working_states(self) -> tuple[str, ...]:
        return ()

    @override
    def get_system_prompt(self, context: ExecutionContext) -> str:
        run = getattr(context, "workflow_run", None)
        if run is None:
            raise RuntimeError("Serial role mode requires an active workflow run")

        step = run.current_step
        checkpoint_name = CheckpointTool.resolve_definition_for_context(
            context
        ).function.name
        allowed = ", ".join(f"`{item}`" for item in step.allowed_next)
        environment: PromptEnvironment = collect_environment(context)
        
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


class ExplorerMode(_RoleMode):
    ROLE = "explore"
    DESCRIPTION = "Inspect the repository and establish grounded constraints."

    INSTRUCTIONS = """
First, use the prompt user tool to clarify the user request as much as possible. Ask for requirements, constraints, how the user wants to feel it, etc.

Second: Start your very first turn with registering the user's requirements with your requirements tool. This is mandatory

Third: Inspect the relevant source, tests, configuration, entry points, and existing
behavior. Run safe diagnostics when useful. Do not modify project files.
Produce a precise map of relevant paths, confirmed behavior, constraints,
risks, and unknowns. Advance to plan when the task is sufficiently grounded;
repeat explore only when a concrete evidence gap remains.

""".strip()

    TASK_STEERING = TaskSteeringConfig(
        include_first=False,
        every_n_turns=4,
        content="""
Re-ground the exploration before continuing.

Check that:

1. Your conclusions come from repository evidence rather than assumptions.
2. You have inspected the relevant source, tests, configuration, entry points,
   and executable behavior where useful.
3. Confirmed and relevant facts are clearly registered via the tool and have accurate citations separated from hypotheses and unknowns.
4. Important paths, requirements, constraints, risks, and unresolved questions
   are represented in durable memory.
5. You have not modified project files.

Ask whether another explorer could reconstruct the important repository facts
from the filesystem and memory without relying on your conversation history.

If a concrete evidence gap remains, keep exploring. If the task is sufficiently
grounded, stop broadening the investigation and prepare a precise handoff to
plan.
""".strip(),
    )

    TOOLS = ToolSet(
        core_tools=(PromptUser,SkillTool, Read, Glob, Tree, Bash, Lsp, *memory_tools()),
        deferred_tools=(),
    )


class PlannerMode(_RoleMode):
    ROLE = "plan"
    DESCRIPTION = "Create an implementation plan from verified evidence."

    INSTRUCTIONS = """
Validate the exploration handoff against the shared filesystem where needed.
Produce an ordered implementation plan with affected paths, invariants,
acceptance criteria, verification commands, and rollback or compatibility
concerns. 

    1. Register every part of your plan into TODOs for the implementer.
    2. Check that on all TODO completions, the project is finished

Do not modify project files. Return to explore if essential facts
are missing, repeat plan if the plan itself remains incomplete, or advance to
implement when it is executable.
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


class ImplementerMode(_RoleMode):
    ROLE = "implement"
    DESCRIPTION = "Implement the approved plan in the shared workspace."

    INSTRUCTIONS = """
Implement the smallest coherent change that satisfies the plan. Inspect files
before editing, preserve unrelated work, and run focused checks during the
change. Keep changes in the workflow workspace; do not apply them to the
authoritative source yet. Advance to test when implementation is ready,
return to plan when the design is invalid, or repeat implement when another
implementation pass is required.
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

Inspect the actual workspace when answering these questions; do not rely only
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
            Lsp,
            Commit,
            *memory_tools(),
        ),
        deferred_tools=(SubagentTool,),
    )


class TesterMode(_RoleMode):
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


class ReviewerMode(_RoleMode):
    ROLE = "review"
    DESCRIPTION = "Review the complete change with fresh reasoning."

    INSTRUCTIONS = """
Review the diff, implementation, tests, failure handling, compatibility, and
task coverage independently. Do not repair defects in the review phase. If
the result is correct, stage the intended project changes and apply them to
the authoritative source through the commit tool, then transition to complete.
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

Only apply the intended changes to the authoritative source when the review
evidence supports completion. Otherwise produce a concrete revision handoff
and select the appropriate earlier phase.
""".strip(),
    )

    TOOLS = ToolSet(
        core_tools=(SkillTool, Read, Glob, Tree, Bash, Lsp, Commit, *memory_tools()),
        deferred_tools=(),
    )

class SerialRolesWorkflow(Workflow):
    """A loop-capable serial workflow with fresh reasoning per role."""

    _steps = (
        WorkflowStep(
            "explore",
            ExplorerMode(),
            ("explore", "plan"),
        ),
        WorkflowStep(
            "plan",
            PlannerMode(),
            ("explore", "plan", "implement"),
        ),
        WorkflowStep(
            "implement",
            ImplementerMode(),
            ("plan", "implement", "test"),
        ),
        WorkflowStep(
            "test",
            TesterMode(),
            ("plan", "implement", "test", "review"),
        ),
        WorkflowStep(
            "review",
            ReviewerMode(),
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

    @property
    def name(self) -> str:
        return "serial_roles"

    @property
    def description(self) -> str:
        return (
            "Isolated explore, plan, implement, test, and review role turns."
        )

    @property
    def sandbox_config(self) -> SandboxConfig:
        return _SERIAL_SANDBOX

    @property
    def initial_mode(self) -> Mode:
        return self._steps[0].mode

    @property
    def is_serial(self) -> bool:
        return True

    @property
    def requires_memory(self) -> bool:
        return True

    def create_run(self, task: str) -> WorkflowRun:
        return WorkflowRun(
            workflow=self,
            task=task,
            steps=self._steps,
            max_executions=32,
        )