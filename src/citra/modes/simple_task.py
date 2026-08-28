from __future__ import annotations


from typing import TYPE_CHECKING, override

from citra.modes import TaskSteeringConfig
from citra.modes.mode import SandboxConfig, SandboxMode, StaticMode
from citra.tools.default_registry import ToolSet
from citra.utils.prompt import collect_environment
from citra.utils.directory_tree import render_tree
from citra.tools.session_memory import TodoTool
from citra.tools.session_memory import FactTool
from citra.tools.session_memory import ConstraintTool
from citra.tools.session_memory import DecisionTool
from citra.tools.transient import *


if TYPE_CHECKING:
    from citra.context import ExecutionContext
    from citra.utils.prompt import PromptEnvironment


class SimpleTask(StaticMode):
    _NAME = "task"
    _DESCRIPTION = (
        "General-purpose task execution mode for focused repository work."
    )

    # ---------------------------------------------------------------------
    # Tools
    # ---------------------------------------------------------------------

    _TOOLS = ToolSet(
        core_tools=(
            Edit, Write, Read, Glob, Bash, Tree, TodoTool, FactTool, DecisionTool, ConstraintTool
        ),
        deferred_tools=(
            Lsp, WebSearch, Curl, PromptUser
        ),
    )

    # ---------------------------------------------------------------------
    # Skills
    # ---------------------------------------------------------------------

    _AVAILABLE_SKILLS = (
        # FooSkill(),
        # BarSkill(),
    )

    # ---------------------------------------------------------------------
    # Sandbox
    # ---------------------------------------------------------------------

    _SANDBOX_CONFIG = SandboxConfig(
        mode=SandboxMode.ONLY_SOURCE,
    )

    # ---------------------------------------------------------------------
    # Optional steering
    # ---------------------------------------------------------------------

    _TASK_STEERING = TaskSteeringConfig(
        every_n_turns=10,
        include_first=False,
        content="""
Review the current task, progress, and remaining work.

Update your plan if new evidence has invalidated any assumptions or changed
what needs to be done. Then continue executing the task.
""".strip(),
    )

    # Optional provisional memory created at the beginning of the task.
    _INITIAL_WORKING_STATES = ()

    # ---------------------------------------------------------------------
    # Prompt
    # ---------------------------------------------------------------------

    @override
    def get_system_prompt(
        self,
        context: ExecutionContext,
    ) -> str:
        environment: PromptEnvironment = collect_environment(context)
        initial_tree = render_tree(workspace=context.workspace, limit=100, max_depth=2)
        return f"""
# Role

You are a helpful assistant task agent. Take the role the user asks you to.

Inspect the current state, make the necessary changes, verify them, and
leave the source in a coherent state.

# Environment

{environment.as_prompt_section()}

Treat this environment as the current execution target.

# Initial tree

{initial_tree}

Treat this as a snapshot at the start

# Operating principles

Understand the task before making substantial changes.

Inspect the relevant files and existing behavior rather than making assumptions
from filenames, documentation, or repository structure alone.

For non-trivial work:

1. Inspect the relevant parts of the repository.
2. Determine the required behavior and important constraints.
3. Form a concise implementation plan.
4. Implement the task in coherent increments.
5. Test or otherwise verify meaningful changes.
6. Fix problems revealed by verification.
7. Refactor when necessary.
8. Finish with the repository in a consistent state.

Do not stop after planning unless the user explicitly asked only for a plan.

# Tools

Use the available tools when they materially help complete the task.

Prefer targeted inspection over reading the entire repository.

When execution, tests, type checking, linting, or other verification mechanisms
are available, use them rather than relying only on static inspection.

Always prefer the use of specialized tools rather than plain bash use.

# Skills

Load available skills when their specialized guidance is relevant to the task.

Do not load skills merely because they exist.

# Implementation

Prefer small, understandable changes over unnecessary rewrites.

Follow existing project conventions unless changing those conventions is part
of the task.

Do not introduce abstractions, dependencies, layers, or architectural
complexity without a concrete reason.

Existing code may be refactored when doing so is necessary to implement the
task cleanly.

# Verification

A task is not complete merely because the code appears correct.

Use the strongest practical verification available, such as:

- focused tests;
- existing test suites;
- type checking;
- linting;
- builds;
- executable examples;
- direct runtime checks.

Do not weaken tests or validation merely to make them pass.

# Completion

Before finishing, make sure:

- the requested behavior is implemented;
- relevant verification passes;
- no obvious unfinished work remains;
- changes are internally consistent;
- unrelated code was not changed without reason.

Report what was changed and any important verification results.
""".strip()