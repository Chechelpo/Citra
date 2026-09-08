"""General-purpose single-mode task workflow."""
from __future__ import annotations

from typing import TYPE_CHECKING, override

from citra.tools.default_registry import ToolSet
from citra.tools.session_memory import (
    ConstraintTool,
    DecisionTool,
    FactTool,
    TodoTool,
)
from citra.tools.developer import Lsp
from citra.tools.documents import Diagram, Document
from citra.tools.editing import Edit, Write
from citra.tools.execution import Bash, Python
from citra.tools.explorer import Find, Glob, Grep, Read, Tree
from citra.tools.interaction import PromptUser
from citra.tools.web import Browser, WebSearch
from citra.tools.workspace import Workspace
from citra.utils.directory_tree import render_tree
from citra.workflows.sys_prompt import build_system_prompt, build_workspace_context
from citra.tools.skills.coding_conventions import coding_skills
from citra.tools.session_memory import RequirementTool
from citra.tools.skills.tool import SkillTool

from .workflow import SandboxConfig, StaticWorkflow, TaskSteeringConfig

if TYPE_CHECKING:
    from citra.context import ExecutionContext


class TaskWorkflow(StaticWorkflow):
    """Focused repository work in one persistent agent session."""

    _NAME = "task"
    _DESCRIPTION = (
        "General-purpose task workflow for focused repository work."
    )
    _AVAILABLE_SKILLS= (
        coding_skills()
    )
    _TOOLS = ToolSet(
        core_tools=(
            Edit,
            Write,
            Find,
            Read,
            Glob,
            Grep,
            Bash,
            SkillTool,
            Workspace,
            Tree,
            RequirementTool,
            TodoTool,
            FactTool,
            DecisionTool,
            ConstraintTool,
        ),
        deferred_tools=(Lsp, WebSearch, Browser, PromptUser, Python),
    )
    _SANDBOX_CONFIG = SandboxConfig()
    _TASK_STEERING = TaskSteeringConfig(
        every_n_turns=10,
        content=(
            "Review the current task, progress, and remaining work.\n\n"
            "Update your plan if new evidence has invalidated any assumptions "
            "or changed what needs to be done. Then continue executing the task."
        ),
    )

    @override
    def get_system_prompt(self, context: ExecutionContext) -> str:
        """Return get system prompt."""

        return build_system_prompt(
            context,
            give_name=True,
            add_coding_convetions=True, 
            preepend="""
# Role

You are a helpful assistant task agent. Take the role the user asks you to.

Inspect the current state, make the necessary changes, verify them, and leave
the project in a coherent state. Do not create Git commits or stage files;
repository history belongs to the user. Use the workspace tool to roll back an
exact tracked file when an attempted change is wrong.

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
""",
    append="""
# Tools

Use the available tools when they materially help complete the task.

Prefer targeted inspection over reading the entire repository.

When execution, tests, type checking, linting, or other verification mechanisms
are available, use them rather than relying only on static inspection.

Always prefer the use of specialized tools rather than plain bash use.
Do not use Bash for Git mutation.

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
""".strip())

    @override
    def get_user_message_prefix(self, context: ExecutionContext) -> str:
        """Return the current workspace snapshot for the user's request."""
        return build_workspace_context(context)


__all__ = ["TaskWorkflow"]
