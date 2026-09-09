"""Token efficient task workflow."""
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


class ImplementerWorkflow(StaticWorkflow):
    """Focused repository work in one persistent agent session."""

    _NAME = "surgeon"
    _DESCRIPTION = (
        "Token-efficient task workflow for focused repository work."
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

    @override
    def get_system_prompt(self, context: ExecutionContext) -> str:
        """Return get system prompt."""

        return build_system_prompt(
            context,
            give_name=True,
            add_coding_convetions=True, 
            preepend="""
# Role

You are an expert code developer.

# Guide

1. Inspect the repository using the semantic Tree tool.
2. Draft an initial plan to complete your task based only on this information.
3. Start programming directly based on that task

Once you have a slice, write a test for it. If the test fails, and the error of that failure isn't directly attributable to your changes, only then
can you read the rest of the source files.

""",
    append="""
# Tools

Use the available tools when they materially help complete the task.

Prefer targeted inspection over reading the entire repository.

Always prefer the use of specialized tools rather than plain bash use.
Do not use Bash for Git mutation.

# Verification

A task is not complete merely because the code appears correct.

Use the strongest practical verification available, such as:

- focused tests;
- existing test suites;
- builds;
- executable examples;

Do not weaken tests or validation merely to make them pass.

# Completion

Before finishing, make sure:

- the requested behavior is implemented;
- relevant verification passes;
- no obvious unfinished work remains;
- changes are internally consistent;

Report what was changed and any important verification results.
""".strip())

    @override
    def get_user_message_prefix(self, context: ExecutionContext) -> str:
        """Return the current workspace snapshot for the user's request."""
        return build_workspace_context(context)


__all__ = ["ImplementerWorkflow"]
