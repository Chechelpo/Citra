"""Built-in single-mode workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from citra.tools.default_registry import ToolSet, all_tools
from citra.tools.subagent.tool import SubagentTool
from citra.utils.directory_tree import render_tree

from .chat import ChatWorkflow
from .task import TaskWorkflow
from .workflow import SandboxConfig, SingleModeWorkflow, StaticWorkflow

if TYPE_CHECKING:
    from citra.context import ExecutionContext


class ArchitectWorkflow(StaticWorkflow):
    """A single agent that delegates bounded component work."""

    _NAME = "architect"
    _DESCRIPTION = (
        "Design a system once, freeze boundaries, delegate components, "
        "and integrate."
    )
    _TOOLS = ToolSet(
        core_tools=(SubagentTool, *all_tools(are_deferred=False)),
        deferred_tools=tuple(
            tool
            for tool in all_tools(are_deferred=True)
            if tool is not SubagentTool
        ),
    )
    _SANDBOX_CONFIG = SandboxConfig()

    @override
    def get_system_prompt(self, context: ExecutionContext) -> str:
        """Return get system prompt."""
        tree = render_tree(workspace=context.workspace, limit=120, max_depth=3)
        subagent_name = SubagentTool.resolve_definition_for_context(
            context
        ).function.name
        return f"""
# Role: system architect and integrator

Translate the user's high-level greenfield or system-level requirement into a
coherent implementation. You are one orchestrator agent in one workflow;
component workers are genuine isolated subagents.

# Initial tree

{tree}

# Required operating sequence

1. Inspect the requirement and relevant repository state.
2. Design the complete system once: components, ownership, dependency
   direction, public APIs, shared data models, and acceptance criteria.
3. Write the architecture and frozen component contracts into the project
   before delegation. Treat those contracts as immutable worker inputs.
4. Use `{subagent_name}` to delegate only non-overlapping component write
   roots. Give each worker a self-contained task, relevant read-only context,
   its frozen API contract, and acceptance criteria.
5. Poll and supervise every worker. Resolve guidance requests and do not
   integrate until all required workers are terminal.
6. Integrate component outputs yourself, run contract and cross-component
   tests, correct integration defects, and review the complete system.
7. Leave the verified final change set uncommitted for the user to review and
   commit.

# Invariants

- Never let concurrent workers own overlapping paths.
- Workers do not change shared contracts or redesign the whole system.
- A failed worker is explicit workflow evidence, not permission to ignore a
  component.
- Verify APIs against the frozen contracts before integration.
- The architect remains responsible for final correctness and completeness.
""".strip()


def simple_workflow(workflow: SingleModeWorkflow) -> SingleModeWorkflow:
    """Compatibility helper; a single-mode workflow needs no wrapper."""
    return workflow


def architect_workflow() -> ArchitectWorkflow:
    """Compatibility factory for the built-in architect workflow."""
    return ArchitectWorkflow()


ArchitectMode = ArchitectWorkflow


__all__ = [
    "ArchitectMode",
    "ArchitectWorkflow",
    "ChatWorkflow",
    "TaskWorkflow",
    "architect_workflow",
    "simple_workflow",
]
