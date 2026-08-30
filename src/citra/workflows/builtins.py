"""Built-in simple and architect workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from citra.modes import Mode, SandboxConfig
from citra.modes.mode import StaticMode
from citra.sandbox import SandboxMode
from citra.tools.default_registry import ToolSet, all_tools
from citra.tools.subagent import SubagentTool
from citra.utils.directory_tree import render_tree
from citra.utils.prompt import collect_environment

from .workflow import SingleModeWorkflow

if TYPE_CHECKING:
    from citra.context import ExecutionContext


class ArchitectMode(StaticMode):
    """Single orchestrator mode that delegates bounded component work."""

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
    _SANDBOX_CONFIG = SandboxConfig(mode=SandboxMode.FULL_SANDBOX)

    @override
    def get_system_prompt(self, context: ExecutionContext) -> str:
        environment = collect_environment(context)
        tree = render_tree(workspace=context.workspace, limit=120, max_depth=3)
        subagent_name = SubagentTool.resolve_definition_for_context(
            context
        ).function.name
        return f"""
# Role: system architect and integrator

Translate the user's high-level greenfield or system-level requirement into a
coherent implementation. You are one orchestrator agent in one workflow mode;
component workers are genuine isolated subagents.

# Environment

{environment.as_prompt_section()}

# Initial tree

{tree}

# Required operating sequence

1. Inspect the requirement and relevant repository state.
2. Design the complete system once: components, ownership, dependency
   direction, public APIs, shared data models, and acceptance criteria.
3. Materialize the architecture and frozen component contracts in the shared
   workspace before delegation. Treat those contracts as immutable inputs to
   component workers.
4. Use `{subagent_name}` to delegate only non-overlapping component write
   roots. Give each worker a self-contained task, relevant read-only context,
   its frozen API contract, and acceptance criteria.
5. Poll and supervise every worker. Resolve guidance requests and do not
   integrate until all required workers are terminal.
6. Integrate component outputs yourself, run contract and cross-component
   tests, correct integration defects, and review the complete system.
7. Apply only the verified final change set to the authoritative source.

# Invariants

- Never let concurrent workers own overlapping paths.
- Workers do not change shared contracts or redesign the whole system.
- A failed worker is explicit workflow evidence, not permission to ignore a
  component.
- Verify APIs against the frozen contracts before integration.
- The architect remains responsible for final correctness and completeness.
""".strip()


def simple_workflow(mode: Mode) -> SingleModeWorkflow:
    """Wrap a selected mode and inherit its sandbox policy."""
    return SingleModeWorkflow(
        name="simple",
        description="One persistent agent running a selected mode.",
        mode=mode,
        sandbox_config=None,
    )


def architect_workflow() -> SingleModeWorkflow:
    mode = ArchitectMode()
    return SingleModeWorkflow(
        name="architect",
        description=(
            "One architect mode orchestrating isolated component subagents."
        ),
        mode=mode,
        sandbox_config=SandboxConfig(mode=SandboxMode.FULL_SANDBOX),
    )


__all__ = ["ArchitectMode", "architect_workflow", "simple_workflow"]
