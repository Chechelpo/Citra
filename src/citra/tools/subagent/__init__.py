"""
Subagent runtime.

The orchestrator can spawn short-lived worker agents that:

  * have a small, sandboxed filesystem of their own (one writable directory
    plus a configured set of read-only binds);
  * see only the ``read``, ``write``, ``edit``, ``bash`` and
    ``request_guidance`` tools;
  * can ask the orchestrator for guidance through a dedicated inbox;
  * run on dedicated threads and can be observed (poll), redirected (steer)
    or joined (sleep) by the orchestrator.

The subagent runtime is a deliberate, narrow tool: it is for delegating
well-defined component work to an isolated worker, not for letting the
subagent reason about the project as a whole. The orchestrator remains
the only entity that can write back to the user's project.
"""

# Import ordering matters: ``default_registry`` imports ``SubagentTool``
# eagerly, so we must not pull it in transitively while the registry is
# still being constructed. The spec module is the only one that can
# safely be exported here without touching the agent runtime.
from .spec import (
    SubagentSpec,
    SubagentSnapshot,
    SubagentStatus,
    TranscriptEntry,
)

# The remaining classes are exposed lazily through ``__getattr__`` so
# importing ``citra.tools.subagent`` does not transitively pull in
# ``citra.agent`` (which would create a circular import through the
# default tool registry).


_LAZY_EXPORTS = {
    "RequestGuidanceTool": "citra.tools.subagent.guidance",
    "SubagentMode": "citra.tools.subagent.mode",
    "SubagentWorkflow": "citra.tools.subagent.workflow",
    "SubagentSupervisor": "citra.tools.subagent.supervisor",
    "SubagentTool": "citra.tools.subagent.tool",
    "build_subagent_context": "citra.tools.subagent.factory",
    "build_subagent_mode": "citra.tools.subagent.mode",
    "build_subagent_workflow": "citra.tools.subagent.workflow",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(_LAZY_EXPORTS[name])
        value = vars(module)[name]
        globals()[name] = value
        return value
    raise AttributeError(
        f"module 'citra.tools.subagent' has no attribute {name!r}"
    )


__all__ = [
    "RequestGuidanceTool",
    "SubagentMode",
    "SubagentWorkflow",
    "SubagentSnapshot",
    "SubagentSpec",
    "SubagentStatus",
    "SubagentSupervisor",
    "SubagentTool",
    "TranscriptEntry",
    "build_subagent_context",
    "build_subagent_mode",
    "build_subagent_workflow",
]
