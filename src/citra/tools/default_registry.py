from .transient import *
from .session_memory import *

from .tool_registry import ToolRegistry


TOOL_REGISTRY = ToolRegistry()

# Core transient tools
TOOL_REGISTRY.register("read", Read)
TOOL_REGISTRY.register("write", Write)
TOOL_REGISTRY.register("edit", Edit)
TOOL_REGISTRY.register("glob", Glob)
TOOL_REGISTRY.register("grep", Grep)
TOOL_REGISTRY.register("git", Git)
TOOL_REGISTRY.register("tree", Tree)
TOOL_REGISTRY.register("materialize", Materialize)
TOOL_REGISTRY.register("commit", Commit)
TOOL_REGISTRY.register("bash", Bash)
TOOL_REGISTRY.register("prompt_user", PromptUser)
TOOL_REGISTRY.register("lsp", Lsp)
TOOL_REGISTRY.register("skill", SkillTool)

# Deferred / specialized tools
TOOL_REGISTRY.register(
    "subprocess",
    Subprocess,
    deferred=True,
    summary="Run or manage longer-lived subprocesses.",
)
TOOL_REGISTRY.register(
    "browser",
    Browser,
    deferred=True,
    summary="Interact with websites in a browser.",
)
TOOL_REGISTRY.register(
    "curl",
    Curl,
    deferred=True,
    summary="Make direct HTTP requests.",
)
TOOL_REGISTRY.register(
    "web_search",
    WebSearch,
    deferred=True,
    summary="Search the public web.",
)
TOOL_REGISTRY.register(
    "document",
    Document,
    deferred=True,
    summary="Create or modify rich documents.",
)
TOOL_REGISTRY.register(
    "diagram",
    Diagram,
    deferred=True,
    summary="Create diagrams.",
)

# Session / memory tools remain core so persisted memory always stays visible.
TOOL_REGISTRY.register("todo", TodoTool)
TOOL_REGISTRY.register("fact", FactTool)
TOOL_REGISTRY.register("decision", DecisionTool)
TOOL_REGISTRY.register("constraint", ConstraintTool)
TOOL_REGISTRY.register("checkpoint", CheckpointTool)
TOOL_REGISTRY.register("working_state", WorkingStateTool)
