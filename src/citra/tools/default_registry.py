from .transient import *
from .session_memory import *

from .tool_registry import ToolRegistry


TOOL_REGISTRY = ToolRegistry()

# Transient tools
TOOL_REGISTRY.register("read", Read)
TOOL_REGISTRY.register("write", Write)
TOOL_REGISTRY.register("edit", Edit)
TOOL_REGISTRY.register("glob", Glob)
TOOL_REGISTRY.register("grep", Grep)
TOOL_REGISTRY.register("git", Git)
TOOL_REGISTRY.register("tree", Tree)

TOOL_REGISTRY.register("bash", Bash)
TOOL_REGISTRY.register("web_search", WebSearch)
TOOL_REGISTRY.register("prompt_user", PromptUser)

# Session / memory tools
TOOL_REGISTRY.register("todo", TodoTool)
TOOL_REGISTRY.register("fact", FactTool)
TOOL_REGISTRY.register("decision", DecisionTool)
TOOL_REGISTRY.register("constraint", ConstraintTool)