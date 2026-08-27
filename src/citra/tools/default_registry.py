from gettext import find
from citra.tools.session_memory import DecisionTool
from citra.tools.session_memory import WorkingStateTool
from citra.tools.session_memory import MemoryTool
from citra.tools.session_memory import FactTool
from citra.tools.session_memory import ConstraintTool
from citra.tools.session_memory import TodoTool
from functools import reduce
from dataclasses import dataclass
from .transient import *
from .session_memory import *
from .tool import Tool

__all__ =  ["all_tools", "ToolSet"]

@dataclass(frozen=True)
class ToolSet :
    core_tools: tuple[type[Tool], ...]
    deferred_tools : tuple[type[Tool], ...]

    def __post_init__(self):
        def add_if_duplicate(duplicates: list[type[Tool]], check_not_in, new_tool: type[Tool]) -> list[type[Tool]]:
            if new_tool in check_not_in and new_tool not in duplicates:
                duplicates.append(new_tool)
            
            return duplicates
        
        duplicates = reduce(
            lambda duplicated,core_tool: add_if_duplicate(duplicated, self.deferred_tools, core_tool), 
            self.core_tools, 
            []
        )

        if len(duplicates) > 0:
            raise Exception(f"Duplicate tools found: {duplicates}")

    def get_tool_with_id(self, id:str) -> type[Tool] | None:
        for tool in self.core_tools:
            if tool.id == id:
                return tool
                
        for tool in self.deferred_tools:
            if tool.id == id:
                return tool
        
        return None

    def allowed_tools(self) -> tuple[type[Tool], ...]:
        return self.core_tools + self.deferred_tools
    
    def is_core_tool(self, tool:type[Tool]) -> bool:
        return tool in self.core_tools
    
    def is_deffered_tool(self, tool:type[Tool]) -> bool:
        return tool in self.deferred_tools

    def is_allowed_tool(self, tool:type[Tool])-> bool:
        return self.is_deffered_tool(tool) or self.is_core_tool(tool)
    
def memory_tools() -> tuple[type[Tool], ...]:
    return (
        TodoTool,
        DecisionTool,
        ConstraintTool,
        FactTool,
        MemoryTool,
        WorkingStateTool
    )


def all_tools(
    excluded: set[type[Tool]] = set()
) -> tuple[type[Tool], ...]:
    _STATIC_REGISTRY : tuple[type[Tool], ...] = (
        Read,
        Write,
        Edit,
        Glob,
        Tree,
        Commit,
        PromptUser,
        Lsp,
        SkillTool,
        Git,
        Subprocess,
        Browser,
        Curl,
        WebSearch,
        Document,
        Diagram,

        TodoTool,
        FactTool,
        DecisionTool,
        ConstraintTool,
        CheckpointTool,
        WorkingStateTool
    )
    
    return tuple(
        tool
        for tool in _STATIC_REGISTRY
        if tool not in excluded
    )
