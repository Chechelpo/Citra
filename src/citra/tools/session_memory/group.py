from ..tool_group import ToolGroup
from .acceptance_criteria_tool import AcceptanceCriteriaTool
from .change_tool import ChangeTool
from .checkpoint_tool import CheckpointTool
from .constraint_tool import ConstraintTool
from .decision_tool import DecisionTool
from .fact_tool import FactTool
from .issue_tool import IssueTool
from .requirement_tool import RequirementTool
from .scope_tool import ScopeTool
from .todo_tool import TodoTool
from .verification_tool import VerificationTool
from .working_state_tool import WorkingStateTool


class SessionMemoryToolGroup(ToolGroup):
    GROUP_NAME = "Updated memory"
    GROUP_TOOLS = (
        RequirementTool,
        AcceptanceCriteriaTool,
        ScopeTool,
        ConstraintTool,
        FactTool,
        DecisionTool,
        TodoTool,
        ChangeTool,
        VerificationTool,
        IssueTool,
        WorkingStateTool,
        CheckpointTool,
    )
