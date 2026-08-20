from abc import ABC
from citra.utils.json_schema import ChatCompletionTool
from citra.context import ExecutionContext
from citra.agent import AgentSession
from .tool import Tool

class SessionTool(Tool, ABC):
    def __init__(
        self, 
        context: ExecutionContext,
        session: AgentSession,
        definition: ChatCompletionTool
    ):
        super().__init__(context, definition)
        self.__session = session