from abc import ABC

from citra.agent import AgentSession
from citra.context import ExecutionContext
from citra.utils.json_schema import ChatCompletionTool

from .tool import Tool


class SessionTool(Tool, ABC):
    def __init__(
        self,
        context: ExecutionContext,
        session: AgentSession,
        definition: ChatCompletionTool,
    ) -> None:
        super().__init__(
            context=context,
            definition=definition,
        )
        self.__session = session

    @property
    def session(self) -> AgentSession:
        return self.__session
