from abc import ABC

from citra.agent import AgentSession
from citra.context import ExecutionContext

from .tool import Tool


class SessionTool(Tool, ABC):
    """
    Base class for tools that require access to the active agent session.

    Model-facing tool definitions are resolved by Tool through
    definitions_for_context(); SessionTool only adds session state.
    """

    def __init__(
        self,
        context: ExecutionContext,
        session: AgentSession,
    ) -> None:
        super().__init__(
            context=context,
        )

        self.__session = session

    @property
    def session(self) -> AgentSession:
        return self.__session

    def rebind_session(self, session: AgentSession) -> None:
        """Bind a durable session tool to the currently active session."""
        self.__session = session
