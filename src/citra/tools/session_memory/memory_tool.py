from __future__ import annotations

from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar
from weakref import WeakKeyDictionary


from ..session_tool import SessionTool

if TYPE_CHECKING:
    from citra.context import ExecutionContext
    from citra.agent import AgentSession
    from ..tool import ToolDefinition


TExtract = TypeVar("TExtract")


@dataclass(frozen=True)
class PromotionRef:
    """Represent PromotionRef."""
    kind: str
    memory_id: int


@dataclass(frozen=True)
class WorkingStateExtract:
    """Represent WorkingStateExtract."""
    id: int
    content: str
    created_turn: int
    updated_turn: int
    promotions: tuple[PromotionRef, ...] = ()


@dataclass
class _WorkingStateRecord:
    """Represent WorkingStateRecord."""
    extract: WorkingStateExtract
    status: str = "active"


class ConversationMemoryState:
    """Shared provisional state for all memory tools in one AgentSession."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self._working_states: dict[int, _WorkingStateRecord] = {}
        self._next_working_state_id = 1

    def create_working_state(
        self,
        content: str,
        *,
        turn: int,
    ) -> WorkingStateExtract:
        """Handle create working state."""
        content = content.strip()

        if not content:
            raise ValueError(
                "Working-state content cannot be empty."
            )

        working_state = WorkingStateExtract(
            id=self._next_working_state_id,
            content=content,
            created_turn=turn,
            updated_turn=turn,
        )

        self._next_working_state_id += 1

        self._working_states[
            working_state.id
        ] = _WorkingStateRecord(
            extract=working_state,
        )

        return working_state

    def get_working_state(
        self,
        working_state_id: int,
        *,
        require_active: bool = True,
    ) -> WorkingStateExtract:
        """Return get working state."""
        record = self._working_states.get(
            working_state_id
        )

        if record is None:
            raise ValueError(
                f"Working state [W{working_state_id}] does not exist."
            )

        if (
            require_active
            and record.status != "active"
        ):
            raise ValueError(
                f"Working state [W{working_state_id}] is "
                f"{record.status} and cannot be promoted or modified."
            )

        return record.extract

    def active_working_states(
        self,
    ) -> list[WorkingStateExtract]:
        """Handle active working states."""
        return [
            record.extract
            for record in self._working_states.values()
            if record.status == "active"
        ]

    def update_working_state(
        self,
        working_state_id: int,
        content: str,
        *,
        turn: int,
    ) -> WorkingStateExtract:
        """Handle update working state."""
        current = self.get_working_state(
            working_state_id
        )

        content = content.strip()

        if not content:
            raise ValueError(
                "Working-state content cannot be empty."
            )

        updated = replace(
            current,
            content=content,
            updated_turn=turn,
        )

        self._working_states[
            working_state_id
        ].extract = updated

        return updated

    def register_promotion(
        self,
        working_state_id: int,
        *,
        kind: str,
        memory_id: int,
    ) -> None:
        """Handle register promotion."""
        current = self.get_working_state(
            working_state_id
        )

        ref = PromotionRef(
            kind=kind,
            memory_id=memory_id,
        )

        if ref in current.promotions:
            raise ValueError(
                f"Working state [W{working_state_id}] already records "
                f"{kind.upper()} [{memory_id}]."
            )

        updated = replace(
            current,
            promotions=(
                *current.promotions,
                ref,
            ),
        )

        self._working_states[
            working_state_id
        ].extract = updated

    def unregister_promotion(
        self,
        working_state_id: int,
        *,
        kind: str,
        memory_id: int,
    ) -> None:
        """Handle unregister promotion."""
        record = self._working_states.get(
            working_state_id
        )

        if record is None:
            return

        ref = PromotionRef(
            kind=kind,
            memory_id=memory_id,
        )

        record.extract = replace(
            record.extract,
            promotions=tuple(
                existing
                for existing
                in record.extract.promotions
                if existing != ref
            ),
        )

    def resolve_working_state(
        self,
        working_state_id: int,
    ) -> WorkingStateExtract:
        """Return resolve working state."""
        current = self.get_working_state(
            working_state_id
        )

        if not current.promotions:
            raise ValueError(
                f"Working state [W{working_state_id}] has no promotions. "
                "Promote any durable consequences first, or discard it "
                "if no durable memory is warranted."
            )

        self._working_states[
            working_state_id
        ].status = "resolved"

        return current

    def discard_working_state(
        self,
        working_state_id: int,
    ) -> WorkingStateExtract:
        """Handle discard working state."""
        current = self.get_working_state(
            working_state_id
        )

        if current.promotions:
            rendered = ", ".join(
                f"{ref.kind.upper()} [{ref.memory_id}]"
                for ref in current.promotions
            )

            raise ValueError(
                f"Working state [W{working_state_id}] has durable "
                f"promotions ({rendered}) and cannot be discarded. "
                "Remove those memories first or resolve the working state."
            )

        self._working_states[
            working_state_id
        ].status = "discarded"

        return current


_SESSION_STATES: WeakKeyDictionary[
    Any,
    ConversationMemoryState,
] = WeakKeyDictionary()

_FALLBACK_SESSION_STATES: dict[
    int,
    tuple[Any, ConversationMemoryState],
] = {}


def conversation_memory_state(
    session: Any,
) -> ConversationMemoryState:
    """
    Return the shared memory state for a session.

    Weak-key storage is preferred. A strong-reference fallback supports
    session implementations that are unhashable or cannot be weak-referenced.
    """

    try:
        state = _SESSION_STATES.get(
            session
        )
    except TypeError:
        state = None

    else:
        if state is not None:
            return state

        try:
            state = ConversationMemoryState()

            _SESSION_STATES[
                session
            ] = state

            return state

        except TypeError:
            pass

    key = id(
        session
    )

    existing = _FALLBACK_SESSION_STATES.get(
        key
    )

    if (
        existing is not None
        and existing[0] is session
    ):
        return existing[1]

    state = ConversationMemoryState()

    _FALLBACK_SESSION_STATES[
        key
    ] = (
        session,
        state,
    )

    return state


class MemoryTool(
    SessionTool,
    ABC,
    Generic[TExtract],
):
    """
    Base class for session-scoped memory tools.

    All memory tools in the same AgentSession share one
    ConversationMemoryState.
    """

    INVALIDATES_TOOL_CACHE = False

    def __init__(
        self,
        *,
        context: ExecutionContext,
        session: AgentSession,
    ) -> None:
        """Initialize the instance."""
        super().__init__(
            context=context,
            session=session,
        )

        self.memory_state = conversation_memory_state(
            session
        )

    @property
    @abstractmethod
    def heading(
        self,
    ) -> str:
        """Handle heading."""
        ...

    @abstractmethod
    def get_extracts(
        self,
    ) -> list[TExtract]:
        """Return get extracts."""
        ...

    @abstractmethod
    def format_extract(
        self,
        extract: TExtract,
    ) -> str:
        """Handle format extract."""
        ...

    @abstractmethod
    def should_offer_documentation(
        self,
    ) -> bool:
        """
        Return whether this memory type may contain information worth
        persisting into repository documentation at the end of the work.
        """
        ...

    def require_working_state(
        self,
        working_state_id: int,
    ) -> WorkingStateExtract:
        """Handle require working state."""
        return self.memory_state.get_working_state(
            working_state_id,
            require_active=True,
        )

    def register_promotion(
        self,
        working_state_id: int,
        *,
        kind: str,
        memory_id: int,
    ) -> None:
        """Handle register promotion."""
        self.memory_state.register_promotion(
            working_state_id,
            kind=kind,
            memory_id=memory_id,
        )

    def unregister_promotion(
        self,
        working_state_id: int,
        *,
        kind: str,
        memory_id: int,
    ) -> None:
        """Handle unregister promotion."""
        self.memory_state.unregister_promotion(
            working_state_id,
            kind=kind,
            memory_id=memory_id,
        )

    def format_for_llm(
        self,
    ) -> str:
        """Handle format for llm."""
        extracts = self.get_extracts()

        if not extracts:
            return ""

        return "\n".join(
            [
                f"## {self.heading}",
                *(
                    self.format_extract(
                        extract
                    )
                    for extract in extracts
                ),
            ]
        )