from dataclasses import dataclass
from typing import Any, override

from citra.agent import AgentSession
from citra.context import ExecutionContext
from citra.utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)

from .memory_tool import MemoryTool


@dataclass(frozen=True)
class DecisionExtract:
    id: int
    content: str


class DecisionTool(MemoryTool[DecisionExtract]):
    """
    Manage decisions retained in memory for the current agent run.

    Actions:
    - add: record one or more decisions made during the run
    - remove: remove one or more stale, invalid, or superseded decisions
    """

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="decision",
            description=(
                "Manage decisions made during the current agent run. "
                "Operations may target one decision or a batch. "
                "Use 'add' after implementation, architectural, behavioral, "
                "or design choices have actually been made. Use 'remove' "
                "when decisions are invalid, stale, obsolete, or superseded."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Decision operation to perform.",
                            enum=(
                                "add",
                                "remove",
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description=(
                                "Single decision to record for 'add'. "
                                "Use 'contents' to add multiple decisions."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="contents",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Decisions to add as a batch."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description=(
                                "Single decision ID for 'remove'. "
                                "Use 'ids' to remove multiple decisions."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            JsonSchema.integer(),
                            description=(
                                "Decision IDs to remove as a batch."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    def __init__(
        self,
        context: ExecutionContext,
        session: AgentSession,
    ) -> None:
        super().__init__(
            context=context,
            definition=self.DEFINITION,
            session=session,
        )

        self.__extracts: list[DecisionExtract] = []
        self.__next_id = 1

    @property
    @override
    def heading(
        self,
    ) -> str:
        return "Decisions"

    @override
    def get_extracts(
        self,
    ) -> list[DecisionExtract]:
        return list(
            self.__extracts
        )

    @override
    def format_extract(
        self,
        decision: DecisionExtract,
    ) -> str:
        return (
            f"- [{decision.id}] "
            f"{decision.content}"
        )

    @override
    def should_offer_documentation(
        self,
    ) -> bool:
        return bool(
            self.__extracts
        )

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        action = arguments["action"]

        if action == "add":
            return self._add(
                arguments
            )

        if action == "remove":
            return self._remove(
                arguments
            )

        raise ValueError(
            f"Unsupported decision action: {action}"
        )

    def _add(
        self,
        arguments: dict[str, Any],
    ) -> str:
        contents = self._get_contents(
            arguments
        )

        added: list[DecisionExtract] = []

        for content in contents:
            decision = DecisionExtract(
                id=self.__next_id,
                content=content,
            )

            self.__next_id += 1

            self.__extracts.append(
                decision
            )

            added.append(
                decision
            )

        if len(added) == 1:
            decision = added[0]

            return (
                f"Added DECISION [{decision.id}]: "
                f"{decision.content}"
            )

        return (
            f"Added {len(added)} DECISIONs "
            f"[{added[0].id}-{added[-1].id}]."
        )

    def _remove(
        self,
        arguments: dict[str, Any],
    ) -> str:
        ids = self._get_ids(
            arguments
        )

        # Resolve every ID first so an invalid ID cannot
        # partially mutate a batch removal.
        decisions = [
            self.__extracts[
                self._find_index(
                    decision_id
                )
            ]
            for decision_id in ids
        ]

        ids_set = set(
            ids
        )

        self.__extracts = [
            decision
            for decision in self.__extracts
            if decision.id not in ids_set
        ]

        if len(decisions) == 1:
            decision = decisions[0]

            return (
                f"Removed DECISION [{decision.id}]: "
                f"{decision.content}"
            )

        return (
            f"Removed {len(decisions)} DECISIONs "
            f"{self._format_ids(ids)}."
        )

    @staticmethod
    def _get_contents(
        arguments: dict[str, Any],
    ) -> list[str]:
        content = arguments.get(
            "content"
        )
        contents = arguments.get(
            "contents"
        )

        if (
            content is not None
            and contents is not None
        ):
            raise ValueError(
                "Use either 'content' or 'contents', not both."
            )

        if content is not None:
            contents = [
                content
            ]

        if not contents:
            raise ValueError(
                "'content' or 'contents' is required "
                "for decision action 'add'."
            )

        normalized: list[str] = []

        for index, item in enumerate(
            contents
        ):
            item = item.strip()

            if not item:
                raise ValueError(
                    f"contents[{index}] cannot be empty."
                )

            normalized.append(
                item
            )

        return normalized

    @staticmethod
    def _get_ids(
        arguments: dict[str, Any],
    ) -> list[int]:
        decision_id = arguments.get(
            "id"
        )
        ids = arguments.get(
            "ids"
        )

        if (
            decision_id is not None
            and ids is not None
        ):
            raise ValueError(
                "Use either 'id' or 'ids', not both."
            )

        if decision_id is not None:
            ids = [
                decision_id
            ]

        if not ids:
            raise ValueError(
                "'id' or 'ids' is required "
                "for decision action 'remove'."
            )

        if len(ids) != len(set(ids)):
            raise ValueError(
                "'ids' cannot contain duplicates."
            )

        return list(
            ids
        )

    def _find_index(
        self,
        decision_id: int,
    ) -> int:
        for index, decision in enumerate(
            self.__extracts
        ):
            if decision.id == decision_id:
                return index

        raise ValueError(
            f"DECISION [{decision_id}] does not exist."
        )

    @staticmethod
    def _format_ids(
        ids: list[int],
    ) -> str:
        return (
            "["
            + ", ".join(
                str(decision_id)
                for decision_id in ids
            )
            + "]"
        )