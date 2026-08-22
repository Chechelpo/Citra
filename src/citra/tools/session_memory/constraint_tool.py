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
class ConstraintExtract:
    id: int
    content: str


class ConstraintTool(MemoryTool[ConstraintExtract]):
    """
    Manage constraints retained for the current conversation lifecycle.

    Actions:
    - add: record one or more constraints that must be respected
    - remove: remove one or more stale or invalid constraints
    """

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="constraint",
            description=(
                "Manage constraints for the current conversation. "
                "Operations may target one constraint or a batch. "
                "Use 'add' to record rules, requirements, invariants, "
                "limitations, or compatibility constraints that must be "
                "respected. Use 'remove' when constraints are stale, "
                "incorrect, obsolete, or no longer applicable."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Constraint operation to perform.",
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
                                "Single constraint to record for 'add'. "
                                "Use 'contents' to add multiple constraints."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="contents",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Constraints to add as a batch."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description=(
                                "Single constraint ID for 'remove'. "
                                "Use 'ids' to remove multiple constraints."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            JsonSchema.integer(),
                            description=(
                                "Constraint IDs to remove as a batch."
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

        self.__extracts: list[ConstraintExtract] = []
        self.__next_id = 1

    @property
    @override
    def heading(
        self,
    ) -> str:
        return "Constraints"

    @override
    def get_extracts(
        self,
    ) -> list[ConstraintExtract]:
        return list(
            self.__extracts
        )

    @override
    def format_extract(
        self,
        extract: ConstraintExtract,
    ) -> str:
        return (
            f"- [{extract.id}] "
            f"{extract.content}"
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
            f"Unsupported constraint action: {action}"
        )

    def _add(
        self,
        arguments: dict[str, Any],
    ) -> str:
        contents = self._get_contents(
            arguments
        )

        added: list[ConstraintExtract] = []

        for content in contents:
            constraint = ConstraintExtract(
                id=self.__next_id,
                content=content,
            )

            self.__next_id += 1

            self.__extracts.append(
                constraint
            )

            added.append(
                constraint
            )

        if len(added) == 1:
            constraint = added[0]

            return (
                f"Added CONSTRAINT [{constraint.id}]: "
                f"{constraint.content}"
            )

        return (
            f"Added {len(added)} CONSTRAINTs "
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
        constraints = [
            self.__extracts[
                self._find_index(
                    constraint_id
                )
            ]
            for constraint_id in ids
        ]

        ids_set = set(
            ids
        )

        self.__extracts = [
            constraint
            for constraint in self.__extracts
            if constraint.id not in ids_set
        ]

        if len(constraints) == 1:
            constraint = constraints[0]

            return (
                f"Removed CONSTRAINT [{constraint.id}]: "
                f"{constraint.content}"
            )

        return (
            f"Removed {len(constraints)} CONSTRAINTs "
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
                "for constraint action 'add'."
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
        constraint_id = arguments.get(
            "id"
        )
        ids = arguments.get(
            "ids"
        )

        if (
            constraint_id is not None
            and ids is not None
        ):
            raise ValueError(
                "Use either 'id' or 'ids', not both."
            )

        if constraint_id is not None:
            ids = [
                constraint_id
            ]

        if not ids:
            raise ValueError(
                "'id' or 'ids' is required "
                "for constraint action 'remove'."
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
        constraint_id: int,
    ) -> int:
        for index, constraint in enumerate(
            self.__extracts
        ):
            if constraint.id == constraint_id:
                return index

        raise ValueError(
            f"CONSTRAINT [{constraint_id}] does not exist."
        )

    @staticmethod
    def _format_ids(
        ids: list[int],
    ) -> str:
        return (
            "["
            + ", ".join(
                str(constraint_id)
                for constraint_id in ids
            )
            + "]"
        )
