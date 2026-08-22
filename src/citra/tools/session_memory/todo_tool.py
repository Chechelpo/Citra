from dataclasses import dataclass, replace
from typing import Any, override

from citra.context import ExecutionContext
from citra.utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ...agent import AgentSession
from .memory_tool import MemoryTool


@dataclass(frozen=True)
class TodoExtract:
    id: int
    content: str
    completed: bool = False


class TodoTool(MemoryTool[TodoExtract]):
    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="todo",
            description=(
                "Manage TODOs for the current conversation. "
                "Each TODO has a stable ID used by 'check' and 'remove'. "
                "Operations may target one TODO or a batch. "
                "Use 'add' to append required work, 'insert' to place newly "
                "discovered work at a specific zero-based list position, "
                "'check' only after the work has actually been completed, "
                "and 'remove' only when TODOs are stale, invalid, irrelevant, "
                "or based on incorrect assertions."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="TODO operation to perform.",
                            enum=(
                                "add",
                                "insert",
                                "check",
                                "remove",
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description=(
                                "Single TODO description for 'add' or 'insert'. "
                                "Use 'contents' to add multiple TODOs."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="contents",
                        schema=JsonSchema.array(
                            items=JsonSchema.string(),
                            description=(
                                "TODO descriptions to add as a batch. "
                                "Supported by 'add' only."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="index",
                        schema=JsonSchema.integer(
                            description=(
                                "Zero-based list position for 'insert'. "
                                "This is not a TODO ID. Position 0 inserts "
                                "before the first TODO; a position equal to "
                                "the current TODO count inserts at the end. "
                                "Existing TODOs at and after this position "
                                "move down while keeping their IDs."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description=(
                                "Stable TODO ID for a single TODO targeted by "
                                "'check' or 'remove'. This is not a list "
                                "position or step number. Use 'ids' to target "
                                "multiple TODO IDs."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description=(
                                "Stable TODO IDs to check or remove as a batch. "
                                "These are TODO IDs, not list positions or "
                                "step numbers."
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
            session=session,
            definition=self.DEFINITION,
        )

        self.__extracts: list[TodoExtract] = []
        self.__next_id = 1

    @property
    @override
    def heading(self) -> str:
        return "TODOs"

    @override
    def get_extracts(
        self,
    ) -> list[TodoExtract]:
        return list(
            self.__extracts
        )

    @override
    def format_extract(
        self,
        extract: TodoExtract,
    ) -> str:
        mark = (
            "x"
            if extract.completed
            else " "
        )

        return (
            f"- [{mark}] [ID {extract.id}] "
            f"{extract.content}"
        )

    @override
    def should_offer_documentation(
        self,
    ) -> bool:
        return False

    def has_outstanding_todos(
        self,
    ) -> bool:
        return any(
            not todo.completed
            for todo in self.__extracts
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

        if action == "insert":
            return self._insert(
                arguments
            )

        if action == "check":
            return self._check(
                arguments
            )

        if action == "remove":
            return self._remove(
                arguments
            )

        raise ValueError(
            f"Unsupported TODO action: {action}"
        )

    def _add(
        self,
        arguments: dict[str, Any],
    ) -> str:
        if arguments.get("index") is not None:
            raise ValueError(
                "'index' is only valid for TODO action 'insert'."
            )

        contents = self._get_contents(
            arguments
        )

        added: list[TodoExtract] = []

        for content in contents:
            todo = TodoExtract(
                id=self.__next_id,
                content=content,
            )

            self.__next_id += 1
            self.__extracts.append(
                todo
            )
            added.append(
                todo
            )

        if len(added) == 1:
            todo = added[0]

            return (
                f"Added TODO ID {todo.id}: "
                f"{todo.content}"
            )

        first_id = added[0].id
        last_id = added[-1].id

        return (
            f"Added {len(added)} TODOs "
            f"with IDs {first_id}-{last_id}."
        )

    def _insert(
        self,
        arguments: dict[str, Any],
    ) -> str:
        content = arguments.get(
            "content"
        )
        contents = arguments.get(
            "contents"
        )
        index = arguments.get(
            "index"
        )

        if contents is not None:
            raise ValueError(
                "TODO action 'insert' accepts a single 'content', "
                "not 'contents'."
            )

        if content is None:
            raise ValueError(
                "'content' is required for TODO action 'insert'."
            )

        content = content.strip()

        if not content:
            raise ValueError(
                "TODO content cannot be empty."
            )

        if index is None:
            raise ValueError(
                "'index' is required for TODO action 'insert'."
            )

        if not 0 <= index <= len(self.__extracts):
            raise ValueError(
                f"TODO insertion index must be between 0 and "
                f"{len(self.__extracts)}, got {index}."
            )

        todo = TodoExtract(
            id=self.__next_id,
            content=content,
        )

        self.__next_id += 1

        self.__extracts.insert(
            index,
            todo,
        )

        return (
            f"Inserted TODO ID {todo.id} at list index {index}: "
            f"{todo.content}"
        )

    def _check(
        self,
        arguments: dict[str, Any],
    ) -> str:
        ids = self._get_ids(
            arguments,
            action="check",
        )

        checked: list[int] = []
        already_checked: list[int] = []

        for todo_id in ids:
            index = self._find_index(
                todo_id
            )

            todo = self.__extracts[
                index
            ]

            if todo.completed:
                already_checked.append(
                    todo.id
                )
                continue

            self.__extracts[index] = replace(
                todo,
                completed=True,
            )

            checked.append(
                todo.id
            )

        if len(ids) == 1:
            todo_id = ids[0]

            if todo_id in already_checked:
                todo = self.__extracts[
                    self._find_index(
                        todo_id
                    )
                ]

                return (
                    f"TODO ID {todo.id} is already checked: "
                    f"{todo.content}"
                )

            todo = self.__extracts[
                self._find_index(
                    todo_id
                )
            ]

            return (
                f"Checked TODO ID {todo.id}: "
                f"{todo.content}"
            )

        parts: list[str] = []

        if checked:
            parts.append(
                "checked TODO IDs "
                + self._format_ids(
                    checked
                )
            )

        if already_checked:
            parts.append(
                "already checked TODO IDs "
                + self._format_ids(
                    already_checked
                )
            )

        return "; ".join(
            parts
        )

    def _remove(
        self,
        arguments: dict[str, Any],
    ) -> str:
        ids = self._get_ids(
            arguments,
            action="remove",
        )

        # Resolve everything first so a bad ID cannot cause a
        # partially-applied batch.
        todos = [
            self.__extracts[
                self._find_index(
                    todo_id
                )
            ]
            for todo_id in ids
        ]

        ids_set = set(
            ids
        )

        self.__extracts = [
            todo
            for todo in self.__extracts
            if todo.id not in ids_set
        ]

        if len(todos) == 1:
            todo = todos[0]

            return (
                f"Removed TODO ID {todo.id}: "
                f"{todo.content}"
            )

        return (
            f"Removed {len(todos)} TODOs with IDs "
            f"{self._format_ids(ids)}."
        )

    def _get_contents(
        self,
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
                "for TODO action 'add'."
            )

        normalized: list[str] = []

        for item in contents:
            item = item.strip()

            if not item:
                raise ValueError(
                    "TODO content cannot be empty."
                )

            normalized.append(
                item
            )

        return normalized

    def _get_ids(
        self,
        arguments: dict[str, Any],
        *,
        action: str,
    ) -> list[int]:
        todo_id = arguments.get(
            "id"
        )
        ids = arguments.get(
            "ids"
        )

        if (
            todo_id is not None
            and ids is not None
        ):
            raise ValueError(
                "Use either 'id' or 'ids', not both."
            )

        if todo_id is not None:
            ids = [
                todo_id
            ]

        if not ids:
            raise ValueError(
                f"'id' or 'ids' is required for "
                f"TODO action '{action}'."
            )

        if len(ids) != len(set(ids)):
            raise ValueError(
                "'ids' cannot contain duplicate TODO IDs."
            )

        return list(
            ids
        )

    def _find_index(
        self,
        todo_id: int,
    ) -> int:
        for index, todo in enumerate(
            self.__extracts
        ):
            if todo.id == todo_id:
                return index

        raise ValueError(
            f"TODO ID {todo_id} does not exist."
        )

    @staticmethod
    def _format_ids(
        ids: list[int],
    ) -> str:
        return (
            "["
            + ", ".join(
                str(todo_id)
                for todo_id in ids
            )
            + "]"
        )