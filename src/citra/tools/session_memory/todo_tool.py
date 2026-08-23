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
    parent_id: int | None = None


class TodoTool(MemoryTool[TodoExtract]):
    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="todo",
            description=(
                "Manage an eager hierarchical TODO list for the current "
                "conversation. Each TODO has a stable ID and may contain "
                "nested sub-steps. Use 'add' to append top-level TODOs or "
                "sub-steps, 'insert' to place newly discovered work at a "
                "specific zero-based position among siblings, 'check' only "
                "after the work and all of its descendants are complete, "
                "and 'remove' only when TODOs are stale, invalid, irrelevant, "
                "or based on incorrect assertions. Keep the list current as "
                "work expands: add or insert newly discovered sub-steps "
                "instead of waiting until the original plan is exhausted."
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
                                "Supported by 'add' only. All added TODOs share "
                                "the same parent_id."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="parent_id",
                        schema=JsonSchema.integer(
                            description=(
                                "Stable TODO ID of the parent when adding or "
                                "inserting a sub-step. Omit for a top-level TODO. "
                                "Sub-steps may themselves have sub-steps."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="index",
                        schema=JsonSchema.integer(
                            description=(
                                "Zero-based sibling position for 'insert'. "
                                "This is not a TODO ID. With parent_id, position "
                                "is among that parent's direct sub-steps. Without "
                                "parent_id, position is among top-level TODOs. "
                                "Existing siblings at and after this position "
                                "move down while keeping their IDs and subtrees."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description=(
                                "Stable TODO ID for a single TODO targeted by "
                                "'check' or 'remove'. This is not a sibling "
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
                                "These are TODO IDs, not sibling positions or "
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

        # Stored in pre-order so the flat representation remains easy to
        # render while each parent's descendants stay contiguous.
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
        indent = "  " * self._depth(
            extract
        )

        return (
            f"{indent}- [{mark}] [ID {extract.id}] "
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

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        action = arguments.get("action", "?")
        parts = [f"action={action}"]

        content = arguments.get("content")
        contents = arguments.get("contents")
        if content is not None:
            parts.append(f"content={self._truncate(str(content))}")
        elif contents is not None:
            parts.append(f"batch={len(contents)}")

        parent_id = arguments.get("parent_id")
        if parent_id is not None:
            parts.append(f"parent={parent_id}")

        index = arguments.get("index")
        if index is not None:
            parts.append(f"index={index}")

        ids = self._get_ids_raw(arguments)
        if ids is not None:
            parts.append(f"ids={ids}")

        return " | ".join(parts)

    @staticmethod
    def _get_ids_raw(
        arguments: dict[str, Any],
    ) -> str | None:
        single = arguments.get("id")
        multiple = arguments.get("ids")

        if single is not None and multiple is not None:
            raise ValueError(
                "Use either 'id' or 'ids', not both."
            )

        if single is not None:
            return f"[{single}]"

        if multiple is not None:
            return (
                "["
                + ", ".join(str(i) for i in multiple)
                + "]"
            )

        return None

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= 80:
            return value
        return value[:80] + "..."

    def _add(
        self,
        arguments: dict[str, Any],
    ) -> str:
        if arguments.get("index") is not None:
            raise ValueError(
                "'index' is only valid for TODO action 'insert'."
            )

        if (
            arguments.get("id") is not None
            or arguments.get("ids") is not None
        ):
            raise ValueError(
                "'id' and 'ids' are only valid for TODO actions "
                "'check' and 'remove'."
            )

        parent_id = arguments.get(
            "parent_id"
        )
        self._validate_parent(
            parent_id
        )

        contents = self._get_contents(
            arguments
        )

        insertion_index = self._append_index(
            parent_id
        )
        added: list[TodoExtract] = []

        for offset, content in enumerate(
            contents
        ):
            todo = TodoExtract(
                id=self.__next_id,
                content=content,
                parent_id=parent_id,
            )

            self.__next_id += 1
            self.__extracts.insert(
                insertion_index + offset,
                todo,
            )
            added.append(
                todo
            )

        reopened = self._reopen_ancestors(
            parent_id
        )

        if len(added) == 1:
            todo = added[0]
            location = self._format_parent_location(
                parent_id
            )
            result = (
                f"Added TODO ID {todo.id}{location}: "
                f"{todo.content}"
            )
        else:
            first_id = added[0].id
            last_id = added[-1].id
            location = self._format_parent_location(
                parent_id
            )
            result = (
                f"Added {len(added)} TODOs with IDs "
                f"{first_id}-{last_id}{location}."
            )

        return self._with_reopened(
            result,
            reopened,
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
        parent_id = arguments.get(
            "parent_id"
        )
        sibling_index = arguments.get(
            "index"
        )

        if (
            arguments.get("id") is not None
            or arguments.get("ids") is not None
        ):
            raise ValueError(
                "'id' and 'ids' are only valid for TODO actions "
                "'check' and 'remove'."
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

        if sibling_index is None:
            raise ValueError(
                "'index' is required for TODO action 'insert'."
            )

        self._validate_parent(
            parent_id
        )

        siblings = self._siblings(
            parent_id
        )

        if not 0 <= sibling_index <= len(siblings):
            raise ValueError(
                "TODO insertion index must be between 0 and "
                f"{len(siblings)} for this parent, got {sibling_index}."
            )

        insertion_index = self._sibling_insertion_index(
            parent_id=parent_id,
            sibling_index=sibling_index,
            siblings=siblings,
        )
        todo = TodoExtract(
            id=self.__next_id,
            content=content,
            parent_id=parent_id,
        )

        self.__next_id += 1
        self.__extracts.insert(
            insertion_index,
            todo,
        )

        reopened = self._reopen_ancestors(
            parent_id
        )
        location = self._format_parent_location(
            parent_id
        )
        result = (
            f"Inserted TODO ID {todo.id} at sibling index "
            f"{sibling_index}{location}: {todo.content}"
        )

        return self._with_reopened(
            result,
            reopened,
        )

    def _check(
        self,
        arguments: dict[str, Any],
    ) -> str:
        self._reject_structure_arguments(
            arguments,
            action="check",
        )
        ids = self._get_ids(
            arguments,
            action="check",
        )

        # Resolve and validate the entire batch before mutating state.
        selected = [
            self.__extracts[
                self._find_index(
                    todo_id
                )
            ]
            for todo_id in ids
        ]
        selected_ids = set(
            ids
        )

        for todo in selected:
            incomplete_descendants = [
                descendant.id
                for descendant in self._descendants(
                    todo.id
                )
                if (
                    not descendant.completed
                    and descendant.id not in selected_ids
                )
            ]

            if incomplete_descendants:
                raise ValueError(
                    f"TODO ID {todo.id} cannot be checked while "
                    "descendants remain incomplete: "
                    f"{self._format_ids(incomplete_descendants)}."
                )

        # A batch may check descendants and their parent together. Apply
        # deepest items first so the resulting state mirrors dependency order.
        selected.sort(
            key=self._depth,
            reverse=True,
        )

        checked: list[int] = []
        already_checked: list[int] = []

        for todo in selected:
            index = self._find_index(
                todo.id
            )
            current = self.__extracts[
                index
            ]

            if current.completed:
                already_checked.append(
                    current.id
                )
                continue

            self.__extracts[index] = replace(
                current,
                completed=True,
            )
            checked.append(
                current.id
            )

        if len(ids) == 1:
            todo_id = ids[0]
            todo = self.__extracts[
                self._find_index(
                    todo_id
                )
            ]

            if todo_id in already_checked:
                return (
                    f"TODO ID {todo.id} is already checked: "
                    f"{todo.content}"
                )

            return (
                f"Checked TODO ID {todo.id}: "
                f"{todo.content}"
            )

        parts: list[str] = []

        if checked:
            parts.append(
                "checked TODO IDs "
                + self._format_ids(
                    sorted(checked)
                )
            )

        if already_checked:
            parts.append(
                "already checked TODO IDs "
                + self._format_ids(
                    sorted(already_checked)
                )
            )

        return "; ".join(
            parts
        )

    def _remove(
        self,
        arguments: dict[str, Any],
    ) -> str:
        self._reject_structure_arguments(
            arguments,
            action="remove",
        )
        ids = self._get_ids(
            arguments,
            action="remove",
        )

        # Resolve every requested ID before mutating. Removing a TODO removes
        # its entire subtree because descendants are scoped to that parent.
        requested = [
            self.__extracts[
                self._find_index(
                    todo_id
                )
            ]
            for todo_id in ids
        ]
        removed_ids: set[int] = set()

        for todo in requested:
            removed_ids.add(
                todo.id
            )
            removed_ids.update(
                descendant.id
                for descendant in self._descendants(
                    todo.id
                )
            )

        removed = [
            todo
            for todo in self.__extracts
            if todo.id in removed_ids
        ]
        self.__extracts = [
            todo
            for todo in self.__extracts
            if todo.id not in removed_ids
        ]

        if len(requested) == 1:
            todo = requested[0]
            descendant_count = len(removed) - 1

            if descendant_count:
                return (
                    f"Removed TODO ID {todo.id} and "
                    f"{descendant_count} descendant"
                    f"{'s' if descendant_count != 1 else ''}: "
                    f"{todo.content}"
                )

            return (
                f"Removed TODO ID {todo.id}: "
                f"{todo.content}"
            )

        return (
            f"Removed {len(removed)} TODOs across requested IDs "
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

    def _validate_parent(
        self,
        parent_id: int | None,
    ) -> None:
        if parent_id is None:
            return

        self._find_index(
            parent_id
        )

    def _reject_structure_arguments(
        self,
        arguments: dict[str, Any],
        *,
        action: str,
    ) -> None:
        invalid = [
            name
            for name in (
                "content",
                "contents",
                "parent_id",
                "index",
            )
            if arguments.get(name) is not None
        ]

        if invalid:
            names = ", ".join(
                f"'{name}'"
                for name in invalid
            )
            raise ValueError(
                f"{names} are invalid for TODO action '{action}'."
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

    def _siblings(
        self,
        parent_id: int | None,
    ) -> list[TodoExtract]:
        return [
            todo
            for todo in self.__extracts
            if todo.parent_id == parent_id
        ]

    def _append_index(
        self,
        parent_id: int | None,
    ) -> int:
        if parent_id is None:
            return len(
                self.__extracts
            )

        return self._subtree_end_index(
            parent_id
        )

    def _sibling_insertion_index(
        self,
        *,
        parent_id: int | None,
        sibling_index: int,
        siblings: list[TodoExtract],
    ) -> int:
        if sibling_index < len(siblings):
            return self._find_index(
                siblings[sibling_index].id
            )

        return self._append_index(
            parent_id
        )

    def _subtree_end_index(
        self,
        todo_id: int,
    ) -> int:
        start = self._find_index(
            todo_id
        )
        root_depth = self._depth(
            self.__extracts[start]
        )

        index = start + 1

        while index < len(self.__extracts):
            if self._depth(self.__extracts[index]) <= root_depth:
                break

            index += 1

        return index

    def _descendants(
        self,
        todo_id: int,
    ) -> list[TodoExtract]:
        start = self._find_index(
            todo_id
        )
        end = self._subtree_end_index(
            todo_id
        )

        return list(
            self.__extracts[start + 1:end]
        )

    def _depth(
        self,
        todo: TodoExtract,
    ) -> int:
        depth = 0
        parent_id = todo.parent_id

        while parent_id is not None:
            parent = self.__extracts[
                self._find_index(
                    parent_id
                )
            ]
            depth += 1
            parent_id = parent.parent_id

        return depth

    def _reopen_ancestors(
        self,
        parent_id: int | None,
    ) -> list[int]:
        reopened: list[int] = []
        current_id = parent_id

        while current_id is not None:
            index = self._find_index(
                current_id
            )
            todo = self.__extracts[
                index
            ]

            if todo.completed:
                self.__extracts[index] = replace(
                    todo,
                    completed=False,
                )
                reopened.append(
                    todo.id
                )

            current_id = todo.parent_id

        return reopened

    @staticmethod
    def _format_parent_location(
        parent_id: int | None,
    ) -> str:
        if parent_id is None:
            return ""

        return f" under parent ID {parent_id}"

    @staticmethod
    def _with_reopened(
        result: str,
        reopened: list[int],
    ) -> str:
        if not reopened:
            return result

        return (
            result
            + "; reopened completed ancestor TODO IDs "
            + TodoTool._format_ids(
                reopened
            )
            + " because they now contain unfinished work"
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
