from __future__ import annotations

from dataclasses import dataclass, replace
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
class TodoExtract:
    id: int
    content: str
    working_state_id: int | None = None
    completed: bool = False
    parent_id: int | None = None


class TodoTool(MemoryTool[TodoExtract]):
    """Manage hierarchical TODOs with optional working-state provenance."""

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="todo",
            description=(
                "Manage the conversation's eager hierarchical TODO list. Use "
                "'add' or 'insert' for directly established work, or 'promote' "
                "when an active working state produced the TODO and provenance is "
                "useful. Use 'check' only after the work and all descendants are "
                "complete, and 'remove' only when work is stale, invalid, "
                "irrelevant, or based on an incorrect premise."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="TODO operation.",
                            enum=("add", "insert", "promote", "check", "remove"),
                        ),
                    ),
                    JsonProperty(
                        name="working_state_id",
                        schema=JsonSchema.integer(
                            description="Single working-state ID to promote.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="working_state_ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="Working-state IDs to promote as sibling TODOs.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description=(
                                "TODO text for add/insert, or optional polished text "
                                "for a single promotion."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="contents",
                        schema=JsonSchema.array(
                            items=JsonSchema.string(),
                            description=(
                                "TODO descriptions to add directly as a batch. All "
                                "entries share parent_id. Valid for add only."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="parent_id",
                        schema=JsonSchema.integer(
                            description=(
                                "Parent TODO ID for a sub-step. Omit for top-level "
                                "TODOs."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="index",
                        schema=JsonSchema.integer(
                            description=(
                                "Zero-based sibling insertion position for insert, "
                                "or optional position for a single promotion."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description="Single TODO ID for check/remove.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="TODO IDs to check/remove as a batch.",
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
        # Pre-order storage keeps each subtree contiguous while rendering flatly.
        self.__extracts: list[TodoExtract] = []
        self.__next_id = 1

    @property
    @override
    def heading(self) -> str:
        return "TODOs"

    @override
    def get_extracts(self) -> list[TodoExtract]:
        return list(self.__extracts)

    @override
    def format_extract(self, extract: TodoExtract) -> str:
        mark = "x" if extract.completed else " "
        indent = "  " * self._depth(extract)
        text = f"{indent}- [{mark}] [ID {extract.id}] {extract.content}"
        if extract.working_state_id is not None:
            text += f" (from W{extract.working_state_id})"
        return text

    @override
    def should_offer_documentation(self) -> bool:
        return False

    def has_outstanding_todos(self) -> bool:
        return any(not todo.completed for todo in self.__extracts)

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        action = arguments["action"]
        if action == "add":
            return self._add(arguments)
        if action == "insert":
            return self._insert(arguments)
        if action == "promote":
            return self._promote(arguments)
        if action == "check":
            return self._check(arguments)
        if action == "remove":
            return self._remove(arguments)
        raise ValueError(f"Unsupported TODO action: {action}")

    def _add(self, arguments: dict[str, Any]) -> str:
        self._reject_fields(
            arguments,
            ("working_state_id", "working_state_ids", "index", "id", "ids"),
            action="add",
        )
        parent_id = arguments.get("parent_id")
        self._validate_parent(parent_id)
        contents = self._get_contents(arguments)
        insertion_index = self._append_index(parent_id)

        added: list[TodoExtract] = []
        for offset, content in enumerate(contents):
            todo = self._new_todo(
                content,
                parent_id=parent_id,
                working_state_id=None,
            )
            self.__extracts.insert(insertion_index + offset, todo)
            added.append(todo)

        reopened = self._reopen_ancestors(parent_id)
        location = self._format_parent_location(parent_id)
        if len(added) == 1:
            todo = added[0]
            result = f"Added TODO ID {todo.id}{location}: {todo.content}"
        else:
            result = (
                f"Added {len(added)} TODOs with IDs "
                f"{added[0].id}-{added[-1].id}{location}."
            )
        return self._with_reopened(result, reopened)

    def _insert(self, arguments: dict[str, Any]) -> str:
        self._reject_fields(
            arguments,
            ("working_state_id", "working_state_ids", "contents", "id", "ids"),
            action="insert",
        )
        content_raw = arguments.get("content")
        if content_raw is None:
            raise ValueError("'content' is required for TODO action 'insert'.")
        content = str(content_raw).strip()
        if not content:
            raise ValueError("TODO content cannot be empty.")
        sibling_index = arguments.get("index")
        if sibling_index is None:
            raise ValueError("'index' is required for TODO action 'insert'.")

        parent_id = arguments.get("parent_id")
        self._validate_parent(parent_id)
        siblings = self._siblings(parent_id)
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
        todo = self._new_todo(
            content,
            parent_id=parent_id,
            working_state_id=None,
        )
        self.__extracts.insert(insertion_index, todo)
        reopened = self._reopen_ancestors(parent_id)
        location = self._format_parent_location(parent_id)
        result = (
            f"Inserted TODO ID {todo.id} at sibling index {sibling_index}"
            f"{location}: {todo.content}"
        )
        return self._with_reopened(result, reopened)

    def _promote(self, arguments: dict[str, Any]) -> str:
        self._reject_fields(arguments, ("contents", "id", "ids"), action="promote")
        parent_id = arguments.get("parent_id")
        self._validate_parent(parent_id)
        working_ids = self._working_ids(arguments)
        content_override = arguments.get("content")
        sibling_index = arguments.get("index")

        if len(working_ids) != 1 and content_override is not None:
            raise ValueError("'content' is only valid for a single promotion.")
        if len(working_ids) != 1 and sibling_index is not None:
            raise ValueError("'index' is only valid for a single promotion.")

        prepared: list[tuple[int, str]] = []
        for working_id in working_ids:
            working = self.require_working_state(working_id)
            content = (
                str(content_override).strip()
                if content_override is not None
                else working.content
            )
            if not content:
                raise ValueError("TODO content cannot be empty.")
            prepared.append((working_id, content))

        if sibling_index is None:
            insertion_index = self._append_index(parent_id)
        else:
            siblings = self._siblings(parent_id)
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

        added: list[TodoExtract] = []
        for offset, (working_id, content) in enumerate(prepared):
            todo = self._new_todo(
                content,
                parent_id=parent_id,
                working_state_id=working_id,
            )
            self.__extracts.insert(insertion_index + offset, todo)
            self.register_promotion(
                working_id,
                kind="todo",
                memory_id=todo.id,
            )
            added.append(todo)

        reopened = self._reopen_ancestors(parent_id)
        location = self._format_parent_location(parent_id)
        if len(added) == 1:
            todo = added[0]
            position = (
                f" at sibling index {sibling_index}"
                if sibling_index is not None
                else ""
            )
            result = (
                f"Promoted working state [W{todo.working_state_id}] to "
                f"TODO ID {todo.id}{position}{location}: {todo.content}"
            )
        else:
            result = (
                f"Promoted {len(added)} working states to TODO IDs "
                f"{added[0].id}-{added[-1].id}{location}."
            )
        return self._with_reopened(result, reopened)

    def _check(self, arguments: dict[str, Any]) -> str:
        self._reject_fields(
            arguments,
            (
                "working_state_id",
                "working_state_ids",
                "content",
                "contents",
                "parent_id",
                "index",
            ),
            action="check",
        )
        ids = self._get_ids(arguments, action="check")
        selected = [self.__extracts[self._find_index(todo_id)] for todo_id in ids]
        selected_ids = set(ids)

        for todo in selected:
            incomplete_descendants = [
                descendant.id
                for descendant in self._descendants(todo.id)
                if not descendant.completed and descendant.id not in selected_ids
            ]
            if incomplete_descendants:
                raise ValueError(
                    f"TODO ID {todo.id} cannot be checked while descendants "
                    f"remain incomplete: {self._format_ids(incomplete_descendants)}."
                )

        selected.sort(key=self._depth, reverse=True)
        checked: list[int] = []
        already_checked: list[int] = []
        for todo in selected:
            index = self._find_index(todo.id)
            current = self.__extracts[index]
            if current.completed:
                already_checked.append(current.id)
                continue
            self.__extracts[index] = replace(current, completed=True)
            checked.append(current.id)

        if len(ids) == 1:
            todo = self.__extracts[self._find_index(ids[0])]
            if todo.id in already_checked:
                return f"TODO ID {todo.id} is already checked: {todo.content}"
            return f"Checked TODO ID {todo.id}: {todo.content}"

        parts: list[str] = []
        if checked:
            parts.append("checked TODO IDs " + self._format_ids(sorted(checked)))
        if already_checked:
            parts.append(
                "already checked TODO IDs "
                + self._format_ids(sorted(already_checked))
            )
        return "; ".join(parts)

    def _remove(self, arguments: dict[str, Any]) -> str:
        self._reject_fields(
            arguments,
            (
                "working_state_id",
                "working_state_ids",
                "content",
                "contents",
                "parent_id",
                "index",
            ),
            action="remove",
        )
        ids = self._get_ids(arguments, action="remove")
        requested = [self.__extracts[self._find_index(todo_id)] for todo_id in ids]

        removed_ids: set[int] = set()
        for todo in requested:
            removed_ids.add(todo.id)
            removed_ids.update(desc.id for desc in self._descendants(todo.id))

        removed = [todo for todo in self.__extracts if todo.id in removed_ids]
        self.__extracts = [
            todo for todo in self.__extracts if todo.id not in removed_ids
        ]
        for todo in removed:
            if todo.working_state_id is not None:
                self.unregister_promotion(
                    todo.working_state_id,
                    kind="todo",
                    memory_id=todo.id,
                )

        if len(requested) == 1:
            todo = requested[0]
            descendant_count = len(removed) - 1
            if descendant_count:
                return (
                    f"Removed TODO ID {todo.id} and {descendant_count} "
                    f"descendant{'s' if descendant_count != 1 else ''}: "
                    f"{todo.content}"
                )
            return f"Removed TODO ID {todo.id}: {todo.content}"
        return (
            f"Removed {len(removed)} TODOs across requested IDs "
            f"{self._format_ids(ids)}."
        )

    def _new_todo(
        self,
        content: str,
        *,
        parent_id: int | None,
        working_state_id: int | None,
    ) -> TodoExtract:
        todo = TodoExtract(
            id=self.__next_id,
            content=content,
            working_state_id=working_state_id,
            parent_id=parent_id,
        )
        self.__next_id += 1
        return todo

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        action = arguments.get("action", "?")
        parts = [f"action={action}"]
        working = self._working_ids_summary(arguments)
        if working:
            parts.append(f"working={working}")
        if arguments.get("content") is not None:
            parts.append(f"content={self._truncate(str(arguments['content']))}")
        elif arguments.get("contents") is not None:
            parts.append(f"batch={len(arguments['contents'])}")
        if arguments.get("parent_id") is not None:
            parts.append(f"parent={arguments['parent_id']}")
        if arguments.get("index") is not None:
            parts.append(f"index={arguments['index']}")
        ids = self._ids_summary(arguments)
        if ids:
            parts.append(f"ids={ids}")
        return " | ".join(parts)

    @staticmethod
    def _get_contents(arguments: dict[str, Any]) -> list[str]:
        content = arguments.get("content")
        contents = arguments.get("contents")
        if content is not None and contents is not None:
            raise ValueError("Use either 'content' or 'contents', not both.")
        raw = [content] if content is not None else contents
        if not raw:
            raise ValueError("'content' or 'contents' is required for TODO action 'add'.")
        normalized: list[str] = []
        for index, item in enumerate(raw):
            text = str(item).strip()
            if not text:
                raise ValueError(f"contents[{index}] cannot be empty.")
            normalized.append(text)
        return normalized

    @staticmethod
    def _working_ids(arguments: dict[str, Any]) -> list[int]:
        single = arguments.get("working_state_id")
        multiple = arguments.get("working_state_ids")
        if single is not None and multiple is not None:
            raise ValueError("Use either 'working_state_id' or 'working_state_ids', not both.")
        ids = [single] if single is not None else multiple
        if not ids:
            raise ValueError("'working_state_id' or 'working_state_ids' is required for promote.")
        if len(ids) != len(set(ids)):
            raise ValueError("Working-state IDs cannot contain duplicates.")
        return list(ids)

    @staticmethod
    def _get_ids(arguments: dict[str, Any], *, action: str) -> list[int]:
        single = arguments.get("id")
        multiple = arguments.get("ids")
        if single is not None and multiple is not None:
            raise ValueError("Use either 'id' or 'ids', not both.")
        ids = [single] if single is not None else multiple
        if not ids:
            raise ValueError(f"'id' or 'ids' is required for TODO action '{action}'.")
        if len(ids) != len(set(ids)):
            raise ValueError("TODO IDs cannot contain duplicates.")
        return list(ids)

    @staticmethod
    def _reject_fields(
        arguments: dict[str, Any],
        names: tuple[str, ...],
        *,
        action: str,
    ) -> None:
        invalid = [name for name in names if arguments.get(name) is not None]
        if invalid:
            raise ValueError(
                ", ".join(f"'{name}'" for name in invalid)
                + f" are invalid for TODO action '{action}'."
            )

    def _validate_parent(self, parent_id: int | None) -> None:
        if parent_id is not None:
            self._find_index(parent_id)

    def _find_index(self, todo_id: int) -> int:
        for index, todo in enumerate(self.__extracts):
            if todo.id == todo_id:
                return index
        raise ValueError(f"TODO ID {todo_id} does not exist.")

    def _siblings(self, parent_id: int | None) -> list[TodoExtract]:
        return [todo for todo in self.__extracts if todo.parent_id == parent_id]

    def _append_index(self, parent_id: int | None) -> int:
        if parent_id is None:
            return len(self.__extracts)
        return self._subtree_end_index(parent_id)

    def _sibling_insertion_index(
        self,
        *,
        parent_id: int | None,
        sibling_index: int,
        siblings: list[TodoExtract],
    ) -> int:
        if sibling_index < len(siblings):
            return self._find_index(siblings[sibling_index].id)
        return self._append_index(parent_id)

    def _subtree_end_index(self, todo_id: int) -> int:
        start = self._find_index(todo_id)
        root_depth = self._depth(self.__extracts[start])
        index = start + 1
        while index < len(self.__extracts):
            if self._depth(self.__extracts[index]) <= root_depth:
                break
            index += 1
        return index

    def _descendants(self, todo_id: int) -> list[TodoExtract]:
        start = self._find_index(todo_id)
        end = self._subtree_end_index(todo_id)
        return list(self.__extracts[start + 1:end])

    def _depth(self, todo: TodoExtract) -> int:
        depth = 0
        parent_id = todo.parent_id
        while parent_id is not None:
            parent = self.__extracts[self._find_index(parent_id)]
            depth += 1
            parent_id = parent.parent_id
        return depth

    def _reopen_ancestors(self, parent_id: int | None) -> list[int]:
        reopened: list[int] = []
        current_id = parent_id
        while current_id is not None:
            index = self._find_index(current_id)
            todo = self.__extracts[index]
            if todo.completed:
                self.__extracts[index] = replace(todo, completed=False)
                reopened.append(todo.id)
            current_id = todo.parent_id
        return reopened

    @staticmethod
    def _format_parent_location(parent_id: int | None) -> str:
        return "" if parent_id is None else f" under parent ID {parent_id}"

    @staticmethod
    def _with_reopened(result: str, reopened: list[int]) -> str:
        if not reopened:
            return result
        return (
            result
            + "; reopened completed ancestor TODO IDs "
            + TodoTool._format_ids(reopened)
            + " because they now contain unfinished work"
        )

    @staticmethod
    def _working_ids_summary(arguments: dict[str, Any]) -> str | None:
        single = arguments.get("working_state_id")
        multiple = arguments.get("working_state_ids")
        if single is not None:
            return f"[W{single}]"
        if multiple is not None:
            return "[" + ", ".join(f"W{x}" for x in multiple) + "]"
        return None

    @staticmethod
    def _ids_summary(arguments: dict[str, Any]) -> str | None:
        single = arguments.get("id")
        multiple = arguments.get("ids")
        if single is not None:
            return f"[{single}]"
        if multiple is not None:
            return TodoTool._format_ids(list(multiple))
        return None

    @staticmethod
    def _format_ids(ids: list[int]) -> str:
        return "[" + ", ".join(str(todo_id) for todo_id in ids) + "]"

    @staticmethod
    def _truncate(value: str) -> str:
        return value if len(value) <= 80 else value[:80] + "..."
