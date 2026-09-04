"""Implementation-change memory for cross-role handoff."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, override

from citra.utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)

from ..capabilities import ToolCapabilities
from ..tool import ToolDefinition
from .acceptance_criteria_tool import AcceptanceCriteriaTool
from .memory_tool import MemoryTool
from .requirement_tool import RequirementTool
from .todo_tool import TodoTool

if TYPE_CHECKING:
    from citra.agent import AgentSession
    from citra.context import ExecutionContext


@dataclass(frozen=True)
class ChangeExtract:
    """Describe one versioned implementation change and its task coverage."""

    id: int
    revision: int
    summary: str
    paths: tuple[str, ...]
    todo_ids: tuple[int, ...] = ()
    requirement_ids: tuple[int, ...] = ()
    acceptance_criterion_ids: tuple[int, ...] = ()
    notes: str | None = None


class ChangeTool(MemoryTool[ChangeExtract]):
    """Record versioned implemented changes for independent test and review."""

    TOOL_ID = "change"
    CAPABILITIES = ToolCapabilities(actions=("record", "update", "remove"))

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="change",
            description=(
                "Record coherent implementation changes for later test and "
                "review roles. Include exact repository paths and optional "
                "links to TODO, requirement, and acceptance-criterion IDs. "
                "Use notes only for behavior, compatibility, or verification "
                "details another role must know."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Change-memory operation.",
                            enum=("record", "update", "remove"),
                        ),
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description="Change ID for update or remove."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="Change IDs for batch removal.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="summary",
                        schema=JsonSchema.string(
                            description="Implemented behavior or refactor."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="paths",
                        schema=JsonSchema.array(
                            items=JsonSchema.string(),
                            description="Exact changed repository paths.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="todo_ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="TODO IDs implemented by this change.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="requirement_ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="Requirement IDs addressed by this change.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="acceptance_criterion_ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description=(
                                "Acceptance-criterion IDs addressed by this change."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="notes",
                        schema=JsonSchema.string(
                            description=(
                                "Compatibility, behavior, or verification detail "
                                "needed by later roles."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        )
    )

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        """Return the single model-facing change definition."""
        del context
        return (ToolDefinition(definition=cls.DEFINITION),)

    def __init__(
        self,
        context: ExecutionContext,
        session: AgentSession,
    ) -> None:
        """Initialize empty implementation-change memory."""
        super().__init__(context=context, session=session)
        self._extracts: list[ChangeExtract] = []
        self._next_id = 1

    @property
    @override
    def heading(self) -> str:
        """Return the memory-section heading."""
        return "Implemented Changes"

    @override
    def get_extracts(self) -> list[ChangeExtract]:
        """Return a defensive copy of implementation changes."""
        return list(self._extracts)

    @override
    def format_extract(self, extract: ChangeExtract) -> str:
        """Render one change with paths and outcome traceability."""
        lines = [f"- [CH{extract.id}@r{extract.revision}] {extract.summary}"]
        lines.append(f"  - paths: {', '.join(extract.paths)}")
        coverage = self._format_coverage(extract)
        if coverage:
            lines.append(f"  - covers: {coverage}")
        if extract.notes:
            lines.append(f"  - notes: {extract.notes}")
        return "\n".join(lines)

    @override
    def should_offer_documentation(self) -> bool:
        """Keep task-specific change records out of durable documentation."""
        return False

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        """Dispatch one validated change-memory action."""
        action = arguments["action"]
        if action == "record":
            return self._record(arguments)
        if action == "update":
            return self._update(arguments)
        if action == "remove":
            return self._remove(arguments)
        raise ValueError(f"Unsupported change action: {action}")

    def _record(self, arguments: dict[str, Any]) -> str:
        """Record one coherent implementation result."""
        self._reject(arguments, ("id", "ids"), action="record")
        summary = self._required_text(arguments.get("summary"), field="summary")
        paths = self._string_tuple(arguments.get("paths"), field="paths", required=True)
        todo_ids, requirement_ids, criterion_ids = self._coverage(arguments)
        item = ChangeExtract(
            id=self._next_id,
            revision=1,
            summary=summary,
            paths=paths,
            todo_ids=todo_ids,
            requirement_ids=requirement_ids,
            acceptance_criterion_ids=criterion_ids,
            notes=self._optional_text(arguments.get("notes"), field="notes"),
        )
        self._next_id += 1
        self._extracts.append(item)
        return f"Recorded CHANGE [CH{item.id}] across {len(item.paths)} path(s)."

    def _update(self, arguments: dict[str, Any]) -> str:
        """Update one change record after the implementation evolves."""
        self._reject(arguments, ("ids",), action="update")
        change_id = arguments.get("id")
        if change_id is None:
            raise ValueError("'id' is required for update.")
        mutable = (
            "summary",
            "paths",
            "todo_ids",
            "requirement_ids",
            "acceptance_criterion_ids",
            "notes",
        )
        if not any(arguments.get(field) is not None for field in mutable):
            raise ValueError("Change update requires at least one changed field.")

        index = self._find_index(change_id)
        current = self._extracts[index]
        todo_ids, requirement_ids, criterion_ids = self._coverage(
            arguments,
            current=current,
        )
        updated = replace(
            current,
            revision=current.revision + 1,
            summary=(
                current.summary
                if arguments.get("summary") is None
                else self._required_text(arguments["summary"], field="summary")
            ),
            paths=(
                current.paths
                if arguments.get("paths") is None
                else self._string_tuple(
                    arguments["paths"],
                    field="paths",
                    required=True,
                )
            ),
            todo_ids=todo_ids,
            requirement_ids=requirement_ids,
            acceptance_criterion_ids=criterion_ids,
            notes=(
                current.notes
                if arguments.get("notes") is None
                else self._optional_text(arguments["notes"], field="notes")
            ),
        )
        self._extracts[index] = updated
        return f"Updated CHANGE [CH{updated.id}]."

    def _remove(self, arguments: dict[str, Any]) -> str:
        """Remove records that no longer describe the implementation."""
        self._reject(
            arguments,
            (
                "summary",
                "paths",
                "todo_ids",
                "requirement_ids",
                "acceptance_criterion_ids",
                "notes",
            ),
            action="remove",
        )
        ids = self._selected_ids(arguments)
        for change_id in ids:
            self._find_index(change_id)
        selected = set(ids)
        self._extracts = [item for item in self._extracts if item.id not in selected]
        return "Removed changes " + self._format_ids(ids) + "."

    def _coverage(
        self,
        arguments: dict[str, Any],
        *,
        current: ChangeExtract | None = None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Normalize and validate TODO, requirement, and criterion links."""
        todo_ids = self._linked_ids(arguments, "todo_ids", current)
        requirement_ids = self._linked_ids(arguments, "requirement_ids", current)
        criterion_ids = self._linked_ids(
            arguments,
            "acceptance_criterion_ids",
            current,
        )
        self.require_memory_ids(TodoTool, todo_ids, field="todo_ids")
        self.require_memory_ids(
            RequirementTool,
            requirement_ids,
            field="requirement_ids",
        )
        self.require_memory_ids(
            AcceptanceCriteriaTool,
            criterion_ids,
            field="acceptance_criterion_ids",
        )
        return todo_ids, requirement_ids, criterion_ids

    def _linked_ids(
        self,
        arguments: dict[str, Any],
        field: str,
        current: ChangeExtract | None,
    ) -> tuple[int, ...]:
        """Preserve omitted links on update and normalize supplied links."""
        if current is not None and arguments.get(field) is None:
            if field == "todo_ids":
                return current.todo_ids
            if field == "requirement_ids":
                return current.requirement_ids
            return current.acceptance_criterion_ids
        return self.normalize_reference_ids(arguments.get(field), field=field)

    def _find_index(self, change_id: int) -> int:
        """Return the list index for an existing change ID."""
        for index, item in enumerate(self._extracts):
            if item.id == change_id:
                return index
        raise ValueError(f"CHANGE [CH{change_id}] does not exist.")

    @staticmethod
    def _required_text(value: object, *, field: str) -> str:
        """Normalize one required non-empty text field."""
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"'{field}' is required and cannot be empty.")
        return normalized

    @staticmethod
    def _optional_text(value: object, *, field: str) -> str | None:
        """Normalize an optional text field while rejecting empty strings."""
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"'{field}' cannot be empty.")
        return normalized

    @staticmethod
    def _string_tuple(
        values: object,
        *,
        field: str,
        required: bool = False,
    ) -> tuple[str, ...]:
        """Normalize a string array and reject empty or duplicate entries."""
        if values is None:
            if required:
                raise ValueError(f"'{field}' is required.")
            return ()
        if not isinstance(values, list):
            raise ValueError(f"'{field}' must be an array of strings.")
        normalized = tuple(str(value).strip() for value in values)
        if required and not normalized:
            raise ValueError(f"'{field}' cannot be empty.")
        if any(not value for value in normalized):
            raise ValueError(f"'{field}' cannot contain empty values.")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"'{field}' cannot contain duplicate values.")
        return normalized

    @staticmethod
    def _selected_ids(arguments: dict[str, Any]) -> tuple[int, ...]:
        """Normalize one-or-many change IDs for removal."""
        single = arguments.get("id")
        multiple = arguments.get("ids")
        if single is not None and multiple is not None:
            raise ValueError("Use either 'id' or 'ids', not both.")
        values = [single] if single is not None else multiple
        if not values:
            raise ValueError("'id' or 'ids' is required for remove.")
        if len(values) != len(set(values)):
            raise ValueError("Change IDs cannot contain duplicates.")
        return tuple(values)

    @staticmethod
    def _reject(
        arguments: dict[str, Any],
        fields: tuple[str, ...],
        *,
        action: str,
    ) -> None:
        """Reject fields that are semantically invalid for an action."""
        invalid = tuple(field for field in fields if arguments.get(field) is not None)
        if invalid:
            rendered = ", ".join(f"'{field}'" for field in invalid)
            raise ValueError(f"{rendered} are invalid for change action {action!r}.")

    @staticmethod
    def _format_ids(ids: tuple[int, ...]) -> str:
        """Format change IDs for concise results."""
        return "[" + ", ".join(f"CH{item}" for item in ids) + "]"

    @staticmethod
    def _format_coverage(extract: ChangeExtract) -> str:
        """Format typed TODO, requirement, and criterion references."""
        refs = [f"TODO {item}" for item in extract.todo_ids]
        refs.extend(f"R{item}" for item in extract.requirement_ids)
        refs.extend(f"A{item}" for item in extract.acceptance_criterion_ids)
        return ", ".join(refs)

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        """Render non-sensitive change operation metadata for logs."""
        action = str(arguments.get("action", "?"))
        change_id = arguments.get("id")
        suffix = f" | id=CH{change_id}" if change_id is not None else ""
        return f"action={action}{suffix}"


__all__ = ["ChangeExtract", "ChangeTool"]
