from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, override

from citra.utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)

from ..tool import ToolDefinition
from .memory_tool import MemoryTool

if TYPE_CHECKING:
    from citra.agent import AgentSession
    from citra.context import ExecutionContext


@dataclass(frozen=True)
class ScopeExtract:
    """Represent ScopeExtract."""

    id: int
    content: str
    included: bool = True


class ScopeTool(MemoryTool[ScopeExtract]):
    """Track durable task scope boundaries.

    Scope records what belongs to the requested change and what is explicitly
    excluded, preventing later workflow phases from silently expanding the
    task.
    """

    TOOL_ID = "scope"

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="scope",
            description=(
                "Manage durable task scope boundaries. Add included or "
                "excluded scope items, update clarified scope boundaries, "
                "or remove obsolete scope items."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Scope operation.",
                            enum=("add", "update", "remove"),
                        ),
                        required=True,
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description=(
                                "Scope item text. Required for update "
                                "and optional for single-item add."
                            )
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="contents",
                        schema=JsonSchema.array(
                            items=JsonSchema.string(),
                            description=(
                                "Multiple scope items to add in one "
                                "operation."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description=(
                                "Single scope item ID. Used for update."
                            )
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description=(
                                "Scope item IDs to remove."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="included",
                        schema=JsonSchema.boolean(
                            description=(
                                "Whether the item belongs to scope. "
                                "False records an explicit exclusion."
                            )
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
        """Handle definitions for context."""
        del context
        return (ToolDefinition(definition=cls.DEFINITION),)

    def __init__(
        self,
        context: ExecutionContext,
        session: AgentSession,
    ) -> None:
        """Initialize the instance."""
        super().__init__(context=context, session=session)
        self._extracts: list[ScopeExtract] = []
        self._next_id = 1

    @property
    @override
    def heading(self) -> str:
        """Handle heading."""
        return "Scope"

    @override
    def get_extracts(self) -> list[ScopeExtract]:
        """Return extracts."""
        return list(self._extracts)

    @override
    def format_extract(self, extract: ScopeExtract) -> str:
        """Format extract."""
        marker = "IN" if extract.included else "OUT"
        return f"- [{marker}] [S{extract.id}] {extract.content}"

    @override
    def should_offer_documentation(self) -> bool:
        """Return whether documentation should be offered."""
        return bool(self._extracts)

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        """Execute the operation."""
        action = arguments["action"]

        if action == "add":
            return self._add(arguments)

        if action == "update":
            return self._update(arguments)

        if action == "remove":
            return self._remove(arguments)

        raise ValueError(
            f"Unsupported scope action: {action}"
        )

    def _reject(
        self,
        arguments: dict[str, Any],
        keys: tuple[str, ...],
        *,
        action: str,
    ) -> None:
        """Reject arguments invalid for an action."""
        for key in keys:
            if arguments.get(key) is not None:
                raise ValueError(
                    f"'{key}' is not valid for {action}."
                )

    def _find_index(self, scope_id: int) -> int:
        """Find scope item index by ID."""
        for index, item in enumerate(self._extracts):
            if item.id == scope_id:
                return index

        raise ValueError(
            f"Scope item S{scope_id} not found."
        )

    def _get_ids(
        self,
        arguments: dict[str, Any],
        *,
        action: str,
    ) -> list[int]:
        """Extract IDs for batch operations."""
        ids = arguments.get("ids")

        if not ids:
            raise ValueError(
                f"'ids' is required for {action}."
            )

        normalized: list[int] = []

        for index, value in enumerate(ids):
            try:
                scope_id = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"ids[{index}] must be an integer."
                ) from exc

            normalized.append(scope_id)

        return normalized

    def _add(self, arguments: dict[str, Any]) -> str:
        """Handle add."""
        self._reject(arguments, ("id", "ids"), action="add")

        content = arguments.get("content")
        contents = arguments.get("contents")

        if content is not None and contents is not None:
            raise ValueError(
                "Use either 'content' or 'contents', not both."
            )

        raw = [content] if content is not None else contents

        if not raw:
            raise ValueError(
                "'content' or 'contents' is required for add."
            )

        included = arguments.get("included", True)

        added: list[ScopeExtract] = []

        for index, value in enumerate(raw):
            normalized = str(value).strip()

            if not normalized:
                raise ValueError(
                    f"contents[{index}] cannot be empty."
                )

            item = ScopeExtract(
                id=self._next_id,
                content=normalized,
                included=bool(included),
            )

            self._next_id += 1
            self._extracts.append(item)
            added.append(item)

        if len(added) == 1:
            return (
                f"Added SCOPE [S{added[0].id}]: "
                f"{added[0].content}"
            )

        return f"Added {len(added)} scope items."

    def _update(self, arguments: dict[str, Any]) -> str:
        """Handle update."""
        self._reject(
            arguments,
            ("contents", "ids"),
            action="update",
        )

        scope_id = arguments.get("id")

        if scope_id is None:
            raise ValueError(
                "'id' is required for update."
            )

        content = str(arguments.get("content") or "").strip()

        if not content:
            raise ValueError(
                "'content' is required for update."
            )

        index = self._find_index(scope_id)
        current = self._extracts[index]

        updated = replace(
            current,
            content=content,
            included=arguments.get(
                "included",
                current.included,
            ),
        )

        self._extracts[index] = updated

        return (
            f"Updated SCOPE [S{updated.id}]: "
            f"{updated.content}"
        )

    def _remove(self, arguments: dict[str, Any]) -> str:
        """Handle remove."""
        ids = self._get_ids(
            arguments,
            action="remove",
        )

        self._extracts = [
            item
            for item in self._extracts
            if item.id not in ids
        ]

        return (
            "Removed scope items: "
            f"{', '.join(str(i) for i in ids)}"
        )