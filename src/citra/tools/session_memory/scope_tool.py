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
                "excluded scope items as they become known, update them when "
                "clarified, and remove obsolete scope boundaries."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Scope operation.",
                            enum=("add", "update", "remove"),
                        ),
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description="Scope item text."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="contents",
                        schema=JsonSchema.array(
                            items=JsonSchema.string(),
                            description="Scope items to add as a batch.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description="Single scope item ID."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="Scope IDs for batch removal.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="included",
                        schema=JsonSchema.boolean(
                            description=(
                                "Whether the item is included in scope. "
                                "False marks an explicit exclusion."
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
        del context
        return (ToolDefinition(definition=cls.DEFINITION),)

    def __init__(
        self,
        context: ExecutionContext,
        session: AgentSession,
    ) -> None:
        super().__init__(context=context, session=session)
        self._extracts: list[ScopeExtract] = []
        self._next_id = 1

    @property
    @override
    def heading(self) -> str:
        return "Scope"

    @override
    def get_extracts(self) -> list[ScopeExtract]:
        return list(self._extracts)

    @override
    def format_extract(self, extract: ScopeExtract) -> str:
        marker = "IN" if extract.included else "OUT"
        return f"- [{marker}] [S{extract.id}] {extract.content}"

    @override
    def should_offer_documentation(self) -> bool:
        return bool(self._extracts)

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        action = arguments["action"]
        if action == "add":
            return self._add(arguments)
        if action == "update":
            return self._update(arguments)
        if action == "remove":
            return self._remove(arguments)
        raise ValueError(f"Unsupported scope action: {action}")

    def _add(self, arguments: dict[str, Any]) -> str:
        self._reject(arguments, ("id", "ids"), action="add")
        content = arguments.get("content")
        contents = arguments.get("contents")
        if content is not None and contents is not None:
            raise ValueError("Use either 'content' or 'contents', not both.")
        raw = [content] if content is not None else contents
        if not raw:
            raise ValueError("'content' or 'contents' is required for add.")

        included = arguments.get("included", True)
        added: list[ScopeExtract] = []
        for index, value in enumerate(raw):
            normalized = str(value).strip()
            if not normalized:
                raise ValueError(f"contents[{index}] cannot be empty.")
            item = ScopeExtract(
                id=self._next_id,
                content=normalized,
                included=bool(included),
            )
            self._next_id += 1
            self._extracts.append(item)
            added.append(item)

        if len(added) == 1:
            return f"Added SCOPE [S{added[0].id}]: {added[0].content}"
        return f"Added {len(added)} scope items."

    def _update(self, arguments: dict[str, Any]) -> str:
        self._reject(arguments, ("contents", "ids"), action="update")
        scope_id = arguments.get("id")
        if scope_id is None:
            raise ValueError("'id' is required for update.")
        content = str(arguments.get("content") or "").strip()
        if not content:
            raise ValueError("'content' is required for update.")
        index = self._find_index(scope_id)
        current = self._extracts[index]
        updated = replace(
            current,
            content=content,
            included=arguments.get("included", current.included),
        )
        self._extracts[index] = updated
        return f"Updated SCOPE [S{updated.id}]: {updated.content}"

    def _remove(self, arguments: dict[str, Any]) -> str:
        ids = self._get_ids(arguments, action="remove")
        self._extracts = [
            item for item in self._extracts if item.id not in ids
        ]
        return f"Removed scope items: {', '.join(str(i) for i in ids)}"
