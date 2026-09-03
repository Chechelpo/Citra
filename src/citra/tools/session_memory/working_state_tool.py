from __future__ import annotations

from typing import Any, override, TYPE_CHECKING


from citra.utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)

from ..tool import ToolDefinition
from .memory_tool import MemoryTool, WorkingStateExtract

if TYPE_CHECKING:
    from citra.agent import AgentSession
    from citra.context import ExecutionContext


class WorkingStateTool(MemoryTool[WorkingStateExtract]):
    """Manage active hypotheses and provisional interpretations."""

    TOOL_ID = "working_state"

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="working_state",
            description=(
                "Manage optional provisional working state for unresolved "
                "reasoning that must survive context trimming. Working state "
                "may contain hypotheses, tentative interpretations, and "
                "immediate verification plans; it is not authoritative. "
                "Durable TODO, FACT, DECISION, and CONSTRAINT entries may be "
                "created directly without working state. When a working state "
                "genuinely produces durable memory, the corresponding tool may "
                "promote it and preserve provenance. Resolve it after promoted "
                "consequences are captured, or discard it when no durable "
                "memory is warranted."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Working-state operation.",
                            enum=(
                                "create",
                                "update",
                                "resolve",
                                "discard",
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description=(
                                "Working-state content for create/update. "
                                "Use 'contents' to create several states "
                                "at once."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="contents",
                        schema=JsonSchema.array(
                            items=JsonSchema.string(),
                            description=(
                                "Batch of working states for create."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description=(
                                "Single working-state ID for update, "
                                "resolve, or discard."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description=(
                                "Working-state IDs for batch resolve/discard."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        """Handle definitions for context."""
        del context

        return (
            ToolDefinition(
                definition=cls.DEFINITION,
            ),
        )

    def __init__(
        self,
        context: ExecutionContext,
        session: AgentSession,
    ) -> None:
        """Initialize the instance."""
        super().__init__(
            context=context,
            session=session,
        )

    @property
    @override
    def heading(self) -> str:
        """Handle heading."""
        return "Working state"

    @override
    def get_extracts(
        self,
    ) -> list[WorkingStateExtract]:
        """Return get extracts."""
        return self.memory_state.active_working_states()

    @override
    def format_extract(
        self,
        extract: WorkingStateExtract,
    ) -> str:
        """Handle format extract."""
        line = (
            f"- [W{extract.id}] "
            f"{extract.content}"
        )

        if extract.promotions:
            promoted = ", ".join(
                f"{ref.kind.upper()} [{ref.memory_id}]"
                for ref in extract.promotions
            )

            line += (
                f"\n  - promoted: {promoted}"
            )

        return line

    @override
    def should_offer_documentation(
        self,
    ) -> bool:
        """Return whether should offer documentation."""
        return False


    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        """Execute the execute operation."""
        action = arguments["action"]
        if action == "create":
            return self._create(arguments)
        if action == "update":
            return self._update(arguments)
        if action == "resolve":
            return self._resolve(arguments)
        if action == "discard":
            return self._discard(arguments)
        raise ValueError(f"Unsupported working-state action: {action}")

    def _create(self, arguments: dict[str, Any]) -> str:
        """Handle create."""
        if arguments.get("id") is not None or arguments.get("ids") is not None:
            raise ValueError("'id' and 'ids' are invalid for create.")

        contents = self._get_contents(arguments)
        created = [
            self.memory_state.create_working_state(
                content,
                turn=self.session.turn_number,
            )
            for content in contents
        ]

        if len(created) == 1:
            item = created[0]
            return f"Created working state [W{item.id}]: {item.content}"
        return (
            f"Created {len(created)} working states "
            f"[W{created[0].id}-W{created[-1].id}]."
        )

    def _update(self, arguments: dict[str, Any]) -> str:
        """Handle update."""
        if arguments.get("contents") is not None or arguments.get("ids") is not None:
            raise ValueError("Update accepts one 'id' and one 'content'.")
        working_state_id = arguments.get("id")
        if working_state_id is None:
            raise ValueError("'id' is required for update.")
        content = arguments.get("content")
        if content is None:
            raise ValueError("'content' is required for update.")

        updated = self.memory_state.update_working_state(
            working_state_id,
            str(content),
            turn=self.session.turn_number,
        )
        return f"Updated working state [W{updated.id}]: {updated.content}"

    def _resolve(self, arguments: dict[str, Any]) -> str:
        """Handle resolve."""
        self._reject_content(arguments, "resolve")
        ids = self._get_ids(arguments, "resolve")
        # Validate the full batch before mutating it.
        for working_state_id in ids:
            current = self.require_working_state(working_state_id)
            if not current.promotions:
                raise ValueError(
                    f"Working state [W{working_state_id}] has no promotions."
                )
        resolved = [
            self.memory_state.resolve_working_state(working_state_id)
            for working_state_id in ids
        ]
        if len(resolved) == 1:
            return f"Resolved working state [W{resolved[0].id}]."
        return f"Resolved working states {self._format_ids(ids)}."

    def _discard(self, arguments: dict[str, Any]) -> str:
        """Handle discard."""
        self._reject_content(arguments, "discard")
        ids = self._get_ids(arguments, "discard")
        # Validate the full batch before mutating it.
        for working_state_id in ids:
            current = self.require_working_state(working_state_id)
            if current.promotions:
                raise ValueError(
                    f"Working state [W{working_state_id}] has durable promotions "
                    "and cannot be discarded."
                )
        discarded = [
            self.memory_state.discard_working_state(working_state_id)
            for working_state_id in ids
        ]
        if len(discarded) == 1:
            return f"Discarded working state [W{discarded[0].id}]."
        return f"Discarded working states {self._format_ids(ids)}."

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        """Handle format call log."""
        action = arguments.get("action", "?")
        parts = [f"action={action}"]
        content = arguments.get("content")
        contents = arguments.get("contents")
        if content is not None:
            parts.append(f"content={self._truncate(str(content))}")
        elif contents is not None:
            parts.append(f"batch={len(contents)}")
        ids = self._ids_summary(arguments)
        if ids is not None:
            parts.append(f"ids={ids}")
        return " | ".join(parts)

    @staticmethod
    def _get_contents(arguments: dict[str, Any]) -> list[str]:
        """Handle get contents."""
        content = arguments.get("content")
        contents = arguments.get("contents")
        if content is not None and contents is not None:
            raise ValueError("Use either 'content' or 'contents', not both.")
        if content is not None:
            contents = [content]
        if not contents:
            raise ValueError("'content' or 'contents' is required for create.")

        normalized: list[str] = []
        for index, item in enumerate(contents):
            text = str(item).strip()
            if not text:
                raise ValueError(f"contents[{index}] cannot be empty.")
            normalized.append(text)
        return normalized

    @staticmethod
    def _get_ids(arguments: dict[str, Any], action: str) -> list[int]:
        """Handle get ids."""
        single = arguments.get("id")
        multiple = arguments.get("ids")
        if single is not None and multiple is not None:
            raise ValueError("Use either 'id' or 'ids', not both.")
        ids = [single] if single is not None else multiple
        if not ids:
            raise ValueError(f"'id' or 'ids' is required for {action}.")
        if len(ids) != len(set(ids)):
            raise ValueError("'ids' cannot contain duplicates.")
        return list(ids)

    @staticmethod
    def _reject_content(arguments: dict[str, Any], action: str) -> None:
        """Handle reject content."""
        if arguments.get("content") is not None or arguments.get("contents") is not None:
            raise ValueError(f"'content' and 'contents' are invalid for {action}.")

    @staticmethod
    def _ids_summary(arguments: dict[str, Any]) -> str | None:
        """Handle ids summary."""
        single = arguments.get("id")
        multiple = arguments.get("ids")
        if single is not None:
            return f"[W{single}]"
        if multiple is not None:
            return WorkingStateTool._format_ids(list(multiple))
        return None

    @staticmethod
    def _format_ids(ids: list[int]) -> str:
        """Handle format ids."""
        return "[" + ", ".join(f"W{working_id}" for working_id in ids) + "]"

    @staticmethod
    def _truncate(value: str) -> str:
        """Handle truncate."""
        return value if len(value) <= 80 else value[:80] + "..."
