from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override, TYPE_CHECKING

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
class DecisionExtract:
    id: int
    content: str
    working_state_id: int | None = None


class DecisionTool(MemoryTool[DecisionExtract]):
    """Manage durable decisions, optionally promoted from working state."""

    TOOL_ID = "decision"

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="decision",
            description=(
                "Manage durable decisions for the current conversation. Use "
                "'add' for established implementation, architectural, "
                "behavioral, or design choices that do not need provisional "
                "working state, 'promote' when an active working state "
                "produced the decision, and 'remove' when it becomes stale, "
                "incorrect, or obsolete."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Decision operation.",
                            enum=(
                                "add",
                                "promote",
                                "remove",
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description=(
                                "Decision text for direct add, or optional "
                                "polished text for a single promotion."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="contents",
                        schema=JsonSchema.array(
                            items=JsonSchema.string(),
                            description=(
                                "Decisions to add directly as a batch."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="working_state_id",
                        schema=JsonSchema.integer(
                            description=(
                                "Single working-state ID to promote."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="working_state_ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description=(
                                "Working-state IDs to promote as a batch. "
                                "Their contents become the decisions."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description=(
                                "Single decision ID for remove."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
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

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
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
        super().__init__(
            context=context,
            session=session,
        )

        self.__extracts: list[DecisionExtract] = []
        self.__next_id = 1

    @property
    @override
    def heading(self) -> str:
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
        extract: DecisionExtract,
    ) -> str:
        text = (
            f"- [{extract.id}] "
            f"{extract.content}"
        )

        if extract.working_state_id is not None:
            text += (
                " (from working state "
                f"W{extract.working_state_id})"
            )

        return text

    @override
    def should_offer_documentation(
        self,
    ) -> bool:
        return bool(
            self.__extracts
        )

    # Everything below this point can remain unchanged.

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        action = arguments["action"]
        if action == "add":
            return self._add(arguments)
        if action == "promote":
            return self._promote(arguments)
        if action == "remove":
            return self._remove(arguments)
        raise ValueError(f"Unsupported decision action: {action}")

    def _add(self, arguments: dict[str, Any]) -> str:
        self._reject_fields(
            arguments,
            ("working_state_id", "working_state_ids", "id", "ids"),
            action="add",
        )
        contents = self._direct_contents(arguments)
        added = [self._append(content, working_state_id=None) for content in contents]
        if len(added) == 1:
            item = added[0]
            return f"Added DECISION [{item.id}]: {item.content}"
        return (
            f"Added {len(added)} DECISIONs "
            f"{self._format_ids([item.id for item in added])}."
        )

    def _promote(self, arguments: dict[str, Any]) -> str:
        self._reject_fields(arguments, ("contents", "id", "ids"), action="promote")
        working_ids = self._working_ids(arguments)
        content_override = arguments.get("content")
        if len(working_ids) != 1 and content_override is not None:
            raise ValueError("'content' is only valid for a single promotion.")

        prepared: list[tuple[int, str]] = []
        for working_id in working_ids:
            working = self.require_working_state(working_id)
            content = (
                str(content_override).strip()
                if content_override is not None
                else working.content
            )
            if not content:
                raise ValueError("Decision content cannot be empty.")
            prepared.append((working_id, content))

        added: list[DecisionExtract] = []
        for working_id, content in prepared:
            item = self._append(content, working_state_id=working_id)
            self.register_promotion(
                working_id,
                kind="decision",
                memory_id=item.id,
            )
            added.append(item)

        if len(added) == 1:
            item = added[0]
            return (
                f"Promoted working state [W{item.working_state_id}] to "
                f"DECISION [{item.id}]: {item.content}"
            )
        return (
            f"Promoted {len(added)} working states to DECISION entries "
            f"{self._format_ids([item.id for item in added])}."
        )

    def _remove(self, arguments: dict[str, Any]) -> str:
        self._reject_fields(
            arguments,
            ("content", "contents", "working_state_id", "working_state_ids"),
            action="remove",
        )
        ids = self._ids(arguments)
        selected = [self.__extracts[self._find_index(item_id)] for item_id in ids]
        id_set = set(ids)
        self.__extracts = [item for item in self.__extracts if item.id not in id_set]
        for item in selected:
            if item.working_state_id is not None:
                self.unregister_promotion(
                    item.working_state_id,
                    kind="decision",
                    memory_id=item.id,
                )
        if len(selected) == 1:
            item = selected[0]
            return f"Removed DECISION [{item.id}]: {item.content}"
        return f"Removed {len(selected)} DECISION entries {self._format_ids(ids)}."

    def _append(self, content: str, *, working_state_id: int | None) -> DecisionExtract:
        item = DecisionExtract(
            id=self.__next_id,
            content=content,
            working_state_id=working_state_id,
        )
        self.__next_id += 1
        self.__extracts.append(item)
        return item

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
        ids = self._ids_summary(arguments)
        if ids:
            parts.append(f"ids={ids}")
        return " | ".join(parts)

    @staticmethod
    def _direct_contents(arguments: dict[str, Any]) -> list[str]:
        content = arguments.get("content")
        contents = arguments.get("contents")
        if content is not None and contents is not None:
            raise ValueError("Use either 'content' or 'contents', not both.")
        raw = [content] if content is not None else contents
        if not raw:
            raise ValueError("'content' or 'contents' is required for add.")
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
    def _ids(arguments: dict[str, Any]) -> list[int]:
        single = arguments.get("id")
        multiple = arguments.get("ids")
        if single is not None and multiple is not None:
            raise ValueError("Use either 'id' or 'ids', not both.")
        ids = [single] if single is not None else multiple
        if not ids:
            raise ValueError("'id' or 'ids' is required for remove.")
        if len(ids) != len(set(ids)):
            raise ValueError("Decision IDs cannot contain duplicates.")
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
                + f" are invalid for decision action '{action}'."
            )

    def _find_index(self, decision_id: int) -> int:
        for index, item in enumerate(self.__extracts):
            if item.id == decision_id:
                return index
        raise ValueError(f"DECISION [{decision_id}] does not exist.")

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
            return DecisionTool._format_ids(list(multiple))
        return None

    @staticmethod
    def _format_ids(ids: list[int]) -> str:
        return "[" + ", ".join(str(item_id) for item_id in ids) + "]"

    @staticmethod
    def _truncate(value: str) -> str:
        return value if len(value) <= 80 else value[:80] + "..."
