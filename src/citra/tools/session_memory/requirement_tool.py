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
from .memory_tool import MemoryTool

if TYPE_CHECKING:
    from citra.agent import AgentSession
    from citra.context import ExecutionContext


@dataclass(frozen=True)
class RequirementExtract:
    """Represent RequirementExtract."""
    id: int
    content: str
    satisfied: bool = False
    evidence: str | None = None


class RequirementTool(MemoryTool[RequirementExtract]):
    """Track durable task requirements and their verified satisfaction."""

    TOOL_ID = "requirement"
    CAPABILITIES = ToolCapabilities(
        actions=("add", "update", "satisfy", "reopen", "remove"),
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="requirement",
            description=(
                "Manage durable task requirements. Add requirements as soon "
                "as they are established, update them when clarified, mark "
                "them satisfied only after verification, reopen them when "
                "evidence is invalidated, and remove only obsolete or "
                "incorrect requirements."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Requirement operation.",
                            enum=(
                                "add",
                                "update",
                                "satisfy",
                                "reopen",
                                "remove",
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description="Requirement text for add or update."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="contents",
                        schema=JsonSchema.array(
                            items=JsonSchema.string(),
                            description="Requirements to add as a batch.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description="Single requirement ID."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="Requirement IDs for a batch action.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="evidence",
                        schema=JsonSchema.string(
                            description=(
                                "Concise verification evidence for satisfy."
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
        self._extracts: list[RequirementExtract] = []
        self._next_id = 1

    @property
    @override
    def heading(self) -> str:
        """Handle heading."""
        return "Requirements"

    @override
    def get_extracts(self) -> list[RequirementExtract]:
        """Return get extracts."""
        return list(self._extracts)

    @override
    def format_extract(self, extract: RequirementExtract) -> str:
        """Handle format extract."""
        mark = "x" if extract.satisfied else " "
        text = f"- [{mark}] [R{extract.id}] {extract.content}"
        if extract.evidence:
            text += f"\n  - evidence: {extract.evidence}"
        return text

    @override
    def should_offer_documentation(self) -> bool:
        """Return whether should offer documentation."""
        return bool(self._extracts)

    def has_unsatisfied_requirements(self) -> bool:
        """Return whether has unsatisfied requirements."""
        return any(not item.satisfied for item in self._extracts)

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        """Execute the execute operation."""
        action = arguments["action"]
        if action == "add":
            return self._add(arguments)
        if action == "update":
            return self._update(arguments)
        if action == "satisfy":
            return self._set_satisfied(arguments, satisfied=True)
        if action == "reopen":
            return self._set_satisfied(arguments, satisfied=False)
        if action == "remove":
            return self._remove(arguments)
        raise ValueError(f"Unsupported requirement action: {action}")

    def _add(self, arguments: dict[str, Any]) -> str:
        """Handle add."""
        self._reject(arguments, ("id", "ids", "evidence"), action="add")
        content = arguments.get("content")
        contents = arguments.get("contents")
        if content is not None and contents is not None:
            raise ValueError("Use either 'content' or 'contents', not both.")
        raw = [content] if content is not None else contents
        if not raw:
            raise ValueError("'content' or 'contents' is required for add.")

        added: list[RequirementExtract] = []
        for index, value in enumerate(raw):
            normalized = str(value).strip()
            if not normalized:
                raise ValueError(f"contents[{index}] cannot be empty.")
            item = RequirementExtract(id=self._next_id, content=normalized)
            self._next_id += 1
            self._extracts.append(item)
            added.append(item)
        if len(added) == 1:
            return f"Added REQUIREMENT [R{added[0].id}]: {added[0].content}"
        return f"Added {len(added)} requirements {self._format_ids(added)}."

    def _update(self, arguments: dict[str, Any]) -> str:
        """Handle update."""
        self._reject(arguments, ("contents", "ids", "evidence"), action="update")
        requirement_id = arguments.get("id")
        if requirement_id is None:
            raise ValueError("'id' is required for update.")
        content = str(arguments.get("content") or "").strip()
        if not content:
            raise ValueError("'content' is required for update.")
        index = self._find_index(requirement_id)
        current = self._extracts[index]
        updated = replace(
            current,
            content=content,
            satisfied=False,
            evidence=None,
        )
        self._extracts[index] = updated
        return f"Updated REQUIREMENT [R{updated.id}]: {updated.content}"

    def _set_satisfied(
        self,
        arguments: dict[str, Any],
        *,
        satisfied: bool,
    ) -> str:
        """Handle set satisfied."""
        action = "satisfy" if satisfied else "reopen"
        self._reject(arguments, ("content", "contents"), action=action)
        if not satisfied and arguments.get("evidence") is not None:
            raise ValueError("'evidence' is invalid for reopen.")
        ids = self._get_ids(arguments, action=action)
        evidence_raw = arguments.get("evidence")
        evidence = (
            str(evidence_raw).strip() if evidence_raw is not None else None
        )
        if evidence_raw is not None and not evidence:
            raise ValueError("'evidence' cannot be empty.")
        for requirement_id in ids:
            index = self._find_index(requirement_id)
            self._extracts[index] = replace(
                self._extracts[index],
                satisfied=satisfied,
                evidence=evidence if satisfied else None,
            )
        verb = "Satisfied" if satisfied else "Reopened"
        return f"{verb} requirements {self._format_raw_ids(ids)}."

    def _remove(self, arguments: dict[str, Any]) -> str:
        """Handle remove."""
        self._reject(
            arguments,
            ("content", "contents", "evidence"),
            action="remove",
        )
        ids = self._get_ids(arguments, action="remove")
        for requirement_id in ids:
            self._find_index(requirement_id)
        selected = set(ids)
        self._extracts = [
            item for item in self._extracts if item.id not in selected
        ]
        return f"Removed requirements {self._format_raw_ids(ids)}."

    def _get_ids(self, arguments: dict[str, Any], *, action: str) -> list[int]:
        """Handle get ids."""
        single = arguments.get("id")
        multiple = arguments.get("ids")
        if single is not None and multiple is not None:
            raise ValueError("Use either 'id' or 'ids', not both.")
        ids = [single] if single is not None else multiple
        if not ids:
            raise ValueError(f"'id' or 'ids' is required for {action}.")
        if len(ids) != len(set(ids)):
            raise ValueError("Requirement IDs cannot contain duplicates.")
        return list(ids)

    def _find_index(self, requirement_id: int) -> int:
        """Handle find index."""
        for index, item in enumerate(self._extracts):
            if item.id == requirement_id:
                return index
        raise ValueError(f"REQUIREMENT [R{requirement_id}] does not exist.")

    @staticmethod
    def _reject(
        arguments: dict[str, Any],
        names: tuple[str, ...],
        *,
        action: str,
    ) -> None:
        """Handle reject."""
        invalid = [name for name in names if arguments.get(name) is not None]
        if invalid:
            rendered = ", ".join(f"'{name}'" for name in invalid)
            raise ValueError(
                f"{rendered} are invalid for requirement action {action!r}."
            )

    @staticmethod
    def _format_ids(items: list[RequirementExtract]) -> str:
        """Handle format ids."""
        return RequirementTool._format_raw_ids([item.id for item in items])

    @staticmethod
    def _format_raw_ids(ids: list[int]) -> str:
        """Handle format raw ids."""
        return "[" + ", ".join(f"R{item}" for item in ids) + "]"

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        """Handle format call log."""
        action = arguments.get("action", "?")
        return f"action={action}"


__all__ = ["RequirementExtract", "RequirementTool"]
