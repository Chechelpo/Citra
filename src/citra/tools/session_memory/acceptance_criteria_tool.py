"""Acceptance criteria with requirement links and review lifecycle."""

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
from .requirement_tool import RequirementTool

if TYPE_CHECKING:
    from citra.agent import AgentSession
    from citra.context import ExecutionContext


@dataclass(frozen=True)
class AcceptanceCriterionExtract:
    """Represent one observable success condition and its requirement links."""

    id: int
    content: str
    requirement_ids: tuple[int, ...] = ()
    satisfied: bool = False
    evidence: str | None = None


class AcceptanceCriteriaTool(MemoryTool[AcceptanceCriterionExtract]):
    """Track observable success conditions for later independent review."""

    TOOL_ID = "acceptance_criteria"
    CAPABILITIES = ToolCapabilities(
        actions=("add", "update", "satisfy", "reopen", "remove"),
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="acceptance_criteria",
            description=(
                "Manage durable, observable acceptance criteria. Link each "
                "criterion to the requirement IDs it proves. Updating or "
                "reopening a criterion invalidates earlier satisfaction "
                "evidence; satisfy it only after independent verification."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Acceptance-criterion operation.",
                            enum=("add", "update", "satisfy", "reopen", "remove"),
                        ),
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description="Criterion text for single add or update."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="contents",
                        schema=JsonSchema.array(
                            items=JsonSchema.string(),
                            description="Criteria to add as a batch.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description="Single criterion ID."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="Criterion IDs for a batch action.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="requirement_ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description=(
                                "Requirement IDs whose success this criterion "
                                "helps demonstrate."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="evidence",
                        schema=JsonSchema.string(
                            description="Concise evidence used for satisfaction."
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
        """Return the single model-facing acceptance definition."""
        del context
        return (ToolDefinition(definition=cls.DEFINITION),)

    def __init__(
        self,
        context: ExecutionContext,
        session: AgentSession,
    ) -> None:
        """Initialize empty acceptance-criterion memory."""
        super().__init__(context=context, session=session)
        self._extracts: list[AcceptanceCriterionExtract] = []
        self._next_id = 1

    @property
    @override
    def heading(self) -> str:
        """Return the memory-section heading."""
        return "Acceptance Criteria"

    @override
    def get_extracts(self) -> list[AcceptanceCriterionExtract]:
        """Return a defensive copy of acceptance criteria."""
        return list(self._extracts)

    @override
    def format_extract(self, extract: AcceptanceCriterionExtract) -> str:
        """Render one criterion with links and adjudication evidence."""
        status = "DONE" if extract.satisfied else "PENDING"
        links = ""
        if extract.requirement_ids:
            linked = ", ".join(f"R{item}" for item in extract.requirement_ids)
            links = f" | proves: {linked}"
        evidence = f" | evidence: {extract.evidence}" if extract.evidence else ""
        return f"- [{status}] [A{extract.id}] {extract.content}{links}{evidence}"

    @override
    def should_offer_documentation(self) -> bool:
        """Offer established acceptance criteria for repository documentation."""
        return bool(self._extracts)

    def has_unsatisfied_criteria(self) -> bool:
        """Return whether any valid criterion lacks review acceptance."""
        return any(not item.satisfied for item in self._extracts)

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        """Dispatch one validated acceptance-criterion action."""
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
        raise ValueError(f"Unsupported acceptance criteria action: {action}")

    def _add(self, arguments: dict[str, Any]) -> str:
        """Add one or more pending acceptance criteria."""
        self._reject(arguments, ("id", "ids", "evidence"), action="add")
        content = arguments.get("content")
        contents = arguments.get("contents")
        if content is not None and contents is not None:
            raise ValueError("Use either 'content' or 'contents', not both.")
        raw = [content] if content is not None else contents
        if not raw:
            raise ValueError("'content' or 'contents' is required for add.")

        requirement_ids = self._requirement_ids(arguments)
        normalized: list[str] = []
        for index, value in enumerate(raw):
            text = str(value).strip()
            if not text:
                raise ValueError(f"contents[{index}] cannot be empty.")
            normalized.append(text)

        added: list[AcceptanceCriterionExtract] = []
        for text in normalized:
            item = AcceptanceCriterionExtract(
                id=self._next_id,
                content=text,
                requirement_ids=requirement_ids,
            )
            self._next_id += 1
            self._extracts.append(item)
            added.append(item)
        if len(added) == 1:
            return (
                f"Added ACCEPTANCE CRITERION [A{added[0].id}]: "
                f"{added[0].content}"
            )
        ids = tuple(item.id for item in added)
        return f"Added {len(added)} acceptance criteria {self._format_ids(ids)}."

    def _update(self, arguments: dict[str, Any]) -> str:
        """Update one criterion and invalidate prior satisfaction evidence."""
        self._reject(arguments, ("contents", "ids", "evidence"), action="update")
        criterion_id = arguments.get("id")
        if criterion_id is None:
            raise ValueError("'id' is required for update.")
        if (
            arguments.get("content") is None
            and arguments.get("requirement_ids") is None
        ):
            raise ValueError(
                "Acceptance-criterion update requires content or requirement_ids."
            )
        index = self._find_index(criterion_id)
        current = self._extracts[index]
        content = (
            current.content
            if arguments.get("content") is None
            else str(arguments["content"]).strip()
        )
        if not content:
            raise ValueError("'content' cannot be empty.")
        requirement_ids = (
            current.requirement_ids
            if arguments.get("requirement_ids") is None
            else self._requirement_ids(arguments)
        )
        updated = replace(
            current,
            content=content,
            requirement_ids=requirement_ids,
            satisfied=False,
            evidence=None,
        )
        self._extracts[index] = updated
        return f"Updated ACCEPTANCE CRITERION [A{updated.id}]: {updated.content}"

    def _set_satisfied(
        self,
        arguments: dict[str, Any],
        *,
        satisfied: bool,
    ) -> str:
        """Satisfy or reopen selected criteria with coherent evidence state."""
        action = "satisfy" if satisfied else "reopen"
        self._reject(
            arguments,
            ("content", "contents", "requirement_ids"),
            action=action,
        )
        if not satisfied and arguments.get("evidence") is not None:
            raise ValueError("'evidence' is invalid for reopen.")
        ids = self._selected_ids(arguments, action=action)
        evidence_raw = arguments.get("evidence")
        evidence = str(evidence_raw).strip() if evidence_raw is not None else None
        if evidence_raw is not None and not evidence:
            raise ValueError("'evidence' cannot be empty.")
        for criterion_id in ids:
            index = self._find_index(criterion_id)
            self._extracts[index] = replace(
                self._extracts[index],
                satisfied=satisfied,
                evidence=evidence if satisfied else None,
            )
        verb = "Satisfied" if satisfied else "Reopened"
        return f"{verb} acceptance criteria {self._format_ids(ids)}."

    def _remove(self, arguments: dict[str, Any]) -> str:
        """Remove criteria established as obsolete or incorrect."""
        self._reject(
            arguments,
            ("content", "contents", "requirement_ids", "evidence"),
            action="remove",
        )
        ids = self._selected_ids(arguments, action="remove")
        for criterion_id in ids:
            self._find_index(criterion_id)
        selected = set(ids)
        self._extracts = [item for item in self._extracts if item.id not in selected]
        return f"Removed acceptance criteria {self._format_ids(ids)}."

    def _requirement_ids(self, arguments: dict[str, Any]) -> tuple[int, ...]:
        """Normalize and validate requirement links supplied by a caller."""
        ids = self.normalize_reference_ids(
            arguments.get("requirement_ids"),
            field="requirement_ids",
        )
        self.require_memory_ids(RequirementTool, ids, field="requirement_ids")
        return ids

    def _find_index(self, criterion_id: int) -> int:
        """Return the list index for an existing criterion ID."""
        for index, item in enumerate(self._extracts):
            if item.id == criterion_id:
                return index
        raise ValueError(f"Acceptance criterion A{criterion_id} not found.")

    @staticmethod
    def _selected_ids(
        arguments: dict[str, Any],
        *,
        action: str,
    ) -> tuple[int, ...]:
        """Normalize one-or-many criterion IDs for a lifecycle action."""
        single = arguments.get("id")
        multiple = arguments.get("ids")
        if single is not None and multiple is not None:
            raise ValueError("Use either 'id' or 'ids', not both.")
        values = [single] if single is not None else multiple
        if not values:
            raise ValueError(f"'id' or 'ids' is required for {action}.")
        if len(values) != len(set(values)):
            raise ValueError("Acceptance criterion IDs cannot contain duplicates.")
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
            raise ValueError(
                f"{rendered} are invalid for acceptance action {action!r}."
            )

    @staticmethod
    def _format_ids(ids: tuple[int, ...]) -> str:
        """Format criterion IDs for concise results."""
        return "[" + ", ".join(f"A{item}" for item in ids) + "]"

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        """Render non-sensitive acceptance operation metadata for logs."""
        action = str(arguments.get("action", "?"))
        criterion_id = arguments.get("id")
        suffix = f" | id=A{criterion_id}" if criterion_id is not None else ""
        return f"action={action}{suffix}"


__all__ = ["AcceptanceCriterionExtract", "AcceptanceCriteriaTool"]
