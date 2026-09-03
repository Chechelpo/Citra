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
class AcceptanceCriterionExtract:
    """Represent AcceptanceCriterionExtract."""

    id: int
    content: str
    satisfied: bool = False
    evidence: str | None = None


class AcceptanceCriteriaTool(MemoryTool[AcceptanceCriterionExtract]):
    """Track durable acceptance criteria for a requested change.

    Acceptance criteria describe how the team can verify that the requested
    behavior is complete. They bridge requirements and later verification
    phases without becoming implementation details.
    """

    TOOL_ID = "acceptance_criteria"

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="acceptance_criteria",
            description=(
                "Manage durable acceptance criteria. Add criteria, update "
                "criterion wording or evidence, mark criteria satisfied, "
                "or remove obsolete criteria."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Acceptance criteria operation.",
                            enum=("add", "update", "satisfy", "remove"),
                        ),
                        required=True,
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description=(
                                "Acceptance criterion text. Required for "
                                "update and can be used for single-item add."
                            )
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="contents",
                        schema=JsonSchema.array(
                            items=JsonSchema.string(),
                            description=(
                                "Multiple acceptance criteria to add in "
                                "a single operation."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description=(
                                "Single acceptance criterion ID. Used by "
                                "update and satisfy operations."
                            )
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description=(
                                "Acceptance criterion IDs to remove."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="evidence",
                        schema=JsonSchema.string(
                            description=(
                                "Evidence associated with the criterion."
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
        self._extracts: list[AcceptanceCriterionExtract] = []
        self._next_id = 1

    @property
    @override
    def heading(self) -> str:
        """Handle heading."""
        return "Acceptance Criteria"

    @override
    def get_extracts(self) -> list[AcceptanceCriterionExtract]:
        """Return extracts."""
        return list(self._extracts)

    @override
    def format_extract(
        self,
        extract: AcceptanceCriterionExtract,
    ) -> str:
        """Format extract."""
        status = "DONE" if extract.satisfied else "PENDING"
        evidence = (
            f" | Evidence: {extract.evidence}"
            if extract.evidence
            else ""
        )
        return f"- [{status}] [A{extract.id}] {extract.content}{evidence}"

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

        if action == "satisfy":
            return self._satisfy(arguments)

        if action == "remove":
            return self._remove(arguments)

        raise ValueError(
            f"Unsupported acceptance criteria action: {action}"
        )

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

        added = []

        for index, value in enumerate(raw):
            normalized = str(value).strip()

            if not normalized:
                raise ValueError(
                    f"contents[{index}] cannot be empty."
                )

            item = AcceptanceCriterionExtract(
                id=self._next_id,
                content=normalized,
            )

            self._next_id += 1
            self._extracts.append(item)
            added.append(item)

        if len(added) == 1:
            return (
                f"Added ACCEPTANCE CRITERION "
                f"[A{added[0].id}]: {added[0].content}"
            )

        return f"Added {len(added)} acceptance criteria."

    def _update(self, arguments: dict[str, Any]) -> str:
        """Handle update."""
        self._reject(
            arguments,
            ("contents", "ids"),
            action="update",
        )

        criterion_id = arguments.get("id")

        if criterion_id is None:
            raise ValueError(
                "'id' is required for update."
            )

        content = str(arguments.get("content") or "").strip()

        if not content:
            raise ValueError(
                "'content' is required for update."
            )

        index = self._find_index(criterion_id)
        current = self._extracts[index]

        updated = replace(
            current,
            content=content,
            evidence=arguments.get(
                "evidence",
                current.evidence,
            ),
        )

        self._extracts[index] = updated

        return (
            f"Updated ACCEPTANCE CRITERION "
            f"[A{updated.id}]: {updated.content}"
        )

    def _satisfy(self, arguments: dict[str, Any]) -> str:
        """Handle satisfy."""
        self._reject(
            arguments,
            ("contents", "ids"),
            action="satisfy",
        )

        criterion_id = arguments.get("id")

        if criterion_id is None:
            raise ValueError(
                "'id' is required for satisfy."
            )

        index = self._find_index(criterion_id)
        current = self._extracts[index]

        updated = replace(
            current,
            satisfied=True,
            evidence=arguments.get(
                "evidence",
                current.evidence,
            ),
        )

        self._extracts[index] = updated

        return (
            f"Satisfied ACCEPTANCE CRITERION "
            f"[A{updated.id}]."
        )

    def _remove(self, arguments: dict[str, Any]) -> str:
        """Handle remove."""
        ids = self._get_ids(arguments, action="remove")

        self._extracts = [
            item
            for item in self._extracts
            if item.id not in ids
        ]

        return (
            "Removed acceptance criteria: "
            f"{', '.join(str(i) for i in ids)}"
        )
    
    def _reject(
        self,
        arguments: dict[str, Any],
        keys: tuple[str, ...],
        *,
        action: str,
    ) -> None:
        """Reject arguments that are invalid for an action."""
        for key in keys:
            if arguments.get(key) is not None:
                raise ValueError(
                    f"'{key}' is not valid for {action}."
                )

    def _find_index(self, criterion_id: int) -> int:
        """Find criterion index by ID."""
        for index, item in enumerate(self._extracts):
            if item.id == criterion_id:
                return index

        raise ValueError(
            f"Acceptance criterion A{criterion_id} not found."
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

        normalized = []

        for index, value in enumerate(ids):
            try:
                criterion_id = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"ids[{index}] must be an integer."
                ) from exc

            normalized.append(criterion_id)

        return normalized