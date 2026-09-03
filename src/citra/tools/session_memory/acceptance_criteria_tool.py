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
                "their wording or evidence, mark them satisfied, or remove "
                "obsolete criteria."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Acceptance criteria operation.",
                            enum=("add", "update", "satisfy", "remove"),
                        ),
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description="Acceptance criterion text."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="contents",
                        schema=JsonSchema.array(
                            items=JsonSchema.string(),
                            description="Criteria to add in batch.",
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
                            description="Criterion IDs for removal.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="evidence",
                        schema=JsonSchema.string(
                            description="Evidence supporting satisfaction."
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
        self._extracts: list[AcceptanceCriterionExtract] = []
        self._next_id = 1

    @property
    @override
    def heading(self) -> str:
        return "Acceptance Criteria"

    @override
    def get_extracts(self) -> list[AcceptanceCriterionExtract]:
        return list(self._extracts)

    @override
    def format_extract(self, extract: AcceptanceCriterionExtract) -> str:
        status = "DONE" if extract.satisfied else "PENDING"
        evidence = (
            f" | Evidence: {extract.evidence}"
            if extract.evidence
            else ""
        )
        return f"- [{status}] [A{extract.id}] {extract.content}{evidence}"

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
        if action == "satisfy":
            return self._satisfy(arguments)
        if action == "remove":
            return self._remove(arguments)
        raise ValueError(f"Unsupported acceptance criteria action: {action}")

    def _add(self, arguments: dict[str, Any]) -> str:
        self._reject(arguments, ("id", "ids"), action="add")
        content = arguments.get("content")
        contents = arguments.get("contents")

        if content is not None and contents is not None:
            raise ValueError("Use either 'content' or 'contents', not both.")

        raw = [content] if content is not None else contents
        if not raw:
            raise ValueError("'content' or 'contents' is required for add.")

        added = []
        for index, value in enumerate(raw):
            normalized = str(value).strip()
            if not normalized:
                raise ValueError(f"contents[{index}] cannot be empty.")

            item = AcceptanceCriterionExtract(
                id=self._next_id,
                content=normalized,
            )
            self._next_id += 1
            self._extracts.append(item)
            added.append(item)

        if len(added) == 1:
            return f"Added ACCEPTANCE CRITERION [A{added[0].id}]: {added[0].content}"
        return f"Added {len(added)} acceptance criteria."

    def _update(self, arguments: dict[str, Any]) -> str:
        self._reject(arguments, ("contents", "ids"), action="update")
        criterion_id = arguments.get("id")
        if criterion_id is None:
            raise ValueError("'id' is required for update.")

        content = str(arguments.get("content") or "").strip()
        if not content:
            raise ValueError("'content' is required for update.")

        index = self._find_index(criterion_id)
        current = self._extracts[index]
        updated = replace(
            current,
            content=content,
            evidence=arguments.get("evidence", current.evidence),
        )
        self._extracts[index] = updated
        return f"Updated ACCEPTANCE CRITERION [A{updated.id}]: {updated.content}"

    def _satisfy(self, arguments: dict[str, Any]) -> str:
        self._reject(arguments, ("contents", "ids"), action="satisfy")
        criterion_id = arguments.get("id")
        if criterion_id is None:
            raise ValueError("'id' is required for satisfy.")

        index = self._find_index(criterion_id)
        current = self._extracts[index]
        updated = replace(
            current,
            satisfied=True,
            evidence=arguments.get("evidence", current.evidence),
        )
        self._extracts[index] = updated
        return f"Satisfied ACCEPTANCE CRITERION [A{updated.id}]."

    def _remove(self, arguments: dict[str, Any]) -> str:
        ids = self._get_ids(arguments, action="remove")
        self._extracts = [
            item for item in self._extracts if item.id not in ids
        ]
        return f"Removed acceptance criteria: {', '.join(str(i) for i in ids)}"
