from __future__ import annotations

from dataclasses import dataclass
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
class Citation:
    type: str
    source: str
    line: int | None = None
    end_line: int | None = None
    reference: str | None = None


@dataclass(frozen=True)
class FactExtract:
    id: int
    content: str
    working_state_id: int | None = None
    citations: tuple[Citation, ...] = ()


class FactTool(MemoryTool[FactExtract]):
    """Manage verified facts, optionally promoted from working state."""

    CITATION_SCHEMA = JsonSchema.object(
        properties=(
            JsonProperty(
                name="type",
                schema=JsonSchema.string(
                    description="Citation source type.",
                    enum=("file", "url"),
                ),
            ),
            JsonProperty(
                name="source",
                schema=JsonSchema.string(
                    description="Workspace-relative file path or source URL.",
                ),
            ),
            JsonProperty(
                name="line",
                schema=JsonSchema.integer(
                    description="Starting line number for a file citation."
                ),
                required=False,
            ),
            JsonProperty(
                name="end_line",
                schema=JsonSchema.integer(
                    description="Ending line number for a file citation."
                ),
                required=False,
            ),
            JsonProperty(
                name="reference",
                schema=JsonSchema.string(
                    description=(
                        "Optional symbol, heading, field, anchor, or other "
                        "locator within the source."
                    ),
                ),
                required=False,
            ),
        ),
        additional_properties=False,
    )

    FACT_SCHEMA = JsonSchema.object(
        properties=(
            JsonProperty(
                name="content",
                schema=JsonSchema.string(description="Fact text."),
            ),
            JsonProperty(
                name="working_state_id",
                schema=JsonSchema.integer(
                    description=(
                        "Working-state ID for promote. Omit when adding a fact "
                        "directly."
                    ),
                ),
                required=False,
            ),
            JsonProperty(
                name="citations",
                schema=JsonSchema.array(
                    items=CITATION_SCHEMA,
                    description="Supporting citations for this fact.",
                ),
                required=False,
            ),
        ),
        additional_properties=False,
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="fact",
            description=(
                "Manage verified facts retained for the current conversation. "
                "Use 'add' when a fact is already established and does not need "
                "provisional working state. Use 'promote' when an active working "
                "state produced the fact and provenance is useful. Use 'remove' "
                "when a fact is stale, incorrect, or superseded."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Fact operation.",
                            enum=("add", "promote", "remove"),
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
                        name="content",
                        schema=JsonSchema.string(
                            description=(
                                "Fact text for direct add, or optional polished "
                                "text for a single promotion."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="citations",
                        schema=JsonSchema.array(
                            items=CITATION_SCHEMA,
                            description="Supporting citations for a single fact.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="facts",
                        schema=JsonSchema.array(
                            items=FACT_SCHEMA,
                            description=(
                                "Batch of facts. For add, omit working_state_id. "
                                "For promote, each entry requires working_state_id."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description="Single fact ID for remove.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="Fact IDs to remove as a batch.",
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
            definition=self.DEFINITION,
            session=session,
        )
        self.__extracts: list[FactExtract] = []
        self.__next_id = 1

    @property
    @override
    def heading(self) -> str:
        return "Facts"

    @override
    def get_extracts(self) -> list[FactExtract]:
        return list(self.__extracts)

    @override
    def format_extract(self, extract: FactExtract) -> str:
        lines = [f"- [{extract.id}] {extract.content}"]
        if extract.working_state_id is not None:
            lines.append(f"  - origin: working state W{extract.working_state_id}")
        for citation in extract.citations:
            if citation.type == "file":
                location = citation.source
                if citation.line is not None:
                    location += f":{citation.line}"
                    if (
                        citation.end_line is not None
                        and citation.end_line != citation.line
                    ):
                        location += f"-{citation.end_line}"
                if citation.reference:
                    location += f" ({citation.reference})"
                lines.append(f"  - source: {location}")
            elif citation.type == "url":
                location = citation.source
                if citation.reference:
                    location += f" ({citation.reference})"
                lines.append(f"  - source: {location}")
        return "\n".join(lines)

    @override
    def should_offer_documentation(self) -> bool:
        return False

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        action = arguments["action"]
        if action == "add":
            return self._add(arguments)
        if action == "promote":
            return self._promote(arguments)
        if action == "remove":
            return self._remove(arguments)
        raise ValueError(f"Unsupported fact action: {action}")

    def _add(self, arguments: dict[str, Any]) -> str:
        if arguments.get("working_state_id") is not None:
            raise ValueError("'working_state_id' is invalid for fact action 'add'.")
        if arguments.get("id") is not None or arguments.get("ids") is not None:
            raise ValueError("'id' and 'ids' are invalid for fact action 'add'.")

        raw = self._get_fact_inputs(arguments, action="add")
        prepared: list[tuple[str, tuple[Citation, ...]]] = []
        for index, item in enumerate(raw):
            if item.get("working_state_id") is not None:
                raise ValueError(
                    f"facts[{index}]: 'working_state_id' is invalid for add."
                )
            content = str(item.get("content") or "").strip()
            if not content:
                raise ValueError(f"facts[{index}]: 'content' cannot be empty.")
            citations = self._prepare_citations(item.get("citations", []))
            prepared.append((content, citations))

        added = [
            self._append(content, citations, working_state_id=None)
            for content, citations in prepared
        ]
        if len(added) == 1:
            fact = added[0]
            return f"Added FACT [{fact.id}]: {fact.content}"
        return (
            f"Added {len(added)} FACTs "
            f"{self._format_ids([fact.id for fact in added])}."
        )

    def _promote(self, arguments: dict[str, Any]) -> str:
        if arguments.get("id") is not None or arguments.get("ids") is not None:
            raise ValueError("'id' and 'ids' are invalid for fact promotion.")

        raw = self._get_fact_inputs(arguments, action="promote")
        prepared: list[tuple[int, str, tuple[Citation, ...]]] = []
        for index, item in enumerate(raw):
            working_id = item.get("working_state_id")
            if working_id is None:
                raise ValueError(
                    f"facts[{index}]: 'working_state_id' is required for promote."
                )
            working = self.require_working_state(working_id)
            content_raw = item.get("content")
            content = (
                str(content_raw).strip()
                if content_raw is not None
                else working.content
            )
            if not content:
                raise ValueError(f"facts[{index}]: content cannot be empty.")
            citations = self._prepare_citations(item.get("citations", []))
            prepared.append((working_id, content, citations))

        added: list[FactExtract] = []
        for working_id, content, citations in prepared:
            fact = self._append(
                content,
                citations,
                working_state_id=working_id,
            )
            self.register_promotion(
                working_id,
                kind="fact",
                memory_id=fact.id,
            )
            added.append(fact)

        if len(added) == 1:
            fact = added[0]
            return (
                f"Promoted working state [W{fact.working_state_id}] to "
                f"FACT [{fact.id}]: {fact.content}"
            )
        return (
            f"Promoted {len(added)} working states to FACT entries "
            f"{self._format_ids([fact.id for fact in added])}."
        )

    def _remove(self, arguments: dict[str, Any]) -> str:
        invalid = [
            name
            for name in ("working_state_id", "content", "citations", "facts")
            if arguments.get(name) is not None
        ]
        if invalid:
            raise ValueError(
                ", ".join(f"'{name}'" for name in invalid)
                + " are invalid for fact action 'remove'."
            )
        ids = self._get_ids(arguments)
        selected = [self.__extracts[self._find_index(fact_id)] for fact_id in ids]
        id_set = set(ids)
        self.__extracts = [fact for fact in self.__extracts if fact.id not in id_set]
        for fact in selected:
            if fact.working_state_id is not None:
                self.unregister_promotion(
                    fact.working_state_id,
                    kind="fact",
                    memory_id=fact.id,
                )
        if len(selected) == 1:
            fact = selected[0]
            return f"Removed FACT [{fact.id}]: {fact.content}"
        return f"Removed {len(selected)} FACT entries {self._format_ids(ids)}."

    def _append(
        self,
        content: str,
        citations: tuple[Citation, ...],
        *,
        working_state_id: int | None,
    ) -> FactExtract:
        fact = FactExtract(
            id=self.__next_id,
            content=content,
            working_state_id=working_state_id,
            citations=citations,
        )
        self.__next_id += 1
        self.__extracts.append(fact)
        return fact

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        action = arguments.get("action", "?")
        parts = [f"action={action}"]
        if arguments.get("working_state_id") is not None:
            parts.append(f"working=W{arguments['working_state_id']}")
        if arguments.get("content") is not None:
            parts.append(f"content={self._truncate(str(arguments['content']))}")
        elif arguments.get("facts") is not None:
            parts.append(f"batch={len(arguments['facts'])}")
        if arguments.get("citations"):
            parts.append(f"citations={len(arguments['citations'])}")
        ids = self._ids_summary(arguments)
        if ids:
            parts.append(f"ids={ids}")
        return " | ".join(parts)

    @staticmethod
    def _get_fact_inputs(
        arguments: dict[str, Any],
        *,
        action: str,
    ) -> list[dict[str, Any]]:
        working_id = arguments.get("working_state_id")
        content = arguments.get("content")
        citations = arguments.get("citations")
        facts = arguments.get("facts")
        if facts is not None and any(
            value is not None for value in (working_id, content, citations)
        ):
            raise ValueError(
                "Use either single-fact fields or 'facts', not both."
            )
        if facts is not None:
            if not facts:
                raise ValueError("'facts' cannot be empty.")
            return list(facts)
        if action == "promote" and working_id is None:
            raise ValueError("'working_state_id' or 'facts' is required for promote.")
        if action == "add" and content is None:
            raise ValueError("'content' or 'facts' is required for add.")
        return [
            {
                "working_state_id": working_id,
                "content": content,
                "citations": citations or [],
            }
        ]

    @classmethod
    def _prepare_citations(
        cls,
        raw_citations: list[dict[str, Any]],
    ) -> tuple[Citation, ...]:
        cls._validate_citations(raw_citations)
        return tuple(
            Citation(
                type=citation["type"],
                source=citation["source"].strip(),
                line=citation.get("line"),
                end_line=citation.get("end_line"),
                reference=(
                    citation.get("reference").strip()
                    if citation.get("reference") is not None
                    else None
                ),
            )
            for citation in raw_citations
        )

    @staticmethod
    def _get_ids(arguments: dict[str, Any]) -> list[int]:
        single = arguments.get("id")
        multiple = arguments.get("ids")
        if single is not None and multiple is not None:
            raise ValueError("Use either 'id' or 'ids', not both.")
        ids = [single] if single is not None else multiple
        if not ids:
            raise ValueError("'id' or 'ids' is required for fact removal.")
        if len(ids) != len(set(ids)):
            raise ValueError("Fact IDs cannot contain duplicates.")
        return list(ids)

    def _find_index(self, fact_id: int) -> int:
        for index, fact in enumerate(self.__extracts):
            if fact.id == fact_id:
                return index
        raise ValueError(f"FACT [{fact_id}] does not exist.")

    @staticmethod
    def _validate_citations(citations: list[dict[str, Any]]) -> None:
        for index, citation in enumerate(citations):
            citation_type = citation["type"]
            source = citation.get("source")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"citations[{index}]: 'source' cannot be empty.")
            reference = citation.get("reference")
            if reference is not None and (
                not isinstance(reference, str) or not reference.strip()
            ):
                raise ValueError(f"citations[{index}]: 'reference' cannot be empty.")
            line = citation.get("line")
            end_line = citation.get("end_line")
            if citation_type == "file":
                if line is not None and line < 1:
                    raise ValueError(f"citations[{index}]: 'line' must be at least 1.")
                if end_line is not None and line is None:
                    raise ValueError(f"citations[{index}]: 'end_line' requires 'line'.")
                if line is not None and end_line is not None and end_line < line:
                    raise ValueError(
                        f"citations[{index}]: 'end_line' cannot be before 'line'."
                    )
            elif citation_type == "url":
                if line is not None or end_line is not None:
                    raise ValueError(
                        f"citations[{index}]: 'line' and 'end_line' are only "
                        "valid for file citations."
                    )
            else:
                raise ValueError(
                    f"citations[{index}]: unsupported citation type {citation_type!r}."
                )

    @staticmethod
    def _ids_summary(arguments: dict[str, Any]) -> str | None:
        single = arguments.get("id")
        multiple = arguments.get("ids")
        if single is not None:
            return f"[{single}]"
        if multiple is not None:
            return FactTool._format_ids(list(multiple))
        return None

    @staticmethod
    def _format_ids(ids: list[int]) -> str:
        return "[" + ", ".join(str(item_id) for item_id in ids) + "]"

    @staticmethod
    def _truncate(value: str) -> str:
        return value if len(value) <= 80 else value[:80] + "..."
