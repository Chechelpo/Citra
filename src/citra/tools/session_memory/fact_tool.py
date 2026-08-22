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
    citations: tuple[Citation, ...] = ()


class FactTool(MemoryTool[FactExtract]):
    """
    Manage facts retained for the current conversation lifecycle.

    Actions:
    - add: record one fact or a batch of facts with optional citations
    - remove: remove one or more stale, incorrect, or invalid facts
    """

    CITATION_SCHEMA = JsonSchema.object(
        properties=(
            JsonProperty(
                name="type",
                schema=JsonSchema.string(
                    description="Citation source type.",
                    enum=(
                        "file",
                        "url",
                    ),
                ),
            ),
            JsonProperty(
                name="source",
                schema=JsonSchema.string(
                    description=(
                        "Workspace-relative file path "
                        "or source URL."
                    ),
                ),
            ),
            JsonProperty(
                name="line",
                schema=JsonSchema.integer(
                    description=(
                        "Starting line number for a "
                        "file citation."
                    ),
                ),
                required=False,
            ),
            JsonProperty(
                name="end_line",
                schema=JsonSchema.integer(
                    description=(
                        "Ending line number for a "
                        "file citation."
                    ),
                ),
                required=False,
            ),
            JsonProperty(
                name="reference",
                schema=JsonSchema.string(
                    description=(
                        "Optional locator within the source. "
                        "For files, use this for a symbol, heading, "
                        "field, section, or other location when exact "
                        "line numbers are unavailable. "
                        "For URLs, use this for a section, heading, "
                        "anchor, or fragment."
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
                schema=JsonSchema.string(
                    description="Fact to remember.",
                ),
            ),
            JsonProperty(
                name="citations",
                schema=JsonSchema.array(
                    CITATION_SCHEMA,
                    description=(
                        "Optional supporting citations for this fact."
                    ),
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
                "Manage facts retained for the current conversation. "
                "Operations may target one fact or a batch. "
                "Use 'add' to remember verified facts, optionally with "
                "supporting file or URL citations. Use 'remove' when "
                "remembered facts are stale, incorrect, superseded, "
                "or no longer applicable."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Fact operation to perform.",
                            enum=(
                                "add",
                                "remove",
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description=(
                                "Single fact to remember for 'add'. "
                                "Use 'facts' to add multiple facts."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="citations",
                        schema=JsonSchema.array(
                            CITATION_SCHEMA,
                            description=(
                                "Supporting citations for a single "
                                "fact added with 'content'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="facts",
                        schema=JsonSchema.array(
                            FACT_SCHEMA,
                            description=(
                                "Facts to add as a batch. Each fact "
                                "may have its own citations."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description=(
                                "Single fact ID for 'remove'. "
                                "Use 'ids' to remove multiple facts."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            JsonSchema.integer(),
                            description=(
                                "Fact IDs to remove as a batch."
                            ),
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
    def heading(
        self,
    ) -> str:
        return "Facts"

    @override
    def get_extracts(
        self,
    ) -> list[FactExtract]:
        return list(
            self.__extracts
        )

    @override
    def format_extract(
        self,
        extract: FactExtract,
    ) -> str:
        lines = [
            f"- [{extract.id}] {extract.content}",
        ]

        for citation in extract.citations:
            if citation.type == "file":
                location = citation.source

                if citation.line is not None:
                    location += (
                        f":{citation.line}"
                    )

                    if (
                        citation.end_line is not None
                        and citation.end_line != citation.line
                    ):
                        location += (
                            f"-{citation.end_line}"
                        )

                lines.append(
                    f"  - source: {location}"
                )

            elif citation.type == "url":
                location = citation.source

                if citation.reference:
                    location += (
                        f" ({citation.reference})"
                    )

                lines.append(
                    f"  - source: {location}"
                )

        return "\n".join(
            lines
        )

    @override
    def should_offer_documentation(
        self,
    ) -> bool:
        return False

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        action = arguments["action"]

        if action == "add":
            return self._add(
                arguments
            )

        if action == "remove":
            return self._remove(
                arguments
            )

        raise ValueError(
            f"Unsupported fact action: {action}"
        )

    def _add(
        self,
        arguments: dict[str, Any],
    ) -> str:
        raw_facts = self._get_facts(
            arguments
        )

        # Validate the entire batch before mutating state.
        prepared: list[
            tuple[
                str,
                tuple[Citation, ...],
            ]
        ] = []

        for index, raw_fact in enumerate(
            raw_facts
        ):
            content = raw_fact[
                "content"
            ].strip()

            if not content:
                raise ValueError(
                    f"facts[{index}]: "
                    "'content' cannot be empty."
                )

            raw_citations = raw_fact.get(
                "citations",
                [],
            )

            self._validate_citations(
                raw_citations
            )

            citations = tuple(
                Citation(
                    type=citation["type"],
                    source=citation["source"],
                    line=citation.get("line"),
                    end_line=citation.get("end_line"),
                    reference=citation.get("reference"),
                )
                for citation in raw_citations
            )

            prepared.append(
                (
                    content,
                    citations,
                )
            )

        added: list[FactExtract] = []

        for content, citations in prepared:
            fact = FactExtract(
                id=self.__next_id,
                content=content,
                citations=citations,
            )

            self.__next_id += 1

            self.__extracts.append(
                fact
            )

            added.append(
                fact
            )

        if len(added) == 1:
            fact = added[0]

            return (
                f"Added FACT [{fact.id}]: "
                f"{fact.content}"
            )

        return (
            f"Added {len(added)} FACTs "
            f"[{added[0].id}-{added[-1].id}]."
        )

    def _remove(
        self,
        arguments: dict[str, Any],
    ) -> str:
        ids = self._get_ids(
            arguments
        )

        # Resolve everything first so an invalid ID cannot
        # partially apply a batch removal.
        facts = [
            self.__extracts[
                self._find_index(
                    fact_id
                )
            ]
            for fact_id in ids
        ]

        ids_set = set(
            ids
        )

        self.__extracts = [
            fact
            for fact in self.__extracts
            if fact.id not in ids_set
        ]

        if len(facts) == 1:
            fact = facts[0]

            return (
                f"Removed FACT [{fact.id}]: "
                f"{fact.content}"
            )

        return (
            f"Removed {len(facts)} FACTs "
            f"{self._format_ids(ids)}."
        )

    @staticmethod
    def _get_facts(
        arguments: dict[str, Any],
    ) -> list[dict[str, Any]]:
        content = arguments.get(
            "content"
        )
        citations = arguments.get(
            "citations"
        )
        facts = arguments.get(
            "facts"
        )

        if (
            facts is not None
            and content is not None
        ):
            raise ValueError(
                "Use either 'content' or 'facts', not both."
            )

        if (
            facts is not None
            and citations is not None
        ):
            raise ValueError(
                "'citations' is only valid with single-fact "
                "'content'. Batch facts carry their own citations."
            )

        if facts is not None:
            if not facts:
                raise ValueError(
                    "'facts' cannot be empty."
                )

            return list(
                facts
            )

        if content is None:
            raise ValueError(
                "'content' or 'facts' is required "
                "for fact action 'add'."
            )

        return [
            {
                "content": content,
                "citations": citations or [],
            }
        ]

    @staticmethod
    def _get_ids(
        arguments: dict[str, Any],
    ) -> list[int]:
        fact_id = arguments.get(
            "id"
        )
        ids = arguments.get(
            "ids"
        )

        if (
            fact_id is not None
            and ids is not None
        ):
            raise ValueError(
                "Use either 'id' or 'ids', not both."
            )

        if fact_id is not None:
            ids = [
                fact_id
            ]

        if not ids:
            raise ValueError(
                "'id' or 'ids' is required "
                "for fact action 'remove'."
            )

        if len(ids) != len(set(ids)):
            raise ValueError(
                "'ids' cannot contain duplicates."
            )

        return list(
            ids
        )

    def _find_index(
        self,
        fact_id: int,
    ) -> int:
        for index, fact in enumerate(
            self.__extracts
        ):
            if fact.id == fact_id:
                return index

        raise ValueError(
            f"Fact [{fact_id}] does not exist."
        )

    @staticmethod
    def _format_ids(
        ids: list[int],
    ) -> str:
        return (
            "["
            + ", ".join(
                str(fact_id)
                for fact_id in ids
            )
            + "]"
        )

    @staticmethod
    def _validate_citations(
        citations: list[dict[str, Any]],
    ) -> None:
        for index, citation in enumerate(
            citations
        ):
            citation_type = citation["type"]

            source = citation.get("source")

            if (
                not isinstance(source, str)
                or not source.strip()
            ):
                raise ValueError(
                    f"citations[{index}]: "
                    "'source' cannot be empty."
                )

            reference = citation.get(
                "reference"
            )

            if (
                reference is not None
                and (
                    not isinstance(reference, str)
                    or not reference.strip()
                )
            ):
                raise ValueError(
                    f"citations[{index}]: "
                    "'reference' cannot be empty."
                )

            line = citation.get(
                "line"
            )
            end_line = citation.get(
                "end_line"
            )

            if citation_type == "file":
                if (
                    line is not None
                    and line < 1
                ):
                    raise ValueError(
                        f"citations[{index}]: "
                        "'line' must be at least 1."
                    )

                if (
                    end_line is not None
                    and line is None
                ):
                    raise ValueError(
                        f"citations[{index}]: "
                        "'end_line' requires 'line'."
                    )

                if (
                    line is not None
                    and end_line is not None
                    and end_line < line
                ):
                    raise ValueError(
                        f"citations[{index}]: "
                        "'end_line' cannot be before 'line'."
                    )

            elif citation_type == "url":
                if (
                    line is not None
                    or end_line is not None
                ):
                    raise ValueError(
                        f"citations[{index}]: "
                        "'line' and 'end_line' are only "
                        "valid for file citations."
                    )

            else:
                raise ValueError(
                    f"citations[{index}]: "
                    f"unsupported citation type "
                    f"{citation_type!r}."
                )
