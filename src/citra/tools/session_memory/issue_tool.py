"""Lifecycle memory for risks, defects, and workflow gaps."""

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
from .acceptance_criteria_tool import AcceptanceCriteriaTool
from .change_tool import ChangeTool
from .memory_tool import MemoryTool
from .requirement_tool import RequirementTool

if TYPE_CHECKING:
    from citra.agent import AgentSession
    from citra.context import ExecutionContext


_ISSUE_KINDS = ("risk", "defect", "requirement_gap", "plan_gap", "test_gap")
_SEVERITIES = ("low", "medium", "high", "critical")
_ROUTES = ("explore", "plan", "implement", "test")


@dataclass(frozen=True)
class IssueExtract:
    """Represent one routed risk, defect, or workflow gap."""

    id: int
    kind: str
    severity: str
    content: str
    route: str
    blocking: bool = True
    status: str = "open"
    evidence: str | None = None
    resolution: str | None = None
    requirement_ids: tuple[int, ...] = ()
    acceptance_criterion_ids: tuple[int, ...] = ()
    change_ids: tuple[int, ...] = ()


class IssueTool(MemoryTool[IssueExtract]):
    """Track actionable findings across feedback loops until resolved."""

    TOOL_ID = "issue"
    CAPABILITIES = ToolCapabilities(
        actions=("add", "update", "resolve", "reopen", "remove"),
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="issue",
            description=(
                "Manage routed workflow findings. Record risks, implementation "
                "defects, requirement gaps, plan gaps, or verification gaps; "
                "link them to requirements, acceptance criteria, and implemented "
                "changes; resolve them only with concrete correction evidence. "
                "Open blocking issues prevent workflow completion."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Issue lifecycle operation.",
                            enum=("add", "update", "resolve", "reopen", "remove"),
                        ),
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description="Issue ID for update, resolve, or reopen."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="Issue IDs for batch removal.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="kind",
                        schema=JsonSchema.string(
                            description="Finding category.",
                            enum=_ISSUE_KINDS,
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="severity",
                        schema=JsonSchema.string(
                            description="Consequence if the issue remains.",
                            enum=_SEVERITIES,
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description="Concise observable problem or risk."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="route",
                        schema=JsonSchema.string(
                            description=(
                                "Earliest workflow phase capable of correction."
                            ),
                            enum=_ROUTES,
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="blocking",
                        schema=JsonSchema.boolean(
                            description=(
                                "Whether the open issue must block completion."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="evidence",
                        schema=JsonSchema.string(
                            description=(
                                "Observed failure, reproduction, or risk basis."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="resolution",
                        schema=JsonSchema.string(
                            description=(
                                "Correction evidence for resolve, or invalidation "
                                "reason for reopen."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="requirement_ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="Affected requirement IDs.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="acceptance_criterion_ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="Affected acceptance-criterion IDs.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="change_ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="Implemented-change IDs related to the issue.",
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
        """Return the single model-facing issue definition."""
        del context
        return (ToolDefinition(definition=cls.DEFINITION),)

    def __init__(
        self,
        context: ExecutionContext,
        session: AgentSession,
    ) -> None:
        """Initialize empty issue memory for the active conversation."""
        super().__init__(context=context, session=session)
        self._extracts: list[IssueExtract] = []
        self._next_id = 1

    @property
    @override
    def heading(self) -> str:
        """Return the memory-section heading."""
        return "Issues and Risks"

    @override
    def get_extracts(self) -> list[IssueExtract]:
        """Return a defensive copy of issue records."""
        return list(self._extracts)

    @override
    def format_extract(self, extract: IssueExtract) -> str:
        """Render one issue with routing and traceability metadata."""
        gate = "BLOCKING" if extract.blocking else "ADVISORY"
        details = [
            f"- [{extract.status.upper()}] [I{extract.id}] "
            f"{extract.kind}/{extract.severity}/{gate} -> {extract.route}: "
            f"{extract.content}"
        ]
        refs = self._format_coverage(extract)
        if refs:
            details.append(f"  - affects: {refs}")
        if extract.evidence:
            details.append(f"  - evidence: {extract.evidence}")
        if extract.resolution:
            details.append(f"  - resolution: {extract.resolution}")
        return "\n".join(details)

    @override
    def should_offer_documentation(self) -> bool:
        """Offer unresolved risks, but not transient corrected defects."""
        return any(
            item.kind == "risk" and item.status == "open"
            for item in self._extracts
        )

    def has_blocking_issues(self) -> bool:
        """Return whether an unresolved blocking issue remains."""
        return any(
            item.status == "open" and item.blocking
            for item in self._extracts
        )

    def required_route(self) -> str | None:
        """Return the earliest correction phase among open blocking issues."""
        priorities = {phase: index for index, phase in enumerate(_ROUTES)}
        routes = (
            item.route
            for item in self._extracts
            if item.status == "open" and item.blocking
        )
        return min(routes, key=priorities.__getitem__, default=None)

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        """Dispatch one validated issue-lifecycle action."""
        action = arguments["action"]
        if action == "add":
            return self._add(arguments)
        if action == "update":
            return self._update(arguments)
        if action == "resolve":
            return self._resolve(arguments)
        if action == "reopen":
            return self._reopen(arguments)
        if action == "remove":
            return self._remove(arguments)
        raise ValueError(f"Unsupported issue action: {action}")

    def _add(self, arguments: dict[str, Any]) -> str:
        """Add one open issue with its correction owner and evidence."""
        self._reject(arguments, ("id", "ids", "resolution"), action="add")
        kind = self._required_choice(arguments.get("kind"), "kind", _ISSUE_KINDS)
        severity = self._required_choice(
            arguments.get("severity"),
            "severity",
            _SEVERITIES,
        )
        route = self._required_choice(arguments.get("route"), "route", _ROUTES)
        content = self._required_text(arguments.get("content"), field="content")
        evidence = self._optional_text(arguments.get("evidence"), field="evidence")
        requirement_ids, criterion_ids, change_ids = self._references(arguments)
        item = IssueExtract(
            id=self._next_id,
            kind=kind,
            severity=severity,
            content=content,
            route=route,
            blocking=bool(arguments.get("blocking", True)),
            evidence=evidence,
            requirement_ids=requirement_ids,
            acceptance_criterion_ids=criterion_ids,
            change_ids=change_ids,
        )
        self._next_id += 1
        self._extracts.append(item)
        return f"Added ISSUE [I{item.id}] routed to {item.route}: {item.content}"

    def _update(self, arguments: dict[str, Any]) -> str:
        """Revise an issue and reopen it because prior resolution is stale."""
        self._reject(arguments, ("ids", "resolution"), action="update")
        issue_id = self._required_id(arguments, action="update")
        mutable = (
            "kind",
            "severity",
            "content",
            "route",
            "blocking",
            "evidence",
            "requirement_ids",
            "acceptance_criterion_ids",
            "change_ids",
        )
        if not any(arguments.get(field) is not None for field in mutable):
            raise ValueError("Issue update requires at least one changed field.")
        index = self._find_index(issue_id)
        current = self._extracts[index]
        requirement_ids, criterion_ids, change_ids = self._references(
            arguments,
            current=current,
        )
        updated = replace(
            current,
            kind=(
                current.kind
                if arguments.get("kind") is None
                else self._required_choice(arguments["kind"], "kind", _ISSUE_KINDS)
            ),
            severity=(
                current.severity
                if arguments.get("severity") is None
                else self._required_choice(
                    arguments["severity"],
                    "severity",
                    _SEVERITIES,
                )
            ),
            content=(
                current.content
                if arguments.get("content") is None
                else self._required_text(arguments["content"], field="content")
            ),
            route=(
                current.route
                if arguments.get("route") is None
                else self._required_choice(arguments["route"], "route", _ROUTES)
            ),
            blocking=(
                current.blocking
                if arguments.get("blocking") is None
                else bool(arguments["blocking"])
            ),
            evidence=(
                current.evidence
                if arguments.get("evidence") is None
                else self._optional_text(arguments["evidence"], field="evidence")
            ),
            status="open",
            resolution=None,
            requirement_ids=requirement_ids,
            acceptance_criterion_ids=criterion_ids,
            change_ids=change_ids,
        )
        self._extracts[index] = updated
        return f"Updated and reopened ISSUE [I{updated.id}]."

    def _resolve(self, arguments: dict[str, Any]) -> str:
        """Resolve one issue with concrete correction evidence."""
        self._reject(
            arguments,
            (
                "ids",
                "kind",
                "severity",
                "content",
                "route",
                "blocking",
                "evidence",
                "requirement_ids",
                "acceptance_criterion_ids",
                "change_ids",
            ),
            action="resolve",
        )
        issue_id = self._required_id(arguments, action="resolve")
        resolution = self._required_text(
            arguments.get("resolution"),
            field="resolution",
        )
        index = self._find_index(issue_id)
        self._extracts[index] = replace(
            self._extracts[index],
            status="resolved",
            resolution=resolution,
        )
        return f"Resolved ISSUE [I{issue_id}]."

    def _reopen(self, arguments: dict[str, Any]) -> str:
        """Reopen one issue when its prior correction evidence is invalid."""
        self._reject(
            arguments,
            (
                "ids",
                "kind",
                "severity",
                "content",
                "route",
                "blocking",
                "evidence",
                "requirement_ids",
                "acceptance_criterion_ids",
                "change_ids",
            ),
            action="reopen",
        )
        issue_id = self._required_id(arguments, action="reopen")
        reason = self._required_text(arguments.get("resolution"), field="resolution")
        index = self._find_index(issue_id)
        self._extracts[index] = replace(
            self._extracts[index],
            status="open",
            resolution=f"Reopened because: {reason}",
        )
        return f"Reopened ISSUE [I{issue_id}]."

    def _remove(self, arguments: dict[str, Any]) -> str:
        """Remove issue records proven obsolete or incorrectly created."""
        self._reject(
            arguments,
            (
                "kind",
                "severity",
                "content",
                "route",
                "blocking",
                "evidence",
                "resolution",
                "requirement_ids",
                "acceptance_criterion_ids",
                "change_ids",
            ),
            action="remove",
        )
        ids = self._selected_ids(arguments)
        for issue_id in ids:
            self._find_index(issue_id)
        selected = set(ids)
        self._extracts = [item for item in self._extracts if item.id not in selected]
        return "Removed issues " + self._format_ids(ids) + "."

    def _references(
        self,
        arguments: dict[str, Any],
        *,
        current: IssueExtract | None = None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Normalize and validate the task artifacts affected by an issue."""
        requirement_ids = (
            current.requirement_ids
            if current is not None and arguments.get("requirement_ids") is None
            else self.normalize_reference_ids(
                arguments.get("requirement_ids"),
                field="requirement_ids",
            )
        )
        criterion_ids = (
            current.acceptance_criterion_ids
            if current is not None
            and arguments.get("acceptance_criterion_ids") is None
            else self.normalize_reference_ids(
                arguments.get("acceptance_criterion_ids"),
                field="acceptance_criterion_ids",
            )
        )
        change_ids = (
            current.change_ids
            if current is not None and arguments.get("change_ids") is None
            else self.normalize_reference_ids(
                arguments.get("change_ids"),
                field="change_ids",
            )
        )
        self.require_memory_ids(
            RequirementTool,
            requirement_ids,
            field="requirement_ids",
        )
        self.require_memory_ids(
            AcceptanceCriteriaTool,
            criterion_ids,
            field="acceptance_criterion_ids",
        )
        self.require_memory_ids(ChangeTool, change_ids, field="change_ids")
        return requirement_ids, criterion_ids, change_ids

    def _find_index(self, issue_id: int) -> int:
        """Return the list index for an existing issue ID."""
        for index, item in enumerate(self._extracts):
            if item.id == issue_id:
                return index
        raise ValueError(f"ISSUE [I{issue_id}] does not exist.")

    @staticmethod
    def _required_choice(
        value: object,
        field: str,
        choices: tuple[str, ...],
    ) -> str:
        """Normalize one required enumerated field."""
        normalized = str(value or "").strip()
        if normalized not in choices:
            raise ValueError(
                f"'{field}' must be one of: {', '.join(choices)}."
            )
        return normalized

    @staticmethod
    def _required_text(value: object, *, field: str) -> str:
        """Normalize one required non-empty text field."""
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"'{field}' is required and cannot be empty.")
        return normalized

    @staticmethod
    def _optional_text(value: object, *, field: str) -> str | None:
        """Normalize an optional text field while rejecting empty strings."""
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"'{field}' cannot be empty.")
        return normalized

    @staticmethod
    def _required_id(arguments: dict[str, Any], *, action: str) -> int:
        """Return the single issue ID required by one lifecycle action."""
        issue_id = arguments.get("id")
        if issue_id is None:
            raise ValueError(f"'id' is required for {action}.")
        return issue_id

    @staticmethod
    def _selected_ids(arguments: dict[str, Any]) -> tuple[int, ...]:
        """Normalize one-or-many issue IDs for removal."""
        single = arguments.get("id")
        multiple = arguments.get("ids")
        if single is not None and multiple is not None:
            raise ValueError("Use either 'id' or 'ids', not both.")
        values = [single] if single is not None else multiple
        if not values:
            raise ValueError("'id' or 'ids' is required for remove.")
        if len(values) != len(set(values)):
            raise ValueError("Issue IDs cannot contain duplicates.")
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
            raise ValueError(f"{rendered} are invalid for issue action {action!r}.")

    @staticmethod
    def _format_ids(ids: tuple[int, ...]) -> str:
        """Format issue IDs for concise results."""
        return "[" + ", ".join(f"I{item}" for item in ids) + "]"

    @staticmethod
    def _format_coverage(extract: IssueExtract) -> str:
        """Format typed requirement, criterion, and change references."""
        refs = [f"R{item}" for item in extract.requirement_ids]
        refs.extend(f"A{item}" for item in extract.acceptance_criterion_ids)
        refs.extend(f"CH{item}" for item in extract.change_ids)
        return ", ".join(refs)

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        """Render non-sensitive issue operation metadata for logs."""
        action = str(arguments.get("action", "?"))
        issue_id = arguments.get("id")
        suffix = f" | id=I{issue_id}" if issue_id is not None else ""
        return f"action={action}{suffix}"


__all__ = ["IssueExtract", "IssueTool"]
