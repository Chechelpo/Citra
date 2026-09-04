"""Append-oriented verification evidence for requirements and criteria."""

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
from .issue_tool import IssueTool
from .memory_tool import MemoryTool
from .requirement_tool import RequirementTool

if TYPE_CHECKING:
    from citra.agent import AgentSession
    from citra.context import ExecutionContext


_VERIFICATION_STATUSES = ("passed", "failed", "blocked")


@dataclass(frozen=True)
class VerificationExtract:
    """Represent one reproducible verification result and its trace links."""

    id: int
    check: str
    status: str
    evidence: str
    command: str | None = None
    requirement_ids: tuple[int, ...] = ()
    acceptance_criterion_ids: tuple[int, ...] = ()
    change_ids: tuple[int, ...] = ()
    change_revisions: tuple[int, ...] = ()
    issue_ids: tuple[int, ...] = ()
    active: bool = True
    invalidation_reason: str | None = None


class VerificationTool(MemoryTool[VerificationExtract]):
    """Record test and inspection evidence without adjudicating acceptance."""

    TOOL_ID = "verification"
    CAPABILITIES = ToolCapabilities(actions=("record", "invalidate", "remove"))

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="verification",
            description=(
                "Record reproducible test, build, lint, type-check, runtime, or "
                "inspection evidence. Link results to requirement, acceptance-"
                "criterion, implemented-change, and issue IDs. Change revisions "
                "are captured automatically so later edits make old evidence "
                "detectably stale. Record new evidence instead of rewriting "
                "history; invalidate stale results or supersede them with a newer "
                "record. A verification result is evidence for the reviewer, not "
                "acceptance adjudication."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Verification-memory operation.",
                            enum=("record", "invalidate", "remove"),
                        ),
                    ),
                    JsonProperty(
                        name="id",
                        schema=JsonSchema.integer(
                            description="Single verification ID to invalidate/remove."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="Verification IDs to invalidate/remove.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="check",
                        schema=JsonSchema.string(
                            description="Behavior or quality property checked."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="status",
                        schema=JsonSchema.string(
                            description="Observed result.",
                            enum=_VERIFICATION_STATUSES,
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="evidence",
                        schema=JsonSchema.string(
                            description=(
                                "Exact outcome, relevant output, or inspection basis."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="command",
                        schema=JsonSchema.string(
                            description=(
                                "Exact command used, when the check was executable."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="requirement_ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="Requirement IDs evidenced by this result.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="acceptance_criterion_ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description=(
                                "Acceptance-criterion IDs evidenced by this result."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="change_ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description=(
                                "Implemented-change IDs exercised by this result."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="issue_ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description="Issue IDs reproduced or retested.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="supersedes_ids",
                        schema=JsonSchema.array(
                            items=JsonSchema.integer(),
                            description=(
                                "Older active verification IDs made stale by "
                                "this new result."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="reason",
                        schema=JsonSchema.string(
                            description="Why selected evidence is now stale."
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
        """Return the single model-facing verification definition."""
        del context
        return (ToolDefinition(definition=cls.DEFINITION),)

    def __init__(
        self,
        context: ExecutionContext,
        session: AgentSession,
    ) -> None:
        """Initialize empty verification memory for the conversation."""
        super().__init__(context=context, session=session)
        self._extracts: list[VerificationExtract] = []
        self._next_id = 1

    @property
    @override
    def heading(self) -> str:
        """Return the memory-section heading."""
        return "Verification Evidence"

    @override
    def get_extracts(self) -> list[VerificationExtract]:
        """Return a defensive copy of recorded verification results."""
        return list(self._extracts)

    @override
    def format_extract(self, extract: VerificationExtract) -> str:
        """Render one compact verification result for implementation review."""
        outdated = extract.active and self._has_outdated_change_refs(extract)
        freshness = (
            "STALE-CHANGE"
            if outdated
            else "ACTIVE" if extract.active else "STALE"
        )
        lines = [
            f"- [{extract.status.upper()}/{freshness}] [V{extract.id}] "
            f"{extract.check}: {extract.evidence}"
        ]
        if extract.command:
            lines.append(f"  - command: {extract.command}")
        coverage = self._format_coverage(extract)
        if coverage:
            lines.append(f"  - covers: {coverage}")
        if extract.invalidation_reason:
            lines.append(f"  - invalidated: {extract.invalidation_reason}")
        return "\n".join(lines)

    @override
    def should_offer_documentation(self) -> bool:
        """Keep run-specific verification evidence out of durable docs."""
        return False

    def has_blocking_results(self) -> bool:
        """Return whether active evidence failed, blocked, or targets old code."""
        failed = any(
            item.active and item.status in {"failed", "blocked"}
            for item in self._extracts
        )
        outdated = any(
            item.active and self._has_outdated_change_refs(item)
            for item in self._extracts
        )
        if failed or outdated:
            self._logger().warning(
                "Verification evidence blocks completion",
                extra={
                    "origin": type(self).__module__,
                    "failed_or_blocked": failed,
                    "outdated_change_evidence": outdated,
                },
            )
        return failed or outdated

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        """Dispatch one validated verification-memory action."""
        action = arguments["action"]
        if action == "record":
            return self._record(arguments)
        if action == "invalidate":
            return self._invalidate(arguments)
        if action == "remove":
            return self._remove(arguments)
        raise ValueError(f"Unsupported verification action: {action}")

    def _record(self, arguments: dict[str, Any]) -> str:
        """Append one result and atomically supersede selected stale evidence."""
        self._reject(arguments, ("id", "ids", "reason"), action="record")
        check = self._required_text(arguments.get("check"), field="check")
        status = str(arguments.get("status") or "").strip()
        if status not in _VERIFICATION_STATUSES:
            raise ValueError(
                "'status' must be one of: "
                + ", ".join(_VERIFICATION_STATUSES)
                + "."
            )
        evidence = self._required_text(arguments.get("evidence"), field="evidence")
        command = self._optional_text(arguments.get("command"), field="command")
        (
            requirement_ids,
            criterion_ids,
            change_ids,
            change_revisions,
            issue_ids,
        ) = self._references(arguments)
        supersedes = self.normalize_reference_ids(
            arguments.get("supersedes_ids"),
            field="supersedes_ids",
        )
        for verification_id in supersedes:
            index = self._find_index(verification_id)
            if not self._extracts[index].active:
                raise ValueError(
                    f"VERIFICATION [V{verification_id}] is already stale."
                )

        item = VerificationExtract(
            id=self._next_id,
            check=check,
            status=status,
            evidence=evidence,
            command=command,
            requirement_ids=requirement_ids,
            acceptance_criterion_ids=criterion_ids,
            change_ids=change_ids,
            change_revisions=change_revisions,
            issue_ids=issue_ids,
        )
        self._next_id += 1
        self._extracts.append(item)
        for verification_id in supersedes:
            index = self._find_index(verification_id)
            self._extracts[index] = replace(
                self._extracts[index],
                active=False,
                invalidation_reason=f"Superseded by V{item.id}",
            )
        return f"Recorded VERIFICATION [V{item.id}] as {item.status}."

    def _invalidate(self, arguments: dict[str, Any]) -> str:
        """Mark selected evidence stale while retaining its audit history."""
        self._reject(
            arguments,
            (
                "check",
                "status",
                "evidence",
                "command",
                "requirement_ids",
                "acceptance_criterion_ids",
                "change_ids",
                "issue_ids",
                "supersedes_ids",
            ),
            action="invalidate",
        )
        ids = self._selected_ids(arguments, action="invalidate")
        reason = self._required_text(arguments.get("reason"), field="reason")
        for verification_id in ids:
            index = self._find_index(verification_id)
            self._extracts[index] = replace(
                self._extracts[index],
                active=False,
                invalidation_reason=reason,
            )
        return "Invalidated verifications " + self._format_ids(ids) + "."

    def _remove(self, arguments: dict[str, Any]) -> str:
        """Remove evidence created in error rather than evidence merely stale."""
        self._reject(
            arguments,
            (
                "check",
                "status",
                "evidence",
                "command",
                "requirement_ids",
                "acceptance_criterion_ids",
                "change_ids",
                "issue_ids",
                "supersedes_ids",
                "reason",
            ),
            action="remove",
        )
        ids = self._selected_ids(arguments, action="remove")
        for verification_id in ids:
            self._find_index(verification_id)
        selected = set(ids)
        self._extracts = [item for item in self._extracts if item.id not in selected]
        return "Removed verifications " + self._format_ids(ids) + "."

    def _references(
        self,
        arguments: dict[str, Any],
    ) -> tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        """Normalize and validate every typed artifact evidenced by a result."""
        requirement_ids = self.normalize_reference_ids(
            arguments.get("requirement_ids"),
            field="requirement_ids",
        )
        criterion_ids = self.normalize_reference_ids(
            arguments.get("acceptance_criterion_ids"),
            field="acceptance_criterion_ids",
        )
        change_ids = self.normalize_reference_ids(
            arguments.get("change_ids"),
            field="change_ids",
        )
        issue_ids = self.normalize_reference_ids(
            arguments.get("issue_ids"),
            field="issue_ids",
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
        self.require_memory_ids(IssueTool, issue_ids, field="issue_ids")
        change_revisions = self._change_revisions(change_ids)
        return (
            requirement_ids,
            criterion_ids,
            change_ids,
            change_revisions,
            issue_ids,
        )

    def _change_revisions(self, change_ids: tuple[int, ...]) -> tuple[int, ...]:
        """Snapshot current change revisions for later staleness detection."""
        if not change_ids:
            return ()
        service = self.session.memory.get(ChangeTool.TOOL_ID)
        if not isinstance(service, ChangeTool):
            raise ValueError(
                "Cannot snapshot change revisions without change memory."
            )
        revisions = {item.id: item.revision for item in service.get_extracts()}
        return tuple(revisions[item] for item in change_ids)

    def _has_outdated_change_refs(self, extract: VerificationExtract) -> bool:
        """Return whether a referenced change has advanced or disappeared."""
        if not extract.change_ids:
            return False
        service = self.session.memory.get(ChangeTool.TOOL_ID)
        if not isinstance(service, ChangeTool):
            return True
        current = {item.id: item.revision for item in service.get_extracts()}
        return any(
            current.get(change_id) != revision
            for change_id, revision in zip(
                extract.change_ids,
                extract.change_revisions,
                strict=True,
            )
        )

    def _find_index(self, verification_id: int) -> int:
        """Return the list index for an existing verification ID."""
        for index, item in enumerate(self._extracts):
            if item.id == verification_id:
                return index
        raise ValueError(f"VERIFICATION [V{verification_id}] does not exist.")

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
    def _selected_ids(
        arguments: dict[str, Any],
        *,
        action: str,
    ) -> tuple[int, ...]:
        """Normalize one-or-many verification IDs for a lifecycle action."""
        single = arguments.get("id")
        multiple = arguments.get("ids")
        if single is not None and multiple is not None:
            raise ValueError("Use either 'id' or 'ids', not both.")
        values = [single] if single is not None else multiple
        if not values:
            raise ValueError(f"'id' or 'ids' is required for {action}.")
        if len(values) != len(set(values)):
            raise ValueError("Verification IDs cannot contain duplicates.")
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
                f"{rendered} are invalid for verification action {action!r}."
            )

    @staticmethod
    def _format_ids(ids: tuple[int, ...]) -> str:
        """Format verification IDs for concise results."""
        return "[" + ", ".join(f"V{item}" for item in ids) + "]"

    @staticmethod
    def _format_coverage(extract: VerificationExtract) -> str:
        """Format typed requirement, criterion, and issue references."""
        refs = [f"R{item}" for item in extract.requirement_ids]
        refs.extend(f"A{item}" for item in extract.acceptance_criterion_ids)
        refs.extend(
            f"CH{change_id}@r{revision}"
            for change_id, revision in zip(
                extract.change_ids,
                extract.change_revisions,
                strict=True,
            )
        )
        refs.extend(f"I{item}" for item in extract.issue_ids)
        return ", ".join(refs)

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        """Render non-sensitive verification operation metadata for logs."""
        action = str(arguments.get("action", "?"))
        status = arguments.get("status")
        suffix = f" | status={status}" if status is not None else ""
        return f"action={action}{suffix}"


__all__ = ["VerificationExtract", "VerificationTool"]
