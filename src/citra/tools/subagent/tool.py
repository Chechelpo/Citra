"""
Orchestrator-facing ``subagent`` tool.

The tool exposes four actions to the orchestrator's model:

  * ``create`` — spawn a subagent for a well-defined component task;
  * ``poll`` — return the latest digest (transcript, status, pending
    guidance) for every known subagent;
  * ``steer`` — inject a user instruction into one subagent's main
    session;
  * ``cancel`` — stop one running subagent;
  * ``sleep`` — block until one or more subagents finish.

The tool delegates the actual lifecycle management to a
``SubagentSupervisor`` reachable through ``context.subagents``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

from ...context import ExecutionContext
from ..tool import Tool, ToolDefinition
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from .spec import (
    SubagentSpec,
    SubagentStatus,
)


# Hard cap on the number of transcript lines we return in a single
# ``poll`` call to keep the orchestrator's model context healthy.
_POLL_TRANSCRIPT_BUDGET = 200

# Maximum wait time honored by a single ``sleep`` call. This keeps the
# orchestrator responsive and forces the model to ``poll`` again if it
# needs to know what happened after a long delay.
_MAX_SLEEP_SECONDS = 60


def _subagent_definition() -> ChatCompletionTool:
    return ChatCompletionTool(
        function=FunctionDefinition(
            name="subagent",
            description=(
                "Delegate a well-defined component task to an isolated "
                "subagent. A subagent has its own sandboxed workspace "
                "(one writable directory and a configured set of "
                "read-only binds) and only sees the context-selected "
                "filesystem, shell, and guidance tools. "
                "Use ``create`` to spawn a subagent, ``poll`` to inspect "
                "its current transcript and any pending guidance, "
                "``steer`` to inject a correction, and ``sleep`` to "
                "block until one or more subagents finish. Use ``cancel`` "
                "to terminate a worker that should no longer continue. Subagents "
                "are intended for delegating narrow implementation "
                "work; the orchestrator remains the only entity that "
                "can write back to the user's project."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Subagent operation to perform.",
                            enum=(
                                "create",
                                "poll",
                                "steer",
                                "cancel",
                                "sleep",
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="subagent_id",
                        schema=JsonSchema.string(
                            description=(
                                "Identifier of an existing subagent. "
                                "Required for ``steer`` and ``cancel``; "
                                "optional for ``poll`` and ``sleep``."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="subagent_ids",
                        schema=JsonSchema.array(
                            JsonSchema.string(
                                description="Subagent identifier."
                            ),
                            description=(
                                "Identifiers of subagents to wait for. "
                                "Required for ``sleep`` when ``subagent_id`` "
                                "is not provided."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="task",
                        schema=JsonSchema.string(
                            description=(
                                "Clear, self-contained task description "
                                "for the subagent. Required for ``create``."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="write_path",
                        schema=JsonSchema.string(
                            description=(
                                "Path the subagent is allowed to mutate. "
                                "Required for ``create``. Relative paths "
                                "resolve against the orchestrator workspace."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="readonly_binds",
                        schema=JsonSchema.array(
                            JsonSchema.string(
                                description=(
                                    "Read-only model-facing path to expose "
                                    "inside the subagent's sandbox."
                                ),
                            ),
                            description=(
                                "Optional set of read-only host paths "
                                "the subagent is allowed to read. "
                                "Typically the source files the subagent "
                                "needs to understand its component. "
                                "Used by ``create``."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="network",
                        schema=JsonSchema.boolean(
                            description=(
                                "Allow network access for the subagent. "
                                "Defaults to false. Used by ``create``."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="system_prompt_addendum",
                        schema=JsonSchema.string(
                            description=(
                                "Additional instructions appended to "
                                "the subagent's system prompt. "
                                "Used by ``create``."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="message",
                        schema=JsonSchema.string(
                            description=(
                                "Steering message to send to the "
                                "subagent. Required for ``steer``."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="answer",
                        schema=JsonSchema.string(
                            description=(
                                "Response to a pending guidance request "
                                "from the subagent. Used by ``poll`` "
                                "(optional): when provided, the oldest "
                                "pending request for the named subagent "
                                "is answered with this text and the "
                                "request is removed from the pending "
                                "list."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="timeout_seconds",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum number of seconds to wait for "
                                "``sleep``. Defaults to 30. Maximum 60."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="include_completed",
                        schema=JsonSchema.boolean(
                            description=(
                                "For ``poll``: include subagents that have "
                                "already finished. Defaults to true."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )


class SubagentTool(Tool):
    """Model-facing tool for delegating work to subagents."""

    TOOL_ID = "subagent"

    INVALIDATES_TOOL_CACHE = True
    MAX_OUTPUT_TOKENS = 4_000

    DEFINITION = _subagent_definition()

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        super().__init__(
            context=context,
        )
        supervisor = getattr(context, "subagents", None)
        if supervisor is None:
            raise RuntimeError(
                "ExecutionContext is missing a 'subagents' supervisor; "
                "the subagent tool cannot run."
            )
        self.__supervisor: Any = supervisor

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

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action(arguments: dict[str, Any]) -> str:
        action = arguments.get("action")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action is required.")
        return action.strip()

    @staticmethod
    def _str_list(
        arguments: dict[str, Any],
        name: str,
    ) -> tuple[str, ...]:
        value = arguments.get(name)
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"{name} must be a list of strings."
            )
        result: list[str] = []
        for entry in value:
            if not isinstance(entry, str):
                raise ValueError(
                    f"{name} must contain only strings."
                )
            stripped = entry.strip()
            if stripped:
                result.append(stripped)
        return tuple(result)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        action = self._action(arguments)

        if action == "create":
            return self._action_create(arguments)
        if action == "poll":
            return self._action_poll(arguments)
        if action == "steer":
            return self._action_steer(arguments)
        if action == "cancel":
            return self._action_cancel(arguments)
        if action == "sleep":
            return self._action_sleep(arguments)
        raise ValueError(
            f"Unknown subagent action: {action!r}"
        )

    def _action_create(
        self,
        arguments: dict[str, Any],
    ) -> str:
        spec = self._build_spec(arguments)
        subagent_id = self.__supervisor.create(
            spec,
            build_context=self._build_subagent_context,
        )
        return (
            "created subagent "
            f"{subagent_id!r}; use action='poll' to inspect progress."
        )

    def _action_poll(
        self,
        arguments: dict[str, Any],
    ) -> str:
        include_completed = bool(
            arguments.get("include_completed", True)
        )
        subagent_id = self._optional_string(
            arguments,
            "subagent_id",
        )
        answer = self._optional_string(
            arguments,
            "answer",
        )

        if answer and not subagent_id:
            raise ValueError(
                "'answer' is only valid together with 'subagent_id'."
            )

        if answer and subagent_id:
            answered = self.__supervisor.answer_guidance(
                subagent_id,
                answer,
            )
            if not answered:
                return (
                    f"no pending guidance request for "
                    f"subagent {subagent_id!r}."
                )

        if subagent_id:
            snapshot = self.__supervisor.snapshot(subagent_id)
            snapshots = (
                (snapshot,)
                if snapshot is not None
                else ()
            )
        else:
            snapshots = self.__supervisor.poll(
                include_completed=include_completed,
            )

        return _format_poll(
            snapshots,
            transcript_budget=_POLL_TRANSCRIPT_BUDGET,
        )

    def _action_steer(
        self,
        arguments: dict[str, Any],
    ) -> str:
        subagent_id = self._required_string(
            arguments,
            "subagent_id",
        )
        message = self._required_string(
            arguments,
            "message",
        )
        enqueued = self.__supervisor.steer(
            subagent_id,
            message,
        )
        if not enqueued:
            snapshot = self.__supervisor.snapshot(subagent_id)
            if snapshot is None:
                return f"unknown subagent: {subagent_id!r}"
            return (
                f"subagent {subagent_id!r} is not running "
                f"(status={snapshot.status.value}); steering dropped."
            )
        return f"queued steering for subagent {subagent_id!r}."

    def _action_sleep(
        self,
        arguments: dict[str, Any],
    ) -> str:
        subagent_id = self._optional_string(
            arguments,
            "subagent_id",
        )
        if subagent_id:
            subagent_ids = (subagent_id,)
        else:
            subagent_ids = self._str_list(
                arguments,
                "subagent_ids",
            )
            if not subagent_ids:
                raise ValueError(
                    "sleep requires 'subagent_id' or "
                    "'subagent_ids'."
                )

        timeout_raw = arguments.get("timeout_seconds", 30)
        try:
            timeout = float(timeout_raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "timeout_seconds must be a number of seconds."
            ) from error
        if timeout <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )
        timeout = min(timeout, _MAX_SLEEP_SECONDS)

        statuses = self.__supervisor.wait(
            subagent_ids,
            timeout=timeout,
        )

        lines: list[str] = []
        for sid in subagent_ids:
            status = statuses.get(sid, SubagentStatus.FAILED)
            snapshot = self.__supervisor.snapshot(sid)
            if snapshot is None:
                lines.append(
                    f"- {sid}: status={status.value} "
                    f"(no record)"
                )
                continue
            lines.append(
                f"- {sid}: status={status.value} "
                f"started={snapshot.started_at or '-'} "
                f"finished={snapshot.finished_at or '-'} "
                f"error={snapshot.error or '-'}"
            )

        return "subagent status after sleep:\n" + "\n".join(lines)

    def _action_cancel(
        self,
        arguments: dict[str, Any],
    ) -> str:
        subagent_id = self._required_string(
            arguments,
            "subagent_id",
        )
        cancelled = self.__supervisor.cancel(
            subagent_id,
            reason="cancelled by the orchestrator",
        )
        if cancelled:
            return f"cancelled subagent {subagent_id!r}."
        snapshot = self.__supervisor.snapshot(subagent_id)
        if snapshot is None:
            return f"unknown subagent: {subagent_id!r}"
        return (
            f"subagent {subagent_id!r} is already terminal "
            f"(status={snapshot.status.value})."
        )

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_spec(
        self,
        arguments: dict[str, Any],
    ) -> SubagentSpec:
        task = self._required_string(
            arguments,
            "task",
        )
        write_path = self._required_string(
            arguments,
            "write_path",
        )
        readonly_binds = self._str_list(
            arguments,
            "readonly_binds",
        )
        subagent_id = self._optional_string(
            arguments,
            "subagent_id",
        ) or ""
        network = bool(arguments.get("network", False))
        system_prompt_addendum = self._optional_string(
            arguments,
            "system_prompt_addendum",
        ) or ""

        return SubagentSpec(
            task=task,
            write_path=write_path,
            readonly_binds=readonly_binds,
            subagent_id=subagent_id,
            network=network,
            system_prompt_addendum=system_prompt_addendum,
        )

    def _build_subagent_context(
        self,
        spec: SubagentSpec,
        write_path: "Path",
        readonly_binds: "tuple[Path, ...]",
    ) -> ExecutionContext:
        from .factory import build_subagent_context as factory

        parent_workspace = self.context.workspace
        parent_config = self.context.config

        return factory(
            parent_workspace=parent_workspace,
            parent_config=parent_config,
            parent_skills=self.context.skills,
            supervisor=self.__supervisor,
            spec=spec,
            write_path=write_path,
            readonly_binds=readonly_binds,
        )

    # ------------------------------------------------------------------
    # Argument helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _required_string(
        arguments: dict[str, Any],
        name: str,
    ) -> str:
        value = arguments.get(name)
        if not isinstance(value, str):
            raise ValueError(f"{name} is required.")
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} cannot be empty.")
        return stripped

    @staticmethod
    def _optional_string(
        arguments: dict[str, Any],
        name: str,
    ) -> str | None:
        value = arguments.get(name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string.")
        return value.strip() or None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        action = arguments.get("action")
        if not isinstance(action, str):
            return "invalid"
        if action == "create":
            spec = self._build_spec(arguments)
            return (
                f"action=create | id={spec.subagent_id} | "
                f"write={spec.write_path}"
            )
        if action == "poll":
            subagent_id = arguments.get("subagent_id")
            if isinstance(subagent_id, str) and subagent_id:
                return f"action=poll | id={subagent_id}"
            return "action=poll | all"
        if action == "steer":
            subagent_id = arguments.get("subagent_id") or "-"
            return f"action=steer | id={subagent_id}"
        if action == "cancel":
            subagent_id = arguments.get("subagent_id") or "-"
            return f"action=cancel | id={subagent_id}"
        if action == "sleep":
            ids = arguments.get("subagent_ids") or arguments.get(
                "subagent_id"
            )
            if isinstance(ids, (list, tuple)):
                return f"action=sleep | ids={','.join(str(i) for i in ids)}"
            return f"action=sleep | id={ids}"
        return f"action={action}"

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        text = str(result)
        lines = text.count("\n") + 1
        return f"{lines} lines | {len(text)} chars"


# ---------------------------------------------------------------------------
# Poll formatting
# ---------------------------------------------------------------------------

def _format_poll(
    snapshots: tuple,
    *,
    transcript_budget: int,
) -> str:
    if not snapshots:
        return "no subagents are currently tracked."

    blocks: list[str] = []
    for snapshot in snapshots:
        blocks.append(
            _format_snapshot(
                snapshot,
                transcript_budget=transcript_budget,
            )
        )
    return "\n\n".join(blocks)


def _format_snapshot(
    snapshot: Any,
    *,
    transcript_budget: int,
) -> str:
    header = (
        f"subagent {snapshot.subagent_id!r} | "
        f"status={snapshot.status.value} | "
        f"write={snapshot.write_path}"
    )
    if snapshot.error:
        header += f" | error={snapshot.error}"

    lines = [header, "", f"task: {snapshot.task.strip()}"]

    if snapshot.pending_guidance:
        lines.append("")
        lines.append("pending guidance requests:")
        for entry in snapshot.pending_guidance:
            lines.append(f"- {entry.content}")

    if snapshot.transcript:
        lines.append("")
        lines.append("transcript:")
        recent = list(snapshot.transcript)[-transcript_budget:]
        for entry in recent:
            role = entry.role
            content = entry.content.strip()
            if entry.kind in {
                "guidance-request",
                "guidance-response",
            }:
                role = entry.kind
            lines.append(f"  [{role}] {content}")

    if snapshot.status.is_terminal:
        lines.append("")
        lines.append("done.")
    return "\n".join(lines)
