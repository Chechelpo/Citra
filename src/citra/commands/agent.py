"""CLI supervision for background subagents."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from citra.tools.subagent.spec import SubagentSnapshot

from .command import Command, CommandResult


_TRANSCRIPT_TAIL = 30
_ENTRY_CHAR_LIMIT = 4_000
_USAGE = """\
Usage:
  /agent list
  /agent show <id>
  /agent <id>
  /agent steer <id> <message>
  /agent answer <id> <message>
  /agent cancel <id>

The command is also available from the foreground steering prompt while the
orchestrator is running."""


@runtime_checkable
class SubagentController(Protocol):
    def poll(self) -> tuple[SubagentSnapshot, ...]: ...

    def snapshot(self, subagent_id: str) -> SubagentSnapshot | None: ...

    def steer(self, subagent_id: str, message: str) -> bool: ...

    def answer_guidance(self, subagent_id: str, message: str) -> bool: ...

    def cancel(self, subagent_id: str, *, reason: str) -> bool: ...


class AgentCommand(Command):
    """Inspect and control one worker without routing through the model."""

    id = "agent"
    description = "Inspect, steer, answer, or cancel a subagent."

    def _run(self, args: str) -> CommandResult:
        supervisor = self.context.subagents
        if not isinstance(supervisor, SubagentController):
            raise RuntimeError("Subagent supervision is unavailable.")

        parts = args.split(maxsplit=2)
        if not parts or parts[0] == "list":
            if len(parts) > 1:
                return CommandResult(output=_USAGE)
            return CommandResult(output=_format_list(supervisor.poll()))

        action = parts[0]
        if action == "show":
            if len(parts) != 2:
                return CommandResult(output=_USAGE)
            return CommandResult(
                output=_format_one(supervisor.snapshot(parts[1]))
            )
        if action == "steer":
            return CommandResult(
                output=self._send_message(
                    supervisor,
                    parts,
                    operation="steer",
                )
            )
        if action == "answer":
            return CommandResult(
                output=self._send_message(
                    supervisor,
                    parts,
                    operation="answer",
                )
            )
        if action == "cancel":
            if len(parts) != 2:
                return CommandResult(output=_USAGE)
            subagent_id = parts[1]
            if supervisor.cancel(
                subagent_id,
                reason="cancelled from the CLI",
            ):
                return CommandResult(
                    output=f"Cancelled subagent {subagent_id!r}."
                )
            return CommandResult(
                output=_terminal_or_unknown(supervisor, subagent_id)
            )
        if len(parts) == 1:
            return CommandResult(
                output=_format_one(supervisor.snapshot(action))
            )
        return CommandResult(output=_USAGE)

    @staticmethod
    def _send_message(
        supervisor: SubagentController,
        parts: list[str],
        *,
        operation: str,
    ) -> str:
        if len(parts) != 3 or not parts[2].strip():
            return _USAGE
        subagent_id = parts[1]
        message = parts[2].strip()
        if operation == "steer":
            accepted = supervisor.steer(subagent_id, message)
            verb = "Queued steering for"
        else:
            accepted = supervisor.answer_guidance(subagent_id, message)
            verb = "Answered guidance for"
        if accepted:
            return f"{verb} subagent {subagent_id!r}."
        return _terminal_or_unknown(supervisor, subagent_id)


def _terminal_or_unknown(
    supervisor: SubagentController,
    subagent_id: str,
) -> str:
    snapshot = supervisor.snapshot(subagent_id)
    if snapshot is None:
        return f"Unknown subagent: {subagent_id!r}."
    return (
        f"Subagent {subagent_id!r} cannot accept that operation "
        f"(status={snapshot.status.value})."
    )


def _format_list(snapshots: tuple[SubagentSnapshot, ...]) -> str:
    if not snapshots:
        return "No subagents are currently tracked."
    lines = ["Subagents:"]
    for snapshot in snapshots:
        task = " ".join(snapshot.task.split())
        if len(task) > 80:
            task = task[:77] + "..."
        pending = len(snapshot.pending_guidance)
        pending_text = f" | guidance={pending}" if pending else ""
        lines.append(
            f"- {snapshot.subagent_id} | {snapshot.status.value}"
            f"{pending_text} | {task}"
        )
    return "\n".join(lines)


def _format_one(snapshot: SubagentSnapshot | None) -> str:
    if snapshot is None:
        return "Unknown subagent. Use /agent list to inspect tracked ids."

    lines = [
        f"Subagent {snapshot.subagent_id!r}",
        f"status: {snapshot.status.value}",
        f"task: {snapshot.task}",
        f"write path: {snapshot.write_path}",
        f"started: {snapshot.started_at or '-'}",
        f"finished: {snapshot.finished_at or '-'}",
    ]
    if snapshot.error:
        lines.append(f"error: {snapshot.error}")
    if snapshot.pending_guidance:
        lines.extend(("", "Pending guidance:"))
        for entry in snapshot.pending_guidance:
            lines.append(f"- {entry.content.strip()}")
    if snapshot.transcript:
        lines.extend(("", f"Latest work (last {_TRANSCRIPT_TAIL} events):"))
        for entry in snapshot.transcript[-_TRANSCRIPT_TAIL:]:
            label = entry.kind
            if entry.tool:
                label += f":{entry.tool}"
            content = entry.content.strip() or "(empty)"
            if len(content) > _ENTRY_CHAR_LIMIT:
                omitted = len(content) - _ENTRY_CHAR_LIMIT
                content = (
                    content[:_ENTRY_CHAR_LIMIT]
                    + f"\n... <{omitted} characters omitted>"
                )
            content_lines = content.splitlines()
            lines.append(f"[{label}] {content_lines[0]}")
            lines.extend(f"  {line}" for line in content_lines[1:])
    return "\n".join(lines)


__all__ = ["AgentCommand"]
