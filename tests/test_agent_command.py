"""Tests for direct CLI supervision of subagents."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from citra.commands.agent import AgentCommand
from citra.context import ExecutionContext
from citra.tools.subagent.spec import (
    SubagentSnapshot,
    SubagentStatus,
    TranscriptEntry,
)


class _Supervisor:
    def __init__(self) -> None:
        self.steering: list[tuple[str, str]] = []
        self.answers: list[tuple[str, str]] = []
        self.cancelled: list[str] = []
        self.item = SubagentSnapshot(
            subagent_id="worker-1",
            status=SubagentStatus.RUNNING,
            task="Implement parser",
            write_path="/workspace/parser",
            transcript=(
                TranscriptEntry(
                    role="assistant",
                    content="Inspecting parser.py",
                    kind="assistant",
                ),
                TranscriptEntry(
                    role="tool",
                    content="2 tests passed",
                    kind="tool-result",
                    tool="exec_command",
                ),
            ),
            pending_guidance=(),
        )

    def poll(self) -> tuple[SubagentSnapshot, ...]:
        return (self.item,)

    def snapshot(self, subagent_id: str) -> SubagentSnapshot | None:
        return self.item if subagent_id == self.item.subagent_id else None

    def steer(self, subagent_id: str, message: str) -> bool:
        if self.snapshot(subagent_id) is None:
            return False
        self.steering.append((subagent_id, message))
        return True

    def answer_guidance(self, subagent_id: str, message: str) -> bool:
        if self.snapshot(subagent_id) is None:
            return False
        self.answers.append((subagent_id, message))
        return True

    def cancel(self, subagent_id: str, *, reason: str) -> bool:
        del reason
        if self.snapshot(subagent_id) is None:
            return False
        self.cancelled.append(subagent_id)
        return True


def _command(supervisor: _Supervisor) -> AgentCommand:
    context = SimpleNamespace(subagents=supervisor)
    return AgentCommand(cast(ExecutionContext, cast(Any, context)))


def test_agent_command_lists_and_inspects_worker_events() -> None:
    supervisor = _Supervisor()
    command = _command(supervisor)

    assert "worker-1 | running" in command.run("list").output
    shown = command.run("worker-1").output
    assert "Inspecting parser.py" in shown
    assert "tool-result:exec_command" in shown


def test_agent_command_controls_one_worker() -> None:
    supervisor = _Supervisor()
    command = _command(supervisor)

    assert "Queued steering" in command.run(
        "steer worker-1 focus on validation"
    ).output
    assert "Answered guidance" in command.run(
        "answer worker-1 use the existing parser"
    ).output
    assert "Cancelled" in command.run("cancel worker-1").output
    assert supervisor.steering == [("worker-1", "focus on validation")]
    assert supervisor.answers == [("worker-1", "use the existing parser")]
    assert supervisor.cancelled == ["worker-1"]
