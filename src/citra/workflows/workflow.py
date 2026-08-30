"""Workflow definitions and serial-run state.

A workflow owns the sandbox policy for a Citra process.  Modes are ephemeral
agent roles executed inside that workflow-owned sandbox; they do not create or
replace the runtime when a workflow advances to another role.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING

from citra.modes import Mode, SandboxConfig
from citra.sandbox import WorkspaceSandbox

if TYPE_CHECKING:
    from citra.agent import ConversationMemory
    from citra.context import SandboxContextConfig, WorkspaceContext


@dataclass(frozen=True)
class WorkflowStep:
    """One mode turn and the transitions it is allowed to request."""

    step_id: str
    mode: Mode
    allowed_next: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.step_id.strip():
            raise ValueError("Workflow step id cannot be empty")
        if not self.allowed_next:
            raise ValueError(
                f"Workflow step {self.step_id!r} must allow a transition"
            )
        if any(
            not isinstance(item, str) or not item.strip()
            for item in self.allowed_next
        ):
            raise ValueError("Workflow transition ids cannot be empty")
        if len(set(self.allowed_next)) != len(self.allowed_next):
            raise ValueError(
                f"Workflow step {self.step_id!r} has duplicate transitions"
            )
        self.mode.validate()


@dataclass(frozen=True)
class WorkflowHandoff:
    """Validated route plus the outgoing role's actual assistant message."""

    step_id: str
    summary: str
    next_step: str


@dataclass(frozen=True)
class WorkflowSnapshot:
    """Read-only state suitable for CLI inspection."""

    workflow: str
    task: str
    current_step: str | None
    completed: bool
    cancelled: bool
    execution_count: int
    handoffs: tuple[WorkflowHandoff, ...]


class Workflow(ABC):
    """Process-level orchestration definition and sandbox-policy owner."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str | None:
        ...

    @property
    @abstractmethod
    def sandbox_config(self) -> SandboxConfig | None:
        """Optional workflow override; ``None`` inherits the initial mode."""
        ...

    @property
    def resolved_sandbox_config(self) -> SandboxConfig:
        """Policy frozen when the workflow runtime is provisioned."""
        return self.sandbox_config or self.initial_mode.sandbox_config

    @property
    @abstractmethod
    def initial_mode(self) -> Mode:
        ...

    @property
    def is_serial(self) -> bool:
        return False

    @property
    def requires_memory(self) -> bool:
        """Whether memory tools remain enabled for this workflow."""
        return False

    @abstractmethod
    def create_run(self, task: str) -> WorkflowRun:
        ...

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Workflow name cannot be empty")
        self.initial_mode.validate()
        if self.sandbox_config is not None and not isinstance(
            self.sandbox_config,
            SandboxConfig,
        ):
            raise TypeError(
                "workflow sandbox_config must be a SandboxConfig or None"
            )
        if not isinstance(self.resolved_sandbox_config, SandboxConfig):
            raise TypeError(
                "workflow must resolve to a valid SandboxConfig"
            )


class WorkflowRun:
    """Mutable state for one task moving through a workflow definition."""

    def __init__(
        self,
        *,
        workflow: Workflow,
        task: str,
        steps: tuple[WorkflowStep, ...],
        max_executions: int = 32,
    ) -> None:
        task = task.strip()
        if not task:
            raise ValueError("Workflow task cannot be empty")
        if not steps:
            raise ValueError("Workflow run requires at least one step")
        if max_executions < 1:
            raise ValueError("max_executions must be positive")

        self.workflow = workflow
        self.task = task
        self.max_executions = max_executions
        self._steps = {step.step_id: step for step in steps}
        if len(self._steps) != len(steps):
            raise ValueError("Workflow step ids must be unique")
        for step in steps:
            unknown = set(step.allowed_next) - {*self._steps, "complete"}
            if unknown:
                raise ValueError(
                    f"Workflow step {step.step_id!r} has unknown transitions: "
                    + ", ".join(sorted(unknown))
                )

        self._current_step = steps[0].step_id
        self._handoffs: list[WorkflowHandoff] = []
        self._pending_handoff: WorkflowHandoff | None = None
        self._execution_count = 0
        self._completed = False
        self._cancelled = False
        from citra.agent import ConversationMemory

        self._memory = ConversationMemory()
        self._lock = RLock()

    @property
    def memory(self) -> ConversationMemory:
        """Task-scoped structured memory shared by isolated role sessions."""
        return self._memory

    @property
    def current_step(self) -> WorkflowStep:
        with self._lock:
            if self._completed or self._cancelled:
                raise RuntimeError("Workflow run has no active step")
            return self._steps[self._current_step]

    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return self._completed or self._cancelled

    @property
    def has_pending_handoff(self) -> bool:
        with self._lock:
            return self._pending_handoff is not None

    def begin_step(self) -> WorkflowStep:
        """Start one mode execution and reset its handoff slot."""
        with self._lock:
            if self._completed or self._cancelled:
                raise RuntimeError("Workflow run is already terminal")
            if self._execution_count >= self.max_executions:
                self._cancelled = True
                raise RuntimeError(
                    "Workflow exceeded its maximum serial step executions "
                    f"({self.max_executions})"
                )
            self._execution_count += 1
            self._pending_handoff = None
            return self._steps[self._current_step]

    def submit_handoff(self, *, summary: str, next_step: str) -> WorkflowHandoff:
        """Record the active role's sole explicit cross-role artifact."""
        if not isinstance(summary, str) or not isinstance(next_step, str):
            raise TypeError("Workflow handoff fields must be strings")
        summary = summary.strip()
        next_step = next_step.strip()
        if not summary:
            raise ValueError("Workflow handoff summary cannot be empty")
        with self._lock:
            if self._completed or self._cancelled:
                raise RuntimeError("Workflow run is already terminal")
            step = self._steps[self._current_step]
            if next_step not in step.allowed_next:
                allowed = ", ".join(step.allowed_next)
                raise ValueError(
                    f"Step {step.step_id!r} cannot transition to "
                    f"{next_step!r}; allowed: {allowed}"
                )
            if self._pending_handoff is not None:
                raise RuntimeError(
                    f"Step {step.step_id!r} already submitted its handoff"
                )
            handoff = WorkflowHandoff(
                step_id=step.step_id,
                summary=summary,
                next_step=next_step,
            )
            self._pending_handoff = handoff
            return handoff

    def advance(self) -> WorkflowHandoff:
        """Consume the pending handoff and apply its validated transition."""
        with self._lock:
            handoff = self._pending_handoff
            if handoff is None:
                raise RuntimeError(
                    f"Step {self._current_step!r} ended without a workflow "
                    "handoff"
                )
            self._handoffs.append(handoff)
            self._pending_handoff = None
            if handoff.next_step == "complete":
                self._completed = True
            else:
                self._current_step = handoff.next_step
            return handoff

    def cancel(self) -> bool:
        with self._lock:
            if self._completed or self._cancelled:
                return False
            self._cancelled = True
            return True

    def phase_input(self) -> str:
        """Build the role-local task and latest assistant-message handoff."""
        with self._lock:
            step = self._steps[self._current_step]
            lines = [
                "# Workflow task",
                "",
                self.task,
                "",
                "# Active workflow phase",
                "",
                step.step_id,
                "",
                "You have a fresh reasoning context. The sandbox and its "
                "filesystem are shared with the other workflow roles, but "
                "their conversation history is not available. Structured "
                "workflow memory is shared and injected separately.",
            ]
            if self._handoffs:
                handoff = self._handoffs[-1]
                lines.extend(
                    (
                        "",
                        "# Message handoff from the previous role",
                        "",
                        f"From `{handoff.step_id}` to `{handoff.next_step}`:",
                        "",
                        handoff.summary,
                    )
                )
            else:
                lines.extend(("", "No previous role message exists."))
            return "\n".join(lines).strip()

    def snapshot(self) -> WorkflowSnapshot:
        with self._lock:
            return WorkflowSnapshot(
                workflow=self.workflow.name,
                task=self.task,
                current_step=(
                    None
                    if self._completed or self._cancelled
                    else self._current_step
                ),
                completed=self._completed,
                cancelled=self._cancelled,
                execution_count=self._execution_count,
                handoffs=tuple(self._handoffs),
            )


class WorkflowRuntime:
    """Own the concrete sandbox and active run for one workflow process."""

    def __init__(
        self,
        *,
        workflow: Workflow,
        workspace: WorkspaceContext,
        operator_sandbox_config: SandboxContextConfig,
        sandbox: WorkspaceSandbox | None = None,
    ) -> None:
        workflow.validate()
        self.workflow = workflow
        self.workspace = workspace
        self._sandbox_config = workflow.resolved_sandbox_config
        self._sandbox = (
            sandbox
            if sandbox is not None
            else WorkspaceSandbox(
                workspace,
                config=operator_sandbox_config,
                mode_config=self._sandbox_config,
            )
        )
        if self._sandbox.mode != self._sandbox_config.mode:
            raise RuntimeError(
                "Workflow sandbox mode differs from its frozen policy"
            )
        self._active_run: WorkflowRun | None = None
        self._lock = RLock()

    @property
    def sandbox_config(self) -> SandboxConfig:
        return self._sandbox_config

    @property
    def sandbox(self) -> WorkspaceSandbox:
        return self._sandbox

    @property
    def active_run(self) -> WorkflowRun | None:
        with self._lock:
            return self._active_run

    def start_run(self, task: str) -> WorkflowRun:
        with self._lock:
            if (
                self._active_run is not None
                and not self._active_run.is_terminal
            ):
                raise RuntimeError("A workflow run is already active")
            run = self.workflow.create_run(task)
            self._active_run = run
            return run

    def cancel_run(self) -> bool:
        with self._lock:
            if self._active_run is None:
                return False
            return self._active_run.cancel()


class SingleModeWorkflow(Workflow):
    """One persistent agent session running a selected mode."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        mode: Mode,
        sandbox_config: SandboxConfig | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._mode = mode
        self._sandbox_config = sandbox_config
        self.validate()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def sandbox_config(self) -> SandboxConfig | None:
        return self._sandbox_config

    @property
    def initial_mode(self) -> Mode:
        return self._mode

    def create_run(self, task: str) -> WorkflowRun:
        return WorkflowRun(
            workflow=self,
            task=task,
            steps=(WorkflowStep("agent", self._mode, ("complete",)),),
            max_executions=1,
        )


__all__ = [
    "SingleModeWorkflow",
    "Workflow",
    "WorkflowHandoff",
    "WorkflowRun",
    "WorkflowRuntime",
    "WorkflowSnapshot",
    "WorkflowStep",
]
