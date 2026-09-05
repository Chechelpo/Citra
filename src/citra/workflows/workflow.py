"""Workflow definitions, execution profiles, and runtime state.

A workflow is the only agent-execution abstraction in Citra. Workflows that
need one persistent agent derive from :class:`SingleModeWorkflow`; composite
workflows arrange those same objects into validated steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, ClassVar, final

from citra.logging import Logger
from citra.sandbox.sandbox import WorkspaceSandbox
from citra.sandbox.sandbox_mode import SandboxMode
from citra.config._sandbox_policy import SandboxPolicy
from citra.context.workspace_context import WorkspaceContext
from citra.tools.default_registry import ToolConfiguration, ToolSet, ToolSetInput
from citra.tools.skills.skill import Skill
from citra.tools.tool import Tool

if TYPE_CHECKING:
    from citra.agent import ConversationMemory
    from citra.context import ExecutionContext


_logger = Logger(__name__)


@dataclass(frozen=True)
class SandboxConfig:
    """One workflow's contribution to the process sandbox policy."""

    mode: SandboxMode = SandboxMode.PARTIAL_SANDBOX
    additional_ro_binds: tuple[Path, ...] = ()
    additional_w_binds: tuple[Path, ...] = ()
    global_network_disallow: bool = False

    def __post_init__(self) -> None:
        """Validate and initialize the instance after construction."""
        if not isinstance(self.mode, SandboxMode):
            raise TypeError("mode must be a SandboxMode")
        if not isinstance(self.additional_ro_binds, tuple):
            raise TypeError("additional_ro_binds must be a tuple")
        if not isinstance(self.additional_w_binds, tuple):
            raise TypeError("additional_w_binds must be a tuple")
        object.__setattr__(
            self,
            "additional_ro_binds",
            tuple(Path(path).expanduser() for path in self.additional_ro_binds),
        )
        object.__setattr__(
            self,
            "additional_w_binds",
            tuple(Path(path).expanduser() for path in self.additional_w_binds),
        )
        if not isinstance(self.global_network_disallow, bool):
            raise TypeError("global_network_disallow must be boolean")


@dataclass(frozen=True)
class TaskSteeringConfig:
    """A user message injected every N turns; zero disables injection."""

    every_n_turns: int = 0
    content: str = ""
    include_first: bool = False

    def __post_init__(self) -> None:
        """Validate and initialize the instance after construction."""
        if type(self.every_n_turns) is not int:
            raise TypeError("every_n_turns must be an integer")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if not isinstance(self.include_first, bool):
            raise TypeError("include_first must be boolean")
        if self.every_n_turns < 0:
            raise ValueError("every_n_turns cannot be negative")
        if self.every_n_turns == 0 and (self.content or self.include_first):
            raise ValueError(
                "disabled task steering cannot define content or include_first"
            )
        if self.every_n_turns > 0 and not self.content.strip():
            raise ValueError("enabled task steering requires non-empty content")

    @property
    def enabled(self) -> bool:
        """Handle enabled."""
        return self.every_n_turns > 0

    def get_content(self, context: ExecutionContext) -> str:
        """Return context-aware steering; subclasses may override this."""
        del context
        return self.content


class Workflow(ABC):
    """Process-level workflow and sandbox-policy owner."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Handle name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Handle description."""
        ...

    @property
    @abstractmethod
    def sandbox_config(self) -> SandboxConfig:
        """Handle sandbox config."""
        ...

    @property
    @abstractmethod
    def initial_workflow(self) -> SingleModeWorkflow:
        """Return the first executable workflow in this definition."""
        ...

    @property
    def resolved_sandbox_config(self) -> SandboxConfig:
        """Compatibility name for the workflow's already-resolved policy."""
        return self.sandbox_config

    @property
    def is_serial(self) -> bool:
        """Return whether is serial."""
        return False

    @property
    def requires_memory(self) -> bool:
        """Handle requires memory."""
        return False

    @abstractmethod
    def create_run(self, task: str) -> WorkflowRun:
        """Handle create run."""
        ...

    def validate(self) -> None:
        """Handle validate."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Workflow name cannot be empty")
        if not isinstance(self.description, str):
            raise TypeError("workflow description must be a string")
        if not isinstance(self.sandbox_config, SandboxConfig):
            raise TypeError("workflow sandbox_config must be a SandboxConfig")
        initial = self.initial_workflow
        if not isinstance(initial, SingleModeWorkflow):
            raise TypeError(
                "workflow initial_workflow must be a SingleModeWorkflow"
            )
        if initial is not self:
            initial.validate()
            if initial.sandbox_config != self.sandbox_config:
                raise ValueError(
                    "workflow initial_workflow must expose the root workflow's "
                    "sandbox configuration"
                )


class SingleModeWorkflow(Workflow):
    """One persistent agent configuration represented as a workflow."""

    @property
    @abstractmethod
    def tool_set(self) -> ToolSet:
        """Handle tool set."""
        ...

    @property
    def skills(self) -> tuple[Skill, ...]:
        """Handle skills."""
        return ()

    @property
    @abstractmethod
    def task_steering(self) -> TaskSteeringConfig:
        """Handle task steering."""
        ...

    @property
    @abstractmethod
    def initial_working_states(self) -> tuple[str, ...]:
        """Provisional memory states created on the first turn."""
        ...

    @abstractmethod
    def get_system_prompt(self, context: ExecutionContext) -> str:
        """Return get system prompt."""
        ...

    @property
    @final
    def initial_workflow(self) -> SingleModeWorkflow:
        """Handle initial workflow."""
        return self

    @final
    def get_task_steering(
        self,
        current_turn: int,
        context: ExecutionContext,
    ) -> str | None:
        """Return get task steering."""
        if current_turn < 0:
            raise ValueError("current_turn cannot be negative")

        parts: list[str] = []
        if current_turn == 0:
            initial_state_steering = self._initial_state_steering(context)
            if initial_state_steering:
                parts.append(initial_state_steering)

        steering = self.task_steering
        if steering.enabled:
            if current_turn == 0 and steering.include_first:
                parts.append(steering.get_content(context))
            elif current_turn > 0 and current_turn % steering.every_n_turns == 0:
                parts.append(steering.get_content(context))

        return "\n\n".join(part for part in parts if part.strip()) or None

    def _initial_state_steering(self, context: ExecutionContext) -> str | None:
        """Handle initial state steering."""
        states = self.initial_working_states
        if not states or not context.config.memory.enabled:
            return None
        if "working_state" in context.workspace.disabled_tool_ids:
            return None

        tool_type = self.tool_set.get_tool_with_id("working_state")
        if tool_type is None:
            return None
        public_tool_id = tool_type.resolve_definition_for_context(
            context
        ).function.name
        arguments = json.dumps(
            {"action": "create", "contents": list(states)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return (
            "Initialize this workflow's provisional memory before other task "
            f"work. Call the `{public_tool_id}` tool once with exactly these "
            f"arguments: `{arguments}`. These are working states, not "
            "established facts; maintain, promote, resolve, or discard them "
            "through the memory tools as the task develops."
        )

    @final
    def validate(self) -> None:
        """Handle validate."""
        super().validate()
        if not isinstance(self.tool_set, ToolSet):
            raise TypeError("workflow tool_set must be a ToolSet")
        if not isinstance(self.skills, tuple) or any(
            not isinstance(skill, Skill) for skill in self.skills
        ):
            raise TypeError("workflow skills must contain Skill instances")
        if not isinstance(self.task_steering, TaskSteeringConfig):
            raise TypeError("workflow task_steering must be a TaskSteeringConfig")
        if not isinstance(self.initial_working_states, tuple):
            raise TypeError("initial_working_states must be a tuple")
        if any(
            not isinstance(state, str) or not state.strip()
            for state in self.initial_working_states
        ):
            raise ValueError(
                "initial_working_states must contain non-empty strings"
            )
        duplicates = self._duplicates(self.initial_working_states)
        if duplicates:
            raise ValueError(
                "Duplicate initial working states: "
                + ", ".join(repr(state) for state in duplicates)
            )
        if (
            self.initial_working_states
            and "working_state" not in self.tool_set.core_tool_ids
        ):
            raise ValueError(
                "Workflows with initial working states must expose the "
                "'working_state' tool as a core tool"
            )

    def create_run(self, task: str) -> WorkflowRun:
        """Handle create run."""
        return WorkflowRun(
            workflow=self,
            task=task,
            steps=(WorkflowStep("agent", self, ("complete",)),),
            max_executions=1,
        )

    @staticmethod
    def _validate_tool_tuple(
        name: str,
        tools: tuple[ToolSetInput, ...],
    ) -> None:
        """Handle validate tool tuple."""
        if not isinstance(tools, tuple):
            raise TypeError(f"{name} must be a tuple")
        if any(
            not isinstance(tool, ToolConfiguration)
            and (not isinstance(tool, type) or not issubclass(tool, Tool))
            for tool in tools
        ):
            raise TypeError(
                f"{name} must contain Tool subclasses or ToolConfiguration entries"
            )

    @staticmethod
    def _validate_skill_tuple(
        name: str,
        skills: tuple[type[Skill], ...],
    ) -> None:
        """Handle validate skill tuple."""
        if not isinstance(skills, tuple):
            raise TypeError(f"{name} must be a tuple")
        if any(
            not isinstance(skill, type) or not issubclass(skill, Skill)
            for skill in skills
        ):
            raise TypeError(f"{name} must contain Skill subclasses")

    @staticmethod
    def _duplicates(values: tuple[Any, ...]) -> tuple[Any, ...]:
        """Handle duplicates."""
        seen: set[Any] = set()
        duplicates: list[Any] = []
        for value in values:
            if value in seen and value not in duplicates:
                duplicates.append(value)
            else:
                seen.add(value)
        return tuple(duplicates)


class StaticWorkflow(SingleModeWorkflow):
    """Base class for single-mode workflows declared in Python."""

    _NAME: ClassVar[str]
    _DESCRIPTION: ClassVar[str] = ""
    _TOOLS: ClassVar[ToolSet]
    _AVAILABLE_SKILLS: ClassVar[tuple[Skill, ...]] = ()
    _SANDBOX_CONFIG: ClassVar[SandboxConfig] = SandboxConfig()
    _TASK_STEERING: ClassVar[TaskSteeringConfig] = TaskSteeringConfig()
    _INITIAL_WORKING_STATES: ClassVar[tuple[str, ...]] = ()

    def __init__(self) -> None:
        """Initialize the instance."""
        self.validate()

    @property
    @final
    def name(self) -> str:
        """Handle name."""
        return self._NAME

    @property
    @final
    def description(self) -> str:
        """Handle description."""
        return self._DESCRIPTION

    @property
    @final
    def tool_set(self) -> ToolSet:
        """Handle tool set."""
        return self._TOOLS

    @property
    @final
    def skills(self) -> tuple[Skill, ...]:
        """Handle skills."""
        return self._AVAILABLE_SKILLS

    @property
    @final
    def sandbox_config(self) -> SandboxConfig:
        """Handle sandbox config."""
        return self._SANDBOX_CONFIG

    @property
    @final
    def task_steering(self) -> TaskSteeringConfig:
        """Handle task steering."""
        return self._TASK_STEERING

    @property
    @final
    def initial_working_states(self) -> tuple[str, ...]:
        """Handle initial working states."""
        return self._INITIAL_WORKING_STATES


class UserWorkflow(SingleModeWorkflow):
    """A single-mode workflow supplied by user configuration or callers."""

    def __init__(
        self,
        *,
        name: str,
        system_prompt: str,
        description: str = "",
        core_tools: tuple[ToolSetInput, ...] = (),
        allowed_tools: tuple[ToolSetInput, ...] = (),
        available_skills: tuple[Skill, ...] = (),
        sandbox_config: SandboxConfig = SandboxConfig(),
        task_steering: TaskSteeringConfig = TaskSteeringConfig(),
        initial_working_states: tuple[str, ...] = (),
    ) -> None:
        """Initialize the instance."""
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        if not isinstance(description, str):
            raise TypeError("description must be a string")
        if not isinstance(sandbox_config, SandboxConfig):
            raise TypeError("sandbox_config must be a SandboxConfig")
        if not isinstance(task_steering, TaskSteeringConfig):
            raise TypeError("task_steering must be a TaskSteeringConfig")
        self._validate_tool_tuple("core_tools", core_tools)
        self._validate_tool_tuple("allowed_tools", allowed_tools)
        if not isinstance(available_skills, tuple) or any(
            not isinstance(skill, Skill) for skill in available_skills
        ):
            raise TypeError("available_skills must contain Skill instances")
        if not isinstance(initial_working_states, tuple):
            raise TypeError("initial_working_states must be a tuple")
        self._name = name
        self._description = description
        self._system_prompt = system_prompt
        self._available_skills = available_skills
        self._sandbox_config = sandbox_config
        self._task_steering = task_steering
        self._initial_working_states = initial_working_states
        self._tool_set = ToolSet(
            core_tools=core_tools,
            deferred_tools=allowed_tools,
        )
        self.validate()

    @property
    def name(self) -> str:
        """Handle name."""
        return self._name

    @property
    def description(self) -> str:
        """Handle description."""
        return self._description

    @property
    def tool_set(self) -> ToolSet:
        """Handle tool set."""
        return self._tool_set

    @property
    def skills(self) -> tuple[Skill, ...]:
        """Handle skills."""
        return self._available_skills

    @property
    def sandbox_config(self) -> SandboxConfig:
        """Handle sandbox config."""
        return self._sandbox_config

    @property
    def task_steering(self) -> TaskSteeringConfig:
        """Handle task steering."""
        return self._task_steering

    @property
    def initial_working_states(self) -> tuple[str, ...]:
        """Handle initial working states."""
        return self._initial_working_states

    def get_system_prompt(self, context: ExecutionContext) -> str:
        """Return get system prompt."""
        del context
        return self._system_prompt


@dataclass(frozen=True)
class WorkflowStep:
    """One single-mode workflow and its allowed transitions."""

    step_id: str
    workflow: SingleModeWorkflow
    allowed_next: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate and initialize the instance after construction."""
        if not isinstance(self.step_id, str) or not self.step_id.strip():
            raise ValueError("Workflow step id cannot be empty")
        if not isinstance(self.allowed_next, tuple):
            raise TypeError("Workflow allowed_next must be a tuple")
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
        if not isinstance(self.workflow, SingleModeWorkflow):
            raise TypeError("Workflow steps require a SingleModeWorkflow")
        self.workflow.validate()


@dataclass(frozen=True)
class WorkflowHandoff:
    """Validated route plus the outgoing workflow's assistant message."""

    step_id: str
    summary: str
    next_step: str

    def __post_init__(self) -> None:
        """Validate and initialize the instance after construction."""
        for name, value in (
            ("step_id", self.step_id),
            ("summary", self.summary),
            ("next_step", self.next_step),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Workflow handoff {name} cannot be empty")


@dataclass(frozen=True)
class WorkflowSnapshot:
    """Read-only state suitable for CLI inspection."""

    workflow: str
    task: str
    current_step: str
    completed: bool
    cancelled: bool
    execution_count: int
    handoffs: tuple[WorkflowHandoff, ...]


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
        """Initialize the instance."""
        if not isinstance(workflow, Workflow):
            raise TypeError("workflow must be a Workflow")
        if not isinstance(task, str):
            raise TypeError("task must be a string")
        if not isinstance(steps, tuple):
            raise TypeError("steps must be a tuple")
        if type(max_executions) is not int:
            raise TypeError("max_executions must be an integer")
        task = task.strip()
        if not task:
            raise ValueError("Workflow task cannot be empty")
        if not steps:
            raise ValueError("Workflow run requires at least one step")
        if any(not isinstance(step, WorkflowStep) for step in steps):
            raise TypeError("steps must contain WorkflowStep instances")
        if max_executions < 1:
            raise ValueError("max_executions must be positive")

        workflow.validate()
        for step in steps:
            if step.workflow.sandbox_config != workflow.sandbox_config:
                raise ValueError(
                    "Every workflow step must expose the root workflow's "
                    "sandbox configuration"
                )

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
        _logger.info(
            "Initialized workflow run",
            workflow=workflow.name,
            initial_step=self._current_step,
            max_executions=max_executions,
        )

    @property
    def memory(self) -> ConversationMemory:
        """Handle memory."""
        return self._memory

    @property
    def current_step(self) -> WorkflowStep:
        """Handle current step."""
        with self._lock:
            if self._completed or self._cancelled:
                raise RuntimeError("Workflow run has no active step")
            return self._steps[self._current_step]

    @property
    def is_terminal(self) -> bool:
        """Return whether is terminal."""
        with self._lock:
            return self._completed or self._cancelled

    @property
    def has_pending_handoff(self) -> bool:
        """Return whether has pending handoff."""
        with self._lock:
            return self._pending_handoff is not None

    def begin_step(self) -> WorkflowStep:
        """Handle begin step."""
        with self._lock:
            if self._completed or self._cancelled:
                raise RuntimeError("Workflow run is already terminal")
            if self._execution_count >= self.max_executions:
                self._cancelled = True
                _logger.error(
                    "Workflow execution bound exceeded",
                    workflow=self.workflow.name,
                    step=self._current_step,
                    executions=self._execution_count,
                )
                raise RuntimeError(
                    "Workflow exceeded its maximum serial step executions "
                    f"({self.max_executions})"
                )
            self._execution_count += 1
            self._pending_handoff = None
            _logger.info(
                "Began workflow step",
                workflow=self.workflow.name,
                step=self._current_step,
                execution=self._execution_count,
            )
            return self._steps[self._current_step]

    def submit_handoff(self, *, summary: str, next_step: str) -> WorkflowHandoff:
        """Handle submit handoff."""
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
                _logger.warning(
                    "Rejected disallowed workflow transition",
                    workflow=self.workflow.name,
                    step=step.step_id,
                    next_step=next_step,
                )
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
            _logger.debug(
                "Queued workflow handoff",
                workflow=self.workflow.name,
                step=step.step_id,
                next_step=next_step,
            )
            return handoff

    def advance(self) -> WorkflowHandoff:
        """Handle advance."""
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
            _logger.info(
                "Advanced workflow run",
                workflow=self.workflow.name,
                from_step=handoff.step_id,
                next_step=handoff.next_step,
                completed=self._completed,
            )
            return handoff

    def cancel(self) -> bool:
        """Handle cancel."""
        with self._lock:
            if self._completed or self._cancelled:
                return False
            self._cancelled = True
            _logger.warning(
                "Cancelled workflow run",
                workflow=self.workflow.name,
                step=self._current_step,
            )
            return True

    def phase_input(self) -> str:
        """Handle phase input."""
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
        """Handle snapshot."""
        with self._lock:
            current_step = (
                "" if self._completed or self._cancelled else self._current_step
            )
            return WorkflowSnapshot(
                workflow=self.workflow.name,
                task=self.task,
                current_step=current_step,
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
        sandbox: WorkspaceSandbox,
    ) -> None:
        """Initialize the instance."""
        if not isinstance(workflow, Workflow):
            raise TypeError("workflow must be a Workflow")
        if not isinstance(workspace, WorkspaceContext):
            raise TypeError("workspace must be a WorkspaceContext")
        if not isinstance(sandbox, WorkspaceSandbox):
            raise TypeError("sandbox must be a WorkspaceSandbox")
        workflow.validate()
        self.workflow = workflow
        self.workspace = workspace
        self._sandbox_config = workflow.sandbox_config
        self._sandbox = sandbox
        if self._sandbox.mode != self._sandbox_config.mode:
            raise RuntimeError(
                "Workflow sandbox mode differs from its frozen policy: "
                f"{self._sandbox.mode} vs {self._sandbox_config.mode}"
            )
        self._active_run: WorkflowRun | None = None
        self._lock = RLock()
        _logger.info(
            "Initialized workflow runtime",
            workflow=workflow.name,
            sandbox_mode=self._sandbox_config.mode.value,
        )

    @classmethod
    def provision(
        cls,
        *,
        workflow: Workflow,
        workspace: WorkspaceContext,
        policy: SandboxPolicy,
    ) -> WorkflowRuntime:
        """Handle provision."""
        if not isinstance(policy, SandboxPolicy):
            raise TypeError("policy must be a SandboxPolicy")
        sandbox = WorkspaceSandbox(
            workspace,
            policy,
            base_environment=workspace.environment(),
        )
        return cls(workflow=workflow, workspace=workspace, sandbox=sandbox)

    @property
    def sandbox_config(self) -> SandboxConfig:
        """Handle sandbox config."""
        return self._sandbox_config

    @property
    def sandbox(self) -> WorkspaceSandbox:
        """Handle sandbox."""
        return self._sandbox

    @property
    def active_run(self) -> WorkflowRun | None:
        """Handle active run."""
        with self._lock:
            return self._active_run

    def require_active_run(self) -> WorkflowRun:
        """Handle require active run."""
        run = self.active_run
        if run is None or run.is_terminal:
            raise RuntimeError("Workflow runtime has no active run")
        return run

    @property
    def active_workflow(self) -> SingleModeWorkflow:
        """Handle active workflow."""
        run = self.active_run
        if run is not None and not run.is_terminal:
            return run.current_step.workflow
        return self.workflow.initial_workflow

    def start_run(self, task: str) -> WorkflowRun:
        """Handle start run."""
        with self._lock:
            if self._active_run is not None and not self._active_run.is_terminal:
                raise RuntimeError("A workflow run is already active")
            run = self.workflow.create_run(task)
            self._active_run = run
            _logger.info("Started workflow run", workflow=self.workflow.name)
            return run

    def cancel_run(self) -> bool:
        """Handle cancel run."""
        with self._lock:
            if self._active_run is None:
                _logger.trace("Skipped workflow cancellation without active run")
                return False
            cancelled = self._active_run.cancel()
            _logger.info(
                "Processed workflow cancellation",
                workflow=self.workflow.name,
                cancelled=cancelled,
            )
            return cancelled


__all__ = [
    "SandboxConfig",
    "SingleModeWorkflow",
    "StaticWorkflow",
    "TaskSteeringConfig",
    "UserWorkflow",
    "Workflow",
    "WorkflowHandoff",
    "WorkflowRun",
    "WorkflowRuntime",
    "WorkflowSnapshot",
    "WorkflowStep",
]
