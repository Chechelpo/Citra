from __future__ import annotations

from abc import ABC, abstractmethod
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, final

from citra.tools.default_registry import ToolSet
from citra.tools.skills.skill import Skill
from citra.tools.tool import Tool
from citra.sandbox.sandbox_mode import SandboxMode

if TYPE_CHECKING:
    from citra.context import ExecutionContext


@dataclass(frozen=True)
class SandboxConfig:
    """One mode/workflow contribution to the process sandbox policy."""

    mode: SandboxMode = SandboxMode.FULL_SANDBOX
    additional_ro_binds: tuple[Path, ...] = ()
    additional_w_binds: tuple[Path, ...] = ()
    global_network_disallow: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mode, SandboxMode):
            raise TypeError("mode must be a SandboxMode")
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
    """
    User message injected every N turns.
    """

    every_n_turns: int = 0
    content: str = ""
    include_first: bool = False

    def __post_init__(self) -> None:
        if self.every_n_turns <= 0:
            raise ValueError(
                "every_n_turns must be greater than zero"
            )

    def get_content(
        self,
        context: ExecutionContext,
    ) -> str:
        """
        Implementations may override this for context-aware steering.
        """
        del context
        return self.content


class Mode(ABC):
    """
    Interface implemented by every Citra operating mode.

    A mode may be backed by:
      - static Python configuration
      - user configuration
      - another future configuration source

    Consumers should depend only on this interface.
    """

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str | None:
        ...

    # -------------------------------------------------------------------------
    # Tools / skills
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def tool_set(self) -> ToolSet:
        ...

    @property
    def skills(self) -> tuple[Skill, ...]:
        ...
    # -------------------------------------------------------------------------
    # Execution configuration
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def sandbox_config(self) -> SandboxConfig:
        ...

    @property
    @abstractmethod
    def task_steering(
        self,
    ) -> TaskSteeringConfig | None:
        ...

    @property
    @abstractmethod
    def initial_working_states(self) -> tuple[str, ...]:
        """Provisional memory states the model creates on its first turn."""

        ...

    # -------------------------------------------------------------------------
    # Prompt
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_system_prompt(
        self,
        context: ExecutionContext,
    ) -> str:
        ...

    # -------------------------------------------------------------------------
    # Shared behavior
    # -------------------------------------------------------------------------

    @final
    def get_task_steering(
        self,
        current_turn: int,
        context: ExecutionContext,
    ) -> str | None:
        if current_turn < 0:
            raise ValueError(
                "current_turn cannot be negative"
            )

        parts: list[str] = []

        if current_turn == 0:
            initial_state_steering = self._initial_state_steering(context)
            if initial_state_steering:
                parts.append(initial_state_steering)

        steering = self.task_steering
        if steering is None:
            return "\n\n".join(parts) or None

        if current_turn == 0:
            if steering.include_first:
                parts.append(steering.get_content(context))
        elif current_turn % steering.every_n_turns == 0:
            parts.append(steering.get_content(context))

        return "\n\n".join(part for part in parts if part.strip()) or None

    def _initial_state_steering(
        self,
        context: ExecutionContext,
    ) -> str | None:
        states = self.initial_working_states
        if not states:
            return None

        memory_config = getattr(getattr(context, "config", None), "memory", None)
        if not bool(getattr(memory_config, "enabled", True)):
            return None

        disabled_tool_ids = set(
            getattr(getattr(context, "workspace", None), "disabled_tool_ids", ())
        )
        if "working_state" in disabled_tool_ids:
            return None

        tool_type = self.tool_set.get_tool_with_id("working_state")
        if tool_type is None:
            return None

        public_tool_id = tool_type.resolve_definition_for_context(
            context
        ).function.name
        arguments = json.dumps(
            {
                "action": "create",
                "contents": list(states),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return (
            "Initialize this mode's provisional memory before other task work. "
            f"Call the `{public_tool_id}` tool once with exactly these arguments: "
            f"`{arguments}`. These are working states, not established facts; "
            "maintain, promote, resolve, or discard them through the memory tools "
            "as the task develops."
        )

    @final
    def validate(self) -> None:
        """
        Validate common invariants shared by all Mode implementations.
        """

        if not self.name.strip():
            raise ValueError(
                "Mode name cannot be empty"
            )

        if not isinstance(self.sandbox_config, SandboxConfig):
            raise TypeError("sandbox_config must be a SandboxConfig")

        if not isinstance(self.initial_working_states, tuple):
            raise TypeError("initial_working_states must be a tuple")

        if any(
            not isinstance(state, str) or not state.strip()
            for state in self.initial_working_states
        ):
            raise ValueError(
                "initial_working_states must contain non-empty strings"
            )

        duplicate_states = self._duplicates(self.initial_working_states)
        if duplicate_states:
            raise ValueError(
                "Duplicate initial working states: "
                + ", ".join(repr(state) for state in duplicate_states)
            )

        if (
            self.initial_working_states
            and "working_state" not in self.tool_set.core_tool_ids
        ):
            raise ValueError(
                "Modes with initial working states must expose the "
                "'working_state' tool as a core tool"
            )

    @staticmethod
    def _validate_tool_tuple(
        name: str,
        tools: tuple[type[Tool], ...],
    ) -> None:
        if not isinstance(tools, tuple):
            raise TypeError(
                f"{name} must be a tuple"
            )

        for tool in tools:
            if (
                not isinstance(tool, type)
                or not issubclass(tool, Tool)
            ):
                raise TypeError(
                    f"{name} must contain Tool subclasses"
                )

    @staticmethod
    def _validate_skill_tuple(
        name: str,
        skills: tuple[type[Skill], ...],
    ) -> None:
        if not isinstance(skills, tuple):
            raise TypeError(
                f"{name} must be a tuple"
            )

        for skill in skills:
            if (
                not isinstance(skill, type)
                or not issubclass(skill, Skill)
            ):
                raise TypeError(
                    f"{name} must contain Skill subclasses"
                )

    @staticmethod
    def _duplicates(
        values: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        seen: set[Any] = set()
        duplicates: list[Any] = []

        for value in values:
            if value in seen:
                if value not in duplicates:
                    duplicates.append(value)
            else:
                seen.add(value)

        return tuple(duplicates)


class StaticMode(Mode):
    """
    Base class for modes declared directly in Python.
    """

    _NAME: ClassVar[str]
    _DESCRIPTION: ClassVar[str | None] = None

    _TOOLS: ClassVar[ToolSet]

    _AVAILABLE_SKILLS: ClassVar[
        tuple[Skill, ...]
    ] = ()

    _SANDBOX_CONFIG: ClassVar[
        SandboxConfig
    ] = SandboxConfig()

    _TASK_STEERING: ClassVar[
        TaskSteeringConfig | None
    ] = None

    _INITIAL_WORKING_STATES: ClassVar[tuple[str, ...]] = ()

    def __init__(self) -> None:
        self.validate()

    @property
    @final
    def name(self) -> str:
        return self._NAME

    @property
    @final
    def description(self) -> str | None:
        return self._DESCRIPTION

    @property
    @final
    def tool_set(self) -> ToolSet:
        return self._TOOLS

    @property
    @final
    def skills(self) -> tuple[Skill, ...]:
        return self._AVAILABLE_SKILLS

    @property
    @final
    def sandbox_config(self) -> SandboxConfig:
        return self._SANDBOX_CONFIG

    @property
    @final
    def task_steering(
        self,
    ) -> TaskSteeringConfig | None:
        return self._TASK_STEERING

    @property
    @final
    def initial_working_states(self) -> tuple[str, ...]:
        return self._INITIAL_WORKING_STATES


class UserMode(Mode):
    """
    Mode loaded from .citra/config/modes.toml.
    """

    def __init__(
        self,
        *,
        name: str,
        system_prompt: str,
        description: str | None = None,
        core_tools: tuple[type[Tool], ...] = (),
        allowed_tools: tuple[type[Tool], ...] = (),
        available_skills: tuple[Skill, ...] = (),
        sandbox_config: SandboxConfig | None = None,
        task_steering: TaskSteeringConfig | None = None,
        initial_working_states: tuple[str, ...] = (),
    ) -> None:
        self._name = name
        self._description = description
        self._system_prompt = system_prompt

        self._core_tools = core_tools
        self._allowed_tools = allowed_tools
        self._available_skills = available_skills

        self._sandbox_config = (
            sandbox_config
            if sandbox_config is not None
            else SandboxConfig()
        )

        self._task_steering = task_steering
        self._initial_working_states = initial_working_states
        self._tool_set = ToolSet(
            core_tools=core_tools,
            deferred_tools=allowed_tools,
        )

        self.validate()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def tool_set(self) -> ToolSet:
        return self._tool_set

    @property
    def skills(
        self,
    ) -> tuple[Skill, ...]:
        return self._available_skills

    @property
    def sandbox_config(self) -> SandboxConfig:
        return self._sandbox_config

    @property
    def task_steering(
        self,
    ) -> TaskSteeringConfig | None:
        return self._task_steering

    @property
    def initial_working_states(self) -> tuple[str, ...]:
        return self._initial_working_states

    def get_system_prompt(
        self,
        context: ExecutionContext,
    ) -> str:
        return self._system_prompt
