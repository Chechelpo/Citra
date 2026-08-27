from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, final

from citra.tools.default_registry import ToolSet
from citra.tools.skills.skill import Skill
from citra.tools.tool import Tool
from citra.utils.sandbox import SandboxMode

if TYPE_CHECKING:
    from citra.context import ExecutionContext

@dataclass(frozen=True)
class SandboxConfig:
    mode: SandboxMode = SandboxMode.ONLY_SOURCE

    additional_ro_binds: tuple[Path, ...] = field(
        default_factory=tuple
    )

    additional_w_binds: tuple[Path, ...] = field(
        default_factory=tuple
    )

    global_network_disallow: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mode, SandboxMode):
            raise TypeError("mode must be a SandboxMode")

        for name, paths in (
            ("additional_ro_binds", self.additional_ro_binds),
            ("additional_w_binds", self.additional_w_binds),
        ):
            if not isinstance(paths, tuple) or not all(
                isinstance(path, Path)
                for path in paths
            ):
                raise TypeError(f"{name} must be a tuple of Path values")

        if not isinstance(self.global_network_disallow, bool):
            raise TypeError("global_network_disallow must be a boolean")


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
        steering = self.task_steering

        if steering is None:
            return None

        if current_turn < 0:
            raise ValueError(
                "current_turn cannot be negative"
            )

        if current_turn == 0:
            if not steering.include_first:
                return None

            return steering.get_content(
                context
            )

        if (
            current_turn
            % steering.every_n_turns
            != 0
        ):
            return None

        return steering.get_content(
            context
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

    _TOOLS : ClassVar[ToolSet]

    _AVAILABLE_SKILLS: ClassVar[
        tuple[type[Skill], ...]
    ] = ()

    _SANDBOX_CONFIG: ClassVar[
        SandboxConfig
    ] = SandboxConfig()

    _TASK_STEERING: ClassVar[
        TaskSteeringConfig | None
    ] = None

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
    def sandbox_config(self) -> SandboxConfig:
        return self._SANDBOX_CONFIG

    @property
    @final
    def task_steering(
        self,
    ) -> TaskSteeringConfig | None:
        return self._TASK_STEERING

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
        available_skills: tuple[type[Skill], ...] = (),
        sandbox_config: SandboxConfig | None = None,
        task_steering: TaskSteeringConfig | None = None,
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

        self.validate()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def core_tools(
        self,
    ) -> tuple[type[Tool], ...]:
        return self._core_tools

    @property
    def allowed_tools(
        self,
    ) -> tuple[type[Tool], ...]:
        return self._allowed_tools

    @property
    def available_skills(
        self,
    ) -> tuple[type[Skill], ...]:
        return self._available_skills

    @property
    def sandbox_config(self) -> SandboxConfig:
        return self._sandbox_config

    @property
    def task_steering(
        self,
    ) -> TaskSteeringConfig | None:
        return self._task_steering

    def get_system_prompt(
        self,
        context: ExecutionContext,
    ) -> str:
        return self._system_prompt
