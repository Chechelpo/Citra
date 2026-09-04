"""Default tool declarations and capability-aware workflow tool sets."""

from __future__ import annotations

import builtins
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

from citra.logging import Logger

from .capabilities import ToolCapabilities
from .session_memory import *
from .subagent.tool import SubagentTool
from .tool import Tool
from .transient import (
    Bash,
    Browser,
    Diagram,
    Document,
    Edit,
    Git,
    Glob,
    Lsp,
    PromptUser,
    Read,
    ReadImage,
    SkillTool,
    Subprocess,
    Tree,
    WebSearch,
    Workspace,
    Write,
)

__all__ = [
    "ToolConfiguration",
    "ToolEntry",
    "ToolSet",
    "ToolSetInput",
    "ToolSpec",
    "SESSION_MEMORY_TOOL_TYPES",
    "all_tools",
    "memory_tools",
]

T = TypeVar("T", bound=Hashable)
_logger = Logger(__name__)


@dataclass(frozen=True, init=False)
class ToolConfiguration:
    """Pair one tool implementation with its effective action capabilities."""

    type: type[Tool]
    capabilities: ToolCapabilities

    def __init__(
        self,
        type: type[Tool],
        capabilities: ToolCapabilities | None = None,
    ) -> None:
        """Validate the implementation and bind an optional restriction."""
        if not isinstance(type, builtins.type) or not issubclass(type, Tool):
            _logger.error(
                "Rejected non-tool ToolConfiguration entry",
                value=repr(type),
            )
            raise TypeError("type must be a Tool subclass")
        if capabilities is not None and not isinstance(
            capabilities,
            ToolCapabilities,
        ):
            raise TypeError("capabilities must be ToolCapabilities or None")

        resolved = type.configure_capabilities(capabilities)
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "capabilities", resolved)
        _logger.trace(
            "Configured tool",
            tool_id=type.TOOL_ID,
            enabled_actions=resolved.enabled_actions,
        )

    def __eq__(self, other: object) -> bool:
        """Compare configurations while supporting legacy class membership."""
        if isinstance(other, ToolConfiguration):
            return (
                self.type is other.type
                and self.capabilities == other.capabilities
            )
        if isinstance(other, builtins.type) and issubclass(other, Tool):
            return self.type is other
        return False

    def __hash__(self) -> int:
        """Hash by type to remain compatible with legacy class equality."""
        return hash(self.type)


# Concise aliases for callers that prefer spec/entry terminology.
ToolSpec = ToolConfiguration
ToolEntry = ToolConfiguration


ToolSetInput = ToolConfiguration | type[Tool]


@dataclass(frozen=True, init=False)
class ToolSet:
    """Tool implementations made available by a workflow.

    Entries are identified by their stable ``TOOL_ID``. The public function
    name exposed to a model is resolved later, after an execution context is
    available, and must never be used to select a registered implementation.
    """

    core_tools: tuple[ToolConfiguration, ...]
    deferred_tools: tuple[ToolConfiguration, ...]

    def __init__(
        self,
        core_tools: tuple[ToolSetInput, ...],
        deferred_tools: tuple[ToolSetInput, ...],
    ) -> None:
        """Normalize class inputs and validate the complete tool collection."""
        if not isinstance(core_tools, tuple):
            raise TypeError("core_tools must be a tuple")
        if not isinstance(deferred_tools, tuple):
            raise TypeError("deferred_tools must be a tuple")

        normalized_core = tuple(_configure_tool(entry) for entry in core_tools)
        normalized_deferred = tuple(
            _configure_tool(entry) for entry in deferred_tools
        )
        registered = normalized_core + normalized_deferred

        duplicate_types = _duplicates(
            tuple(configuration.type for configuration in registered)
        )
        if duplicate_types:
            names = ", ".join(
                tool_type.__name__ for tool_type in duplicate_types
            )
            _logger.error("Duplicate tool implementations", names=names)
            raise ValueError(f"Duplicate tool implementations: {names}")

        ids = tuple(configuration.type.TOOL_ID for configuration in registered)
        duplicate_ids = _duplicates(ids)
        if duplicate_ids:
            _logger.error("Duplicate internal tool IDs", ids=duplicate_ids)
            raise ValueError(
                "Duplicate internal tool IDs: " + ", ".join(duplicate_ids)
            )

        object.__setattr__(self, "core_tools", normalized_core)
        object.__setattr__(self, "deferred_tools", normalized_deferred)
        _logger.debug(
            "Initialized tool set",
            core=len(normalized_core),
            deferred=len(normalized_deferred),
        )

    @property
    def core_tool_ids(self) -> frozenset[str]:
        """Handle core tool ids."""
        return frozenset(entry.type.TOOL_ID for entry in self.core_tools)

    @property
    def deferred_tool_ids(self) -> frozenset[str]:
        """Handle deferred tool ids."""
        return frozenset(entry.type.TOOL_ID for entry in self.deferred_tools)

    def get_configuration_with_id(
        self,
        tool_id: str,
    ) -> ToolConfiguration | None:
        """Return the type-and-capabilities entry for one stable tool ID."""
        for configuration in self.allowed_configurations():
            if configuration.type.TOOL_ID == tool_id:
                _logger.trace("Resolved tool configuration", tool_id=tool_id)
                return configuration
        _logger.trace("Tool configuration was not registered", tool_id=tool_id)
        return None

    def get_tool_with_id(self, tool_id: str) -> type[Tool] | None:
        """Return one registered type for compatibility with existing callers."""
        configuration = self.get_configuration_with_id(tool_id)
        return None if configuration is None else configuration.type

    def allowed_configurations(self) -> tuple[ToolConfiguration, ...]:
        """Return all entries with both implementation and capabilities."""
        return self.core_tools + self.deferred_tools

    def allowed_tools(self) -> tuple[type[Tool], ...]:
        """Return all registered implementation types for compatibility."""
        return tuple(entry.type for entry in self.allowed_configurations())

    def is_core_tool(self, tool_type: type[Tool]) -> bool:
        """Return whether is core tool."""
        return any(entry.type is tool_type for entry in self.core_tools)

    def is_deferred_tool(self, tool_type: type[Tool]) -> bool:
        """Return whether is deferred tool."""
        return any(entry.type is tool_type for entry in self.deferred_tools)

    # Compatibility for callers using the historical misspelling.
    def is_deffered_tool(self, tool_type: type[Tool]) -> bool:
        """Return whether is deffered tool."""
        return self.is_deferred_tool(tool_type)

    def is_allowed_tool(self, tool_type: type[Tool]) -> bool:
        """Return whether is allowed tool."""
        return self.is_core_tool(tool_type) or self.is_deferred_tool(tool_type)


def _duplicates(values: tuple[T, ...]) -> tuple[T, ...]:
    """Handle duplicates."""
    seen: set[T] = set()
    duplicates: list[T] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _configure_tool(entry: ToolSetInput) -> ToolConfiguration:
    """Normalize a legacy tool class or preserve an explicit configuration."""
    if isinstance(entry, ToolConfiguration):
        return entry
    return ToolConfiguration(entry)


def memory_tools(
    *,
    exclude: tuple[type[Tool], ...] = (),
) -> tuple[type[Tool], ...]:
    """Return concrete conversation-memory tools excluding selected tools."""

    excluded = set(exclude)
    invalid = tuple(
        tool for tool in excluded if tool not in SESSION_MEMORY_TOOL_TYPES
    )
    if invalid:
        names = ", ".join(tool.__name__ for tool in invalid)
        _logger.warning("Ignored non-memory exclusions", names=names)
    selected = tuple(
        tool for tool in SESSION_MEMORY_TOOL_TYPES if tool not in excluded
    )
    _logger.debug(
        "Selected conversation-memory tools",
        selected=tuple(tool.TOOL_ID for tool in selected),
        excluded=tuple(
            tool.TOOL_ID for tool in SESSION_MEMORY_TOOL_TYPES if tool in excluded
        ),
    )
    return selected


SESSION_MEMORY_TOOL_TYPES: tuple[type[Tool], ...] = (
    RequirementTool,
    AcceptanceCriteriaTool,
    ScopeTool,
    ConstraintTool,
    FactTool,
    DecisionTool,
    TodoTool,
    ChangeTool,
    VerificationTool,
    IssueTool,
    WorkingStateTool,
    CheckpointTool,
)


_CORE_TOOL_TYPES: tuple[type[Tool], ...] = (
    Read,
    Write,
    Edit,
    Glob,
    Tree,
    Bash,
    Workspace,
    PromptUser,
    Lsp,
    SkillTool,
    *memory_tools(),
)

_DEFERRED_TOOL_TYPES: tuple[type[Tool], ...] = (
    Git,
    Subprocess,
    Browser,
    WebSearch,
    Document,
    Diagram,
    ReadImage,
    SubagentTool,
)


def all_tools(
    are_deferred: bool | None = None,
    excluded: set[type[Tool]] | frozenset[type[Tool]] = frozenset(),
) -> tuple[type[Tool], ...]:
    """Return the default core, deferred, or complete tool collection."""

    if are_deferred is True:
        registry = _DEFERRED_TOOL_TYPES
    elif are_deferred is False:
        registry = _CORE_TOOL_TYPES
    else:
        registry = _CORE_TOOL_TYPES + _DEFERRED_TOOL_TYPES

    return tuple(tool_type for tool_type in registry if tool_type not in excluded)
