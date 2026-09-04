"""Contract tests for declarative per-tool action capabilities."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, override

import pytest

from citra.agent import AgentSession
from citra.tools.capabilities import (
    InvalidToolCapabilities,
    ToolCapabilities,
)
from citra.tools.default_registry import ToolConfiguration, ToolSet
from citra.tools.tool import InvalidToolArguments, Tool, ToolDefinition
from citra.tools.tool_registry import ToolRegistry
from citra.utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)


class _ActionTool(Tool):
    """Provide a small action-based tool for capability contract tests."""

    TOOL_ID = "test_action_tool"
    CAPABILITIES = ToolCapabilities(actions=("read", "write", "remove"))
    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="test_action_tool",
            description="Exercise action capability restrictions.",
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            enum=("read", "write", "remove"),
                        ),
                    ),
                ),
            ),
        )
    )

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: Any,
    ) -> tuple[ToolDefinition, ...]:
        """Return the test tool's single model-facing definition."""
        del context
        return (ToolDefinition(definition=cls.DEFINITION),)

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        """Echo the validated test action."""
        return str(arguments["action"])


class _AtomicTool(Tool):
    """Provide a non-action tool for invalid-restriction tests."""

    TOOL_ID = "test_atomic_tool"
    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="test_atomic_tool",
            description="Exercise atomic tool configuration.",
            parameters=JsonSchema.object(),
        )
    )

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: Any,
    ) -> tuple[ToolDefinition, ...]:
        """Return the test tool's single model-facing definition."""
        del context
        return (ToolDefinition(definition=cls.DEFINITION),)

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        """Return a fixed result after argument validation."""
        del arguments
        return "ok"


def _context(model_id: str = "test-model") -> SimpleNamespace:
    """Create the minimum execution context needed by the fixture tools."""
    return SimpleNamespace(
        config=SimpleNamespace(
            model=lambda: SimpleNamespace(id=model_id),
        ),
    )


def _action_enum(tool: Tool) -> tuple[str, ...]:
    """Read the action enum from one model-facing tool definition."""
    action = next(
        prop
        for prop in tool.definition.function.parameters.properties
        if prop.name == "action"
    )
    return tuple(str(value) for value in action.schema.enum)


def test_toolset_exposes_type_and_bound_capabilities() -> None:
    """Normalize legacy class entries and expose the richer entry shape."""
    toolset = ToolSet(core_tools=(_ActionTool,), deferred_tools=())

    entry = toolset.core_tools[0]
    assert entry.type is _ActionTool
    assert entry.capabilities.actions == ("read", "write", "remove")
    assert entry.capabilities.enabled_actions == ("read", "write", "remove")
    assert _ActionTool in toolset.core_tools
    assert _ActionTool in set(toolset.core_tools)
    assert toolset.allowed_tools() == (_ActionTool,)


def test_inclusion_restricts_schema_and_execution() -> None:
    """Advertise and execute only actions named by an inclusion restriction."""
    toolset = ToolSet(
        core_tools=(
            ToolConfiguration(
                _ActionTool,
                ToolCapabilities(include=("read", "write")),
            ),
        ),
        deferred_tools=(),
    )
    tool = ToolRegistry(toolset).instantiate(
        _context(),
        AgentSession(),
    )[_ActionTool.TOOL_ID]

    assert _action_enum(tool) == ("read", "write")
    assert tool.execute({"action": "read"}) == "read"
    with pytest.raises(InvalidToolArguments, match="disabled"):
        tool.execute({"action": "remove"})


def test_exclusion_restricts_schema_and_execution() -> None:
    """Advertise all declared actions except an explicitly excluded subset."""
    configuration = ToolConfiguration(
        _ActionTool,
        ToolCapabilities(exclude=("remove",)),
    )
    tool = _ActionTool(_context())
    tool.rebind_capabilities(configuration.capabilities)

    assert configuration.capabilities.enabled_actions == ("read", "write")
    assert _action_enum(tool) == ("read", "write")
    with pytest.raises(InvalidToolArguments, match="remove"):
        tool.validate_arguments({"action": "remove"})


def test_inclusion_and_exclusion_are_mutually_exclusive() -> None:
    """Reject an ambiguous capability option using both selection modes."""
    with pytest.raises(InvalidToolCapabilities, match="inclusion or exclusion"):
        ToolCapabilities(include=("read",), exclude=("remove",))


def test_unknown_and_empty_restrictions_are_rejected() -> None:
    """Reject unsupported actions and restrictions that disable everything."""
    with pytest.raises(InvalidToolCapabilities, match="Unsupported"):
        ToolConfiguration(
            _ActionTool,
            ToolCapabilities(include=("unknown",)),
        )
    with pytest.raises(InvalidToolCapabilities, match="at least one"):
        ToolConfiguration(
            _ActionTool,
            ToolCapabilities(exclude=("read", "write", "remove")),
        )


def test_atomic_tools_cannot_receive_action_restrictions() -> None:
    """Reject include/exclude options for tools without declared actions."""
    with pytest.raises(InvalidToolCapabilities, match="no selectable"):
        ToolConfiguration(
            _AtomicTool,
            ToolCapabilities(include=("read",)),
        )


def test_model_action_aliases_preserve_public_spelling() -> None:
    """Filter model-specific action names through canonical capabilities."""
    declaration = ToolCapabilities(
        actions=("definition", "references"),
        action_arguments=("action", "operation"),
        aliases=(
            ("goToDefinition", "definition"),
            ("findReferences", "references"),
        ),
    )
    restricted = ToolCapabilities(include=("references",)).bind(declaration)
    definition = ChatCompletionTool(
        function=FunctionDefinition(
            name="native_lsp",
            description="Exercise model-specific action spellings.",
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="operation",
                        schema=JsonSchema.string(
                            enum=("goToDefinition", "findReferences"),
                        ),
                    ),
                ),
            ),
        )
    )

    filtered = restricted.apply_to_definition(definition)
    operation = filtered.function.parameters.properties[0]
    assert operation.schema.enum == ("findReferences",)
    restricted.validate_arguments({"operation": "findReferences"})
    with pytest.raises(InvalidToolCapabilities, match="goToDefinition"):
        restricted.validate_arguments({"operation": "goToDefinition"})


def test_restriction_adds_enum_when_base_schema_is_open() -> None:
    """Materialize an enum for tools whose unrestricted schema is open-ended."""
    capabilities = ToolCapabilities(
        actions=("open", "close", "evaluate"),
        include=("open", "close"),
        action_arguments=("operation",),
    )
    definition = ChatCompletionTool(
        function=FunctionDefinition(
            name="open_schema_tool",
            description="Exercise open action schemas.",
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="operation",
                        schema=JsonSchema.string(),
                    ),
                ),
            ),
        )
    )

    restricted = capabilities.apply_to_definition(definition)
    assert restricted.function.parameters.properties[0].schema.enum == (
        "open",
        "close",
    )


def test_tool_execution_logs_the_concrete_source_origin(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Attach the concrete tool module to lifecycle log records."""
    tool = _ActionTool(_context())

    with caplog.at_level(logging.INFO):
        tool.execute({"action": "read"})

    lifecycle_records = [
        record
        for record in caplog.records
        if record.name.endswith("test_tool_capabilities")
    ]
    assert lifecycle_records
    assert all(vars(record)["origin"] == __name__ for record in lifecycle_records)
