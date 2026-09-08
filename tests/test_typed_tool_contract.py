"""Coverage for the typed, model-independent tool boundary."""

from dataclasses import is_dataclass
from types import SimpleNamespace
from typing import Any, cast

from citra.tools.default_registry import all_tools
from citra.tools.explorer import Read
from citra.tools.tool import ToolArguments
from citra.tools.tool_group import ToolGroup
from citra.utils.chat_completions_api import parse_model_response


def _context(model_id: str = "test-model") -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(model=lambda: SimpleNamespace(id=model_id)),
    )


def test_every_registered_tool_has_one_bound_argument_record() -> None:
    for tool_type in all_tools():
        arguments_type = tool_type.arguments_type()
        assert issubclass(arguments_type, ToolArguments)
        assert is_dataclass(arguments_type)
        assert arguments_type.tool_type() is tool_type


def test_definitions_do_not_change_with_model_id() -> None:
    for tool_type in all_tools():
        first = tool_type.resolve_definition_for_context(cast(Any, _context("claude")))
        second = tool_type.resolve_definition_for_context(cast(Any, _context("gpt-codex")))
        assert first == second


def test_provider_tool_calls_are_typed_before_history_storage() -> None:
    tool = Read(cast(Any, _context()))
    response = parse_model_response(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "read",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"read": tool},
    )

    arguments = response.assistant.tool_calls[0].arguments
    assert isinstance(arguments, Read.arguments_type())
    assert arguments.to_dict() == {"path": "README.md"}


def test_every_registered_tool_belongs_to_exactly_one_group() -> None:
    for tool_type in all_tools():
        groups = [group for group in ToolGroup._GROUPS if group.owns(tool_type)]
        assert len(groups) == 1, tool_type.TOOL_ID
