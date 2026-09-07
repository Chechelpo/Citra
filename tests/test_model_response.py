from __future__ import annotations

import pytest

from citra.agent import AssistantMessage, ReasoningMetadata, ToolCall
from citra.utils.chat_completions_api import (
    ModelResponseParseError,
    ModelUsage,
    parse_model_response,
)


def test_parse_model_response_builds_immutable_contract() -> None:
    response = parse_model_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_details": [{"type": "summary", "text": "check"}],
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "read", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        }
    )

    assert response.assistant == AssistantMessage(
        tool_calls=(ToolCall("call-1", "read", "{}"),),
        reasoning=ReasoningMetadata(
            details=({"type": "summary", "text": "check"},)
        ),
    )
    assert response.finish_reason == "tool_calls"
    assert response.usage == ModelUsage(10, 4, 14)


def test_parse_model_response_rejects_duplicate_tool_call_ids() -> None:
    with pytest.raises(ModelResponseParseError, match="Duplicate tool call id"):
        parse_model_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "duplicate",
                                    "function": {"name": "read", "arguments": "{}"},
                                },
                                {
                                    "id": "duplicate",
                                    "function": {"name": "grep", "arguments": "{}"},
                                },
                            ],
                        }
                    }
                ]
            }
        )
