from __future__ import annotations

import json
import io
from types import SimpleNamespace
from typing import Any, cast
import urllib.error

import pytest

from citra.agent import (
    AssistantMessage,
    ReasoningMetadata,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from citra.config import ModelConfig, RetryConfig
from citra.context import ExecutionContext
from citra.utils.chat_completions_api import ModelCall, call_api
from citra.utils.chat_completions_api.persistent_requests import _should_retry_http_error


def test_openrouter_stealth_provider_error_is_retryable() -> None:
    body = json.dumps(
        {
            "error": {
                "message": "Provider returned error",
                "code": 400,
                "metadata": {
                    "provider_name": "Stealth",
                    "raw": "[Stealth] ERROR",
                },
            }
        }
    )
    assert _should_retry_http_error(400, body)


def test_structured_router_fallback_error_is_retryable() -> None:
    body = json.dumps(
        {
            "error": {
                "type": "provider_error",
                "metadata": {"provider_code": "all_fallbacks_failed"},
            }
        }
    )
    assert _should_retry_http_error(400, body)


def test_invalid_context_request_is_not_retried() -> None:
    body = json.dumps(
        {
            "error": {
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "message": "Maximum context length exceeded",
            }
        }
    )
    assert not _should_retry_http_error(400, body)


def test_unknown_bad_request_and_auth_are_not_retried() -> None:
    assert not _should_retry_http_error(400, '{"error":{"message":"bad schema"}}')
    assert not _should_retry_http_error(401, "unauthorized")


def test_rate_limits_and_server_errors_are_retryable() -> None:
    assert _should_retry_http_error(429, "")
    assert _should_retry_http_error(503, "")


class _Response:
    status = 200

    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self) -> bytes:
        return self._body


def _model_config(*, api_keys: tuple[str, ...] = ()) -> ModelConfig:
    return ModelConfig(
        host="https://openrouter.example/v1",
        encrypted_key="",
        id="test-model",
        max_input_tokens=1_000,
        max_output_tokens=128,
        reasoning_effort=None,
        retry=RetryConfig(
            max_attempts=3,
            request_timeout=1,
            initial_backoff=0,
            max_backoff=0,
        ),
        _plaintext_api_key="test",
        _plaintext_api_keys=api_keys,
    )


def _context() -> ExecutionContext:
    return cast(ExecutionContext, cast(Any, SimpleNamespace()))


def _model_call() -> ModelCall:
    """Build a complete typed provider request for retry tests."""
    return ModelCall(
        context=_context(),
        messages=(),
        tools={},
        system_prompt="system",
        model_config=_model_config(),
    )


def test_model_call_detaches_mutable_request_containers() -> None:
    tools: dict[str, Any] = {}
    request = ModelCall(
        context=_context(),
        messages=(UserMessage("hello"),),
        tools=tools,
        system_prompt="system",
        model_config=_model_config(),
    )

    tools["late"] = object()

    assert tuple(request.tools) == ()
    with pytest.raises(TypeError):
        cast(dict[str, Any], request.tools)["late"] = object()


def test_call_api_retries_openrouter_provider_400(monkeypatch) -> None:
    body = json.dumps(
        {
            "error": {
                "message": "Provider returned error",
                "metadata": {
                    "provider_name": "Stealth",
                    "raw": "[Stealth] ERROR",
                },
            }
        }
    ).encode("utf-8")
    responses = iter(
        (
            urllib.error.HTTPError(
                "https://openrouter.example/v1/chat/completions",
                400,
                "Bad Request",
                {},
                io.BytesIO(body),
            ),
            _Response(
                {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            ),
        )
    )
    calls = 0

    def urlopen(*_, **__):
        nonlocal calls
        calls += 1
        value = next(responses)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(
        "citra.utils.chat_completions_api.persistent_requests.urllib.request.urlopen",
        urlopen,
    )
    result = call_api(_model_call())
    assert result.assistant.content == "ok"
    assert calls == 2


def test_call_api_rotates_key_after_three_retryable_failures(monkeypatch) -> None:
    authorizations: list[str] = []

    def urlopen(request, **_keywords):
        authorizations.append(request.get_header("Authorization"))
        if len(authorizations) <= 3:
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "Service Unavailable",
                {},
                io.BytesIO(b'{"error":{"message":"temporary"}}'),
            )
        return _Response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    monkeypatch.setattr(
        "citra.utils.chat_completions_api.persistent_requests.urllib.request.urlopen",
        urlopen,
    )
    model_call = _model_call()
    model_call = ModelCall(
        context=model_call.context,
        messages=model_call.messages,
        tools=model_call.tools,
        system_prompt=model_call.system_prompt,
        model_config=_model_config(api_keys=("primary", "backup")),
        max_attempts=4,
    )

    result = call_api(model_call)

    assert result.assistant.content == "ok"
    assert authorizations == [
        "Bearer primary",
        "Bearer primary",
        "Bearer primary",
        "Bearer backup",
    ]


def test_call_api_does_not_rotate_key_for_nonretryable_failure(monkeypatch) -> None:
    authorizations: list[str] = []

    def urlopen(request, **_keywords):
        authorizations.append(request.get_header("Authorization"))
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":{"message":"invalid key"}}'),
        )

    monkeypatch.setattr(
        "citra.utils.chat_completions_api.persistent_requests.urllib.request.urlopen",
        urlopen,
    )
    model_call = _model_call()
    model_call = ModelCall(
        context=model_call.context,
        messages=model_call.messages,
        tools=model_call.tools,
        system_prompt=model_call.system_prompt,
        model_config=_model_config(api_keys=("primary", "backup")),
    )

    with pytest.raises(RuntimeError, match="HTTP 401"):
        call_api(model_call)

    assert authorizations == ["Bearer primary"]


def test_call_api_serializes_typed_history_at_wire_boundary(monkeypatch) -> None:
    captured_payload: dict[str, Any] = {}

    def urlopen(request, **_keywords) -> _Response:
        captured_payload.update(json.loads(request.data))
        return _Response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    monkeypatch.setattr(
        "citra.utils.chat_completions_api.persistent_requests.urllib.request.urlopen",
        urlopen,
    )
    request = ModelCall(
        context=_context(),
        messages=(
            UserMessage("inspect"),
            AssistantMessage(
                tool_calls=(ToolCall("call-1", "read", '{"path":"README.md"}'),),
                reasoning=ReasoningMetadata(details=({"type": "summary"},)),
            ),
            ToolResultMessage("call-1", "contents"),
        ),
        tools={},
        system_prompt="system",
        model_config=_model_config(),
    )

    response = call_api(request)

    assert response.assistant == AssistantMessage(content="ok")
    assert captured_payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"path":"README.md"}',
                    },
                }
            ],
            "reasoning_details": [{"type": "summary"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "contents"},
    ]


def test_call_api_does_not_retry_permanent_bad_request(monkeypatch) -> None:
    body = b'{"error":{"type":"invalid_request_error","message":"invalid tool"}}'
    calls = 0

    def urlopen(*_, **__):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            "https://openrouter.example/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(body),
        )

    monkeypatch.setattr(
        "citra.utils.chat_completions_api.persistent_requests.urllib.request.urlopen",
        urlopen,
    )
    with pytest.raises(RuntimeError, match="invalid tool"):
        call_api(_model_call())
    assert calls == 1


def test_call_api_routes_debug_output_through_cli_renderer(monkeypatch) -> None:
    rendered: list[str] = []

    def urlopen(*_arguments, **_keywords) -> _Response:
        return _Response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    monkeypatch.setattr(
        "citra.utils.chat_completions_api.persistent_requests.urllib.request.urlopen",
        urlopen,
    )
    monkeypatch.setattr(
        "citra.utils.chat_completions_api.persistent_requests.render_model_debug",
        rendered.append,
    )

    call_api(_model_call())

    assert len(rendered) == 3
    assert rendered[0].startswith("Starting model request")
    assert rendered[1].startswith("Model HTTP 200 received")
    assert rendered[2].startswith("Model finish_reason(s)")
