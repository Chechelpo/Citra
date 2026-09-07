from __future__ import annotations

import json
import io
from types import SimpleNamespace
from typing import Any, cast
import urllib.error

import pytest

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


def _model_config() -> ModelConfig:
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
        messages=({"role": "user", "content": "hello"},),
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
