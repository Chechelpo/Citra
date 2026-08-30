from __future__ import annotations


from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import random
import socket
import ssl
import time
from typing import TYPE_CHECKING, Any, Callable, cast
import urllib.error
import urllib.request

from openai.types.chat import ChatCompletionSystemMessageParam

from ...agent import ChatMessage
from ...context import ExecutionContext
from ...tools.session_memory import MemoryTool
from ...tools.tool import Tool
from ..api import chat_completions_url
from ..prompt import build_system_prompt
from .model_normalization import normalize_model_response

if TYPE_CHECKING:
    from ...context import ModelConfig

from ..terminal import (
    BLUE,
    BOLD,
    CYAN,
    DIM,
    GREEN,
    RED,
    RESET,
    YELLOW,
    separator,
)


# Set to False to immediately fail on HTTP 429 responses instead of
# respecting rate-limit responses and retrying them.
RETRY_ON_RATE_LIMIT: bool = True

# Grey (DIM) diagnostic lines printed around each model request.
# Toggled at runtime by the /debug REPL command.
DEBUG_PRINTING: bool = True

# Despite the historical name, this is used as the maximum number of
# total request attempts, including the initial request.
DEFAULT_MAX_RETRIES: int = 12


def _debug_print(message: str) -> None:
    """Print a grey diagnostic line when debug printing is enabled."""
    if DEBUG_PRINTING:
        print(
            f"{DIM}"
            f"{message}"
            f"{RESET}"
        )


class ModelRequestInterrupted(RuntimeError):
    """A pending retry was superseded by newly queued user steering."""


# HTTP 4xx responses normally indicate a problem with the request and
# should therefore not be retried automatically.
#
# These are exceptions which commonly represent temporary conditions:
#
#   408 Request Timeout
#       The server did not receive the request in time.
#
#   409 Conflict
#       Some APIs use this for temporary resource/locking conflicts.
#       The OpenAI SDK also treats 409 as retryable.
#
#   421 Misdirected Request
#       A proxy/load-balancer may have routed the request incorrectly.
#
#   423 Locked
#       The target resource may only be temporarily locked.
#
#   424 Failed Dependency
#       An upstream dependency may temporarily be unavailable.
#
#   425 Too Early
#       The server explicitly asks the client to retry later.
#
#   429 Too Many Requests
#       Rate limiting. Controlled separately by RETRY_ON_RATE_LIMIT.
#
# All HTTP 5xx responses are handled as temporary server-side failures
# and are retried below.
_RETRYABLE_CLIENT_HTTP_STATUS_CODES: frozenset[int] = frozenset(
    {
        408,
        409,
        421,
        423,
        424,
        425,
    }
)


def merge_consecutive_roles(
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    """
    Merge adjacent plain-text messages with the same role.

    Protocol-bearing messages such as assistant tool calls and tool
    results are preserved exactly.
    """
    merged: list[ChatMessage] = []

    for message in messages:
        current = cast(
            ChatMessage,
            dict(message),
        )

        if not merged:
            merged.append(
                current
            )
            continue

        previous = merged[-1]

        if not _messages_are_mergeable(
            previous,
            current,
        ):
            merged.append(
                current
            )
            continue

        previous_content = previous.get(
            "content"
        )
        current_content = current.get(
            "content"
        )

        # _messages_are_mergeable() guarantees both content values are
        # either strings or None and that these messages contain only
        # role/content fields.
        previous_dict = cast(
            dict[str, Any],
            previous,
        )

        if not previous_content:
            previous_dict[
                "content"
            ] = current_content
            continue

        if not current_content:
            continue

        previous_dict[
            "content"
        ] = (
            f"{previous_content}\n\n"
            f"{current_content}"
        )

    return merged


def _messages_are_mergeable(
    first: ChatMessage,
    second: ChatMessage,
) -> bool:
    role = first.get(
        "role"
    )

    if role != second.get(
        "role"
    ):
        return False

    if role not in {
        "system",
        "user",
        "assistant",
    }:
        return False

    # Only merge ordinary role/content messages. Anything carrying
    # protocol metadata must remain structurally intact.
    if set(first) - {
        "role",
        "content",
    }:
        return False

    if set(second) - {
        "role",
        "content",
    }:
        return False

    first_content = first.get(
        "content"
    )
    second_content = second.get(
        "content"
    )

    if (
        first_content is not None
        and not isinstance(
            first_content,
            str,
        )
    ):
        return False

    if (
        second_content is not None
        and not isinstance(
            second_content,
            str,
        )
    ):
        return False

    return True


def system_prompt(
    context: ExecutionContext,
) -> str:
    return build_system_prompt(
        context
    )


def _backoff_delay(
    attempt: int,
    initial: float,
    maximum: float,
) -> float:
    """
    Calculate an exponential retry delay with positive jitter.

    The delay grows exponentially:

        initial, initial * 2, initial * 4, ...

    until ``maximum`` is reached. Up to 25% positive random jitter is
    then added so that multiple clients which fail at the same time do
    not continuously retry in lockstep.

    ``attempt`` is the number of the request which just failed and is
    expected to start at 1.
    """
    base = min(
        initial
        * (
            2
            ** (
                attempt
                - 1
            )
        ),
        maximum,
    )

    jitter = random.uniform(
        0,
        base * 0.25,
    )

    return (
        base
        + jitter
    )


def _retry_after_error(
    attempt: int,
    max_attempts: int,
    initial_backoff: float,
    max_backoff: float,
    error: Exception,
    *,
    reason: str,
    retry_after: float | None = None,
    interrupt: Callable[[], bool] | None = None,
) -> None:
    """
    Apply retry backoff or raise once attempts are exhausted.
    """
    if interrupt is not None and interrupt():
        raise ModelRequestInterrupted(
            "Model retry interrupted by user steering."
        ) from error

    if attempt >= max_attempts:
        raise RuntimeError(
            f"Model API {reason} after "
            f"{max_attempts} attempts."
        ) from error

    delay = _backoff_delay(
        attempt=attempt,
        initial=initial_backoff,
        maximum=max_backoff,
    )

    if retry_after is not None:
        delay = max(
            delay,
            retry_after,
        )

    print(
        f"{YELLOW}"
        f"⏺ Model request {reason}. "
        f"Retrying in {delay:.1f}s "
        f"(attempt "
        f"{attempt + 1}/{max_attempts})..."
        f"{RESET}"
    )

    deadline = time.monotonic() + delay
    while True:
        if interrupt is not None and interrupt():
            raise ModelRequestInterrupted(
                "Model retry interrupted by user steering."
            ) from error
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.25, remaining))

def _retry_after_from_http_date(
    value: str,
) -> float | None:
    """
    Parse an HTTP-date Retry-After value into a delay in seconds.

    RFC 9110 allows Retry-After to contain either a number of seconds
    or an absolute HTTP date. This helper handles the latter form.
    """
    try:
        retry_at = parsedate_to_datetime(
            value
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(
            tzinfo=timezone.utc
        )

    delay = (
        retry_at
        - datetime.now(
            timezone.utc
        )
    ).total_seconds()

    return max(
        0.0,
        delay,
    )


def _get_retry_after(
    error: urllib.error.HTTPError,
    body: str,
) -> float | None:
    """
    Extract a server-requested retry delay.

    Several representations are supported:

    1. ``retry-after-ms`` header, used by some OpenAI-compatible APIs.
    2. Standard ``Retry-After`` header containing seconds.
    3. Standard ``Retry-After`` header containing an HTTP date.
    4. Common JSON response fields such as
       ``error.metadata.retry_after_seconds``.

    Invalid values are ignored and normal exponential backoff is used
    instead.
    """
    headers = error.headers

    if headers is not None:
        retry_after_ms = headers.get(
            "retry-after-ms"
        )

        if retry_after_ms:
            try:
                value = float(
                    retry_after_ms
                )

                if value >= 0:
                    return (
                        value
                        / 1000.0
                    )
            except ValueError:
                pass

        retry_after_header = headers.get(
            "Retry-After"
        )

        if retry_after_header:
            try:
                value = float(
                    retry_after_header
                )

                if value >= 0:
                    return value
            except ValueError:
                pass

            retry_after_date = (
                _retry_after_from_http_date(
                    retry_after_header
                )
            )

            if retry_after_date is not None:
                return retry_after_date

    try:
        data = json.loads(
            body
        )
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return None

    if not isinstance(
        data,
        dict,
    ):
        return None

    error_data = data.get(
        "error"
    )

    if not isinstance(
        error_data,
        dict,
    ):
        error_data = {}

    metadata = error_data.get(
        "metadata"
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    candidates = (
        metadata.get(
            "retry_after_seconds"
        ),
        error_data.get(
            "retry_after_seconds"
        ),
        data.get(
            "retry_after_seconds"
        ),
    )

    for candidate in candidates:
        if candidate is None:
            continue

        try:
            value = float(
                candidate
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if value >= 0:
            return value

    return None

def _should_retry_http_status(
    status: int,
) -> bool:
    """
    Decide whether an HTTP status is generically retryable.

    This function handles status-only policy. Provider-specific exceptions
    which require inspecting the response body belong in
    ``_should_retry_http_error``.

    Policy:

    * HTTP 429 is retryable only when ``RETRY_ON_RATE_LIMIT`` is enabled.
    * A small allowlist of temporary 4xx statuses is retryable.
    * Every 5xx status is retryable.
    * All other statuses are treated as permanent by default.

    In particular, a normal HTTP 404 is *not* retryable. Some routers return
    HTTP 404 for a temporary downstream routing failure; that special case is
    recognized separately when the structured provider code is
    ``upstream_404``.
    """
    if status == 429:
        return RETRY_ON_RATE_LIMIT

    if status in _RETRYABLE_CLIENT_HTTP_STATUS_CODES:
        return True

    if 500 <= status <= 599:
        return True

    return False


def _http_error_fields(
    body: str,
) -> list[tuple[str, str]]:
    """
    Extract useful classifier fields from a router/provider error body.

    OpenAI-compatible routers may wrap the actual provider response. OpenRouter,
    for example, can place the downstream error in ``error.metadata.raw`` as a
    JSON-encoded string while leaving a generic top-level error message.

    The returned list contains ``(path, value)`` pairs for fields useful to
    retry classification, including ``type``, ``code``, ``message``, provider
    metadata, and nested serialized JSON found in ``raw``.

    The traversal is deliberately bounded so malformed or unexpectedly large
    error bodies cannot recurse indefinitely.
    """
    try:
        data = json.loads(
            body
        )
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        stripped = body.strip()

        if stripped:
            return [
                (
                    "body",
                    stripped,
                )
            ]

        return []

    fields: list[
        tuple[str, str]
    ] = []

    interesting_keys = {
        "code",
        "error_type",
        "message",
        "provider",
        "provider_code",
        "provider_name",
        "raw",
        "type",
    }

    def visit(
        value: Any,
        path: str,
        depth: int = 0,
    ) -> None:
        if depth > 6:
            return

        if isinstance(
            value,
            dict,
        ):
            for key, child in value.items():
                key_text = str(
                    key
                )
                normalized_key = key_text.casefold()

                child_path = (
                    f"{path}.{key_text}"
                    if path
                    else key_text
                )

                if isinstance(
                    child,
                    (dict, list),
                ):
                    visit(
                        child,
                        child_path,
                        depth + 1,
                    )
                    continue

                if child is None:
                    continue

                if normalized_key in interesting_keys:
                    fields.append(
                        (
                            child_path.casefold(),
                            str(child),
                        )
                    )

                # Routers frequently put the real provider error in a
                # JSON-encoded metadata.raw string. Parse that representation
                # as well so codes such as "upstream_404" are visible to the
                # classifier instead of remaining hidden in an opaque string.
                if (
                    normalized_key == "raw"
                    and isinstance(
                        child,
                        str,
                    )
                ):
                    try:
                        nested = json.loads(
                            child
                        )
                    except (
                        json.JSONDecodeError,
                        TypeError,
                        ValueError,
                    ):
                        continue

                    visit(
                        nested,
                        f"{child_path}.json",
                        depth + 1,
                    )

            return

        if isinstance(
            value,
            list,
        ):
            for index, child in enumerate(
                value[:20]
            ):
                visit(
                    child,
                    f"{path}[{index}]",
                    depth + 1,
                )

    visit(
        data,
        "",
    )

    return fields


def _normalized_error_codes(
    fields: list[tuple[str, str]],
) -> set[str]:
    """
    Return normalized structured error ``code`` and ``type`` values.

    Normalization lets providers use forms such as ``invalid-request-error``,
    ``invalid request error``, and ``invalid_request_error`` without requiring
    separate classifier entries.
    """
    return {
        value
        .casefold()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
        for key, value in fields
        if (
            key.endswith(
                "code"
            )
            or key.endswith(
                "type"
            )
        )
    }


def _is_gmicloud_insufficient_balance_error(
    status: int,
    body: str,
) -> bool:
    """Detect GMICloud's misleading balance failure on free-model routes.

    GMICloud can return HTTP 402 with an ``Insufficient balance`` message even
    when the selected route is intended to be free. Treat that narrow provider
    response as a transient upstream failure so it does not terminate the
    client immediately.
    """
    if status != 402:
        return False

    # Inspect the raw body as well as extracted classifier fields. Some router
    # payloads put ``Insufficient balance`` under a generic ``error`` key, which
    # is intentionally not one of _http_error_fields()'s classifier keys.
    fields = _http_error_fields(body)
    detail = " ".join(
        [body, *(value for _, value in fields)]
    ).casefold()

    return (
        "gmicloud" in detail
        and "insufficient balance" in detail
    )


def _should_retry_http_error(
    status: int,
    body: str,
) -> bool:
    """
    Decide whether a specific HTTP failure should be retried.

    Most retry policy is status-based, but routers can translate downstream
    provider failures into otherwise permanent-looking HTTP statuses. This
    function therefore inspects structured provider metadata before falling
    back to the generic status policy.

    Special cases:

    * ``402`` from GMICloud containing ``Insufficient balance`` is treated as
      a transient provider failure. This narrow exception prevents a free-model
      route from terminating the client on a provider-side billing gate.

    * ``404`` + structured code ``upstream_404`` is retryable. This means the
      router accepted the request but the selected provider's own upstream
      route was unavailable. An ordinary 404 remains permanent.

    * ``400`` is retryable only when structured codes or provider text clearly
      identify a transient router/provider condition. Invalid requests,
      authentication failures, context-length failures, and similar client
      errors fail immediately.

    The order of checks matters: the provider-specific ``upstream_404`` case is
    checked before generic 4xx handling.
    """
    fields = _http_error_fields(
        body
    )
    codes = _normalized_error_codes(
        fields
    )

    # GMICloud may emit a billing-looking HTTP 402 for a route that is
    # configured as free. Do not let that provider-specific response stop the
    # client; treat it like a transient upstream failure and continue through
    # the bounded retry policy.
    if _is_gmicloud_insufficient_balance_error(status, body):
        return True

    # Some providers report an internal/routing 404 using HTTP 404 plus an
    # explicit provider code. Retrying this narrow case is useful; retrying
    # arbitrary 404s is not.
    if (
        status == 404
        and "upstream_404" in codes
    ):
        return True

    if status != 400:
        return _should_retry_http_status(
            status
        )

    permanent_codes = {
        "authentication_error",
        "context_length_exceeded",
        "invalid_api_key",
        "invalid_request_error",
        "permission_denied",
        "unprocessable_entity",
    }

    if codes & permanent_codes:
        return False

    transient_codes = {
        "all_fallbacks_failed",
        "empty_response",
        "engine_overloaded",
        "model_not_available",
        "no_available_provider",
        "overloaded_error",
        "provider_error",
        "rate_limit_error",
        "server_error",
        "service_unavailable",
        "temporarily_unavailable",
        "upstream_error",
    }

    detail = " ".join(
        value
        for _, value in fields
    ).casefold()

    permanent_markers = (
        "authentication failed",
        "context length",
        "context_length_exceeded",
        "invalid api key",
        "invalid request",
        "invalid_request_error",
        "invalid tool",
        "maximum context",
        "permission denied",
        "unprocessable",
    )

    if any(
        marker in detail
        for marker in permanent_markers
    ):
        return False

    if codes & transient_codes:
        return True

    # Some routers map a temporary provider failure onto HTTP 400 while
    # retaining only a textual upstream hint rather than a structured code.
    transient_provider_markers = (
        "provider returned error",
        "upstream error",
        "upstream provider",
        "provider unavailable",
        "provider error",
        "all fallbacks failed",
        "no available provider",
        "no endpoints found",
        "model not available",
        "rate limit",
        "server error",
        "service unavailable",
        "temporarily unavailable",
        "overloaded",
    )

    return any(
        marker in detail
        for marker in transient_provider_markers
    )


def _http_retry_reason(
    status: int,
    body: str = "",
) -> str:
    """
    Return a concise human-readable reason for a retryable HTTP failure.

    The response body is inspected so provider-specific failures can receive a
    more accurate explanation than their outer HTTP status alone.
    """
    fields = _http_error_fields(
        body
    )
    codes = _normalized_error_codes(
        fields
    )

    if (
        status == 404
        and "upstream_404" in codes
    ):
        return (
            "hit a temporary upstream "
            "provider routing failure"
        )

    if _is_gmicloud_insufficient_balance_error(status, body):
        return (
            "ignored GMICloud's misleading insufficient-balance "
            "response on a free-model route"
        )

    reasons = {
        400: "hit a transient upstream provider error",
        408: "timed out at the server",
        409: "hit a temporary conflict",
        421: "was misdirected by the server",
        423: "hit a temporarily locked resource",
        424: "hit a failed upstream dependency",
        425: "was asked to retry later",
        429: "was rate limited",
        500: "hit an internal server error",
        502: "hit a bad gateway",
        503: "hit an unavailable server",
        504: "hit a gateway timeout",
    }

    return reasons.get(
        status,
        (
            "received retryable HTTP "
            f"{status}"
        ),
    )

def _is_retryable_url_error(
    error: urllib.error.URLError,
) -> bool:
    """
    Determine whether a urllib transport error is likely temporary.

    DNS failures, socket failures, connection resets/refusals, and
    similar operating-system network errors are treated as transient.

    Certificate verification errors are deliberately NOT retried:
    they normally indicate incorrect certificates, trust-store
    configuration, interception, or a hostname/configuration error and
    will not improve through repeated requests.
    """
    reason = error.reason

    if isinstance(
        reason,
        ssl.SSLCertVerificationError,
    ):
        return False

    if isinstance(
        reason,
        (
            TimeoutError,
            socket.timeout,
            socket.gaierror,
            ConnectionError,
        ),
    ):
        return True

    # urllib commonly wraps lower-level networking errors in OSError.
    # This includes many temporary resolver/socket failures.
    if isinstance(
        reason,
        OSError,
    ):
        return True

    return False

def validate_tool_history(
    messages: list[ChatMessage],
) -> None:
    pending: set[str] = set()

    for index, message in enumerate(
        messages
    ):
        role = message.get("role")

        if role == "assistant":
            if pending:
                raise ValueError(
                    f"Assistant message {index} "
                    "appeared before all previous "
                    f"tool calls were resolved: "
                    f"{sorted(pending)}"
                )

            tool_calls = message.get(
                "tool_calls"
            )

            if isinstance(
                tool_calls,
                list,
            ):
                for call in tool_calls:
                    if not isinstance(
                        call,
                        dict,
                    ):
                        raise ValueError(
                            f"Assistant message "
                            f"{index} contains an "
                            "invalid tool call."
                        )

                    call_id = call.get("id")

                    if not isinstance(
                        call_id,
                        str,
                    ):
                        raise ValueError(
                            f"Assistant message "
                            f"{index} contains a "
                            "tool call without an id."
                        )

                    if call_id in pending:
                        raise ValueError(
                            f"Duplicate tool call "
                            f"id {call_id!r}."
                        )

                    pending.add(
                        call_id
                    )

        elif role == "tool":
            call_id = message.get(
                "tool_call_id"
            )

            if not isinstance(
                call_id,
                str,
            ):
                raise ValueError(
                    f"Tool message {index} "
                    "has no tool_call_id."
                )

            if call_id not in pending:
                raise ValueError(
                    f"Tool message {index} "
                    "references unknown or "
                    "already-resolved tool call "
                    f"{call_id!r}."
                )

            pending.remove(
                call_id
            )

        elif pending:
            raise ValueError(
                f"Message {index} with role "
                f"{role!r} appeared while "
                "tool calls were unresolved: "
                f"{sorted(pending)}"
            )

    if pending:
        raise ValueError(
            "Conversation ends with unresolved "
            f"tool calls: {sorted(pending)}"
        )


def _http_error_detail(
    body: str,
) -> str:
    """
    Extract the most useful human-readable error text from an
    OpenAI-compatible HTTP error response.

    Routers such as OpenRouter may expose a generic top-level message while
    placing the downstream provider's actual response in
    ``error.metadata.raw``. ``raw`` is frequently a JSON-encoded string.

    Preference order:

    1. Nested provider message from ``metadata.raw``.
    2. Nested provider type/code when the provider message is empty.
    3. Top-level ``error.message``.
    4. The original response body.

    When provider metadata identifies the downstream provider, its name is
    included in the returned text.
    """
    if not body:
        return ""

    try:
        data = json.loads(
            body
        )
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return body.strip()

    if not isinstance(
        data,
        dict,
    ):
        return body.strip()

    error_data = data.get(
        "error"
    )

    if not isinstance(
        error_data,
        dict,
    ):
        return body.strip()

    metadata = error_data.get(
        "metadata"
    )

    if isinstance(
        metadata,
        dict,
    ):
        raw = metadata.get(
            "raw"
        )
        provider = (
            metadata.get(
                "provider_name"
            )
            or metadata.get(
                "provider"
            )
        )

        if raw:
            raw_text = str(
                raw
            ).strip()

            raw_data: Any = None

            if isinstance(
                raw,
                (dict, list),
            ):
                raw_data = raw
            else:
                try:
                    raw_data = json.loads(
                        raw_text
                    )
                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ):
                    raw_data = None

            if isinstance(
                raw_data,
                dict,
            ):
                nested_error = raw_data.get(
                    "error"
                )

                if isinstance(
                    nested_error,
                    dict,
                ):
                    nested_message = nested_error.get(
                        "message"
                    )
                    nested_type = nested_error.get(
                        "type"
                    )
                    nested_code = nested_error.get(
                        "code"
                    )

                    if nested_message:
                        raw_text = str(
                            nested_message
                        )
                    elif (
                        nested_type
                        and nested_code
                    ):
                        raw_text = (
                            f"{nested_type}: "
                            f"{nested_code}"
                        )
                    elif nested_code:
                        raw_text = str(
                            nested_code
                        )
                    elif nested_type:
                        raw_text = str(
                            nested_type
                        )

                elif raw_data.get(
                    "message"
                ):
                    raw_text = str(
                        raw_data[
                            "message"
                        ]
                    )

            if provider:
                return (
                    f"[{provider}] "
                    f"{raw_text}"
                )

            return raw_text

    message = error_data.get(
        "message"
    )

    if message:
        return str(
            message
        )

    error_type = error_data.get(
        "type"
    )
    error_code = error_data.get(
        "code"
    )

    if (
        error_type
        and error_code
    ):
        return (
            f"{error_type}: "
            f"{error_code}"
        )

    if error_code:
        return str(
            error_code
        )

    if error_type:
        return str(
            error_type
        )

    return body.strip()

def _finish_reason_entries(
    value: Any,
) -> list[str]:
    """Return printable finish-reason entries for every response choice."""
    if not isinstance(value, dict):
        return ["<response is not an object>"]

    choices = value.get("choices")

    if not isinstance(choices, list) or not choices:
        return ["<no choices>"]

    entries: list[str] = []

    for index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            entries.append(
                f"{index}=<invalid choice>"
            )
            continue

        entries.append(
            f"{index}={choice.get('finish_reason')!r}"
        )

    return entries


def _log_finish_reasons(
    value: Any,
) -> None:
    """Log finish_reason for every decoded HTTP-200 completion response."""
    _debug_print(
        f"⏺ Model finish_reason(s): "
        f"{', '.join(_finish_reason_entries(value))}"
    )


def _choice_output_diagnostic(
    value: Any,
) -> str:
    """Describe response output shape without dumping model content."""
    if not isinstance(value, dict):
        return "response is not a JSON object"

    choices = value.get("choices")

    if not isinstance(choices, list):
        return "choices is not a list"

    if not choices:
        return "choices is empty"

    first = choices[0]

    if not isinstance(first, dict):
        return "choices[0] is not an object"

    message = first.get("message")

    if not isinstance(message, dict):
        return "choices[0].message is missing or invalid"

    content = message.get("content")
    tool_calls = message.get("tool_calls")

    if content is None:
        content_detail = "content=null"
    elif isinstance(content, str):
        content_detail = (
            "content=empty-string"
            if not content.strip()
            else f"content={len(content)} chars"
        )
    else:
        content_detail = (
            f"content_type={type(content).__name__}"
        )

    if tool_calls is None:
        tool_detail = "tool_calls=null"
    elif isinstance(tool_calls, list):
        tool_detail = f"tool_calls={len(tool_calls)}"
    else:
        tool_detail = (
            f"tool_calls_type={type(tool_calls).__name__}"
        )

    return f"{content_detail}, {tool_detail}"


def _has_usable_choice(value: Any) -> bool:
    """
    Check that the first choice contains an actual assistant action.

    A provider response is usable only when it contains non-whitespace text
    or at least one tool call. HTTP 200 responses with null/empty content and
    no tool calls are suspicious empty completions and must be retried rather
    than silently accepted as successful agent output.
    """
    if not isinstance(value, dict):
        return False

    choices = value.get("choices")

    if not isinstance(choices, list) or not choices:
        return False

    first = choices[0]

    if not isinstance(first, dict):
        return False

    message = first.get("message")

    if not isinstance(message, dict):
        return False

    content = message.get("content")
    tool_calls = message.get("tool_calls")

    has_content = (
        isinstance(content, str)
        and bool(content.strip())
    )
    has_tool_calls = (
        isinstance(tool_calls, list)
        and bool(tool_calls)
    )

    return has_content or has_tool_calls


def _is_stealth_continue_work_error(
    status: int,
    body: str,
) -> bool:
    """
    Detect the provider-specific ``[Stealth] ERROR`` failure mode.

    This is deliberately *not* treated as an ordinary retry. Some stealth
    providers reject a hands-off/agentic continuation request but accept the
    same conversation once the client explicitly asks the model to continue.
    For that narrow case, call_api() gets one recovery request with an added
    user message: ``continue your work``.
    """
    if status not in {400, 422}:
        return False

    fields = _http_error_fields(
        body
    )
    detail = " ".join(
        value
        for _, value in fields
    ).casefold()

    return "[stealth] error" in detail


def _append_continue_work_message(
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    """Return a copied request history with an explicit continuation turn."""
    continued = list(messages)
    continued.append(
        cast(
            ChatMessage,
            {
                "role": "user",
                "content": "continue your work",
            },
        )
    )
    return continued


def _resolve_model_snapshot(
    context: ExecutionContext,
    model_config: "ModelConfig | Any | None",
) -> Any:
    """Resolve the model once for the lifetime of one HTTP request.

    ``ExecutionContext.config.model()`` is the canonical source. The fallback
    accepts the older direct ``context.model_config`` test/integration shape
    without allowing retries to re-resolve a different active profile.
    """
    if model_config is not None:
        return model_config

    config = getattr(context, "config", None)
    if config is not None:
        resolver = getattr(config, "model", None)
        if callable(resolver):
            return resolver()

    legacy = getattr(context, "model_config", None)
    if callable(legacy):
        legacy = legacy()
    if legacy is not None:
        return legacy

    raise AttributeError(
        "Execution context does not expose a model configuration."
    )


def _model_max_output_tokens(model: Any) -> int:
    value = getattr(model, "max_output_tokens", None)
    if value is None:
        value = getattr(model, "max_tokens", None)
    if value is None:
        raise AttributeError(
            "Model config does not define max_output_tokens."
        )
    return int(value)


def _model_api_key(model: Any) -> str:
    decrypt = getattr(model, "decrypt_api_key", None)
    if callable(decrypt):
        return str(decrypt())

    legacy = getattr(model, "api_key", None)
    if legacy is None:
        raise AttributeError(
            "Model config does not expose an API credential."
        )
    return str(legacy)


def call_api(
    context: ExecutionContext,
    messages: list[ChatMessage],
    tools: dict[str, Tool],
    reasoning_effort: str | None = None,
    *,
    model_config: ModelConfig | None = None,
    request_timeout: float | None = None,
    max_attempts: int | None = None,
    initial_backoff: float | None = None,
    max_backoff: float | None = None,
    retry_interrupt: Callable[[], bool] | None = None,
    sys_prompt: str | None = None
) -> dict[str, Any]:
    """
    Perform one OpenAI-compatible Chat Completions request.

    ``model_config`` may provide a pre-resolved immutable model snapshot.
    When omitted, the active model is resolved once before the request and
    reused for every retry attempt.

    Requests which fail because of a likely temporary provider,
    upstream, or network condition are retried with exponential
    backoff and jitter.

    Retryable HTTP responses include:

    * 408 Request Timeout
    * 409 Conflict
    * 421 Misdirected Request
    * 423 Locked
    * 424 Failed Dependency
    * 425 Too Early
    * 429 Too Many Requests, when RETRY_ON_RATE_LIMIT is enabled
    * all 5xx server responses

    Most other 4xx responses are considered permanent request errors
    and fail immediately. A narrow exception is made for GMICloud HTTP 402
    responses containing ``Insufficient balance``; those are ignored as a
    provider-side transient failure and retried within the normal attempt cap.
    Examples of permanent responses include:

    * 401 Unauthorized
    * 403 Forbidden
    * ordinary 404 Not Found responses
    * 405 Method Not Allowed
    * 413 Content Too Large
    * 415 Unsupported Media Type
    * 422 Unprocessable Content

    Retrying these would normally submit the same invalid request
    repeatedly without any chance of recovery.

    * 400 Bad Request is retried only when structured error codes or provider
      metadata identify a transient upstream/router failure. Invalid requests
      and context-length failures fail immediately.

    * ``[Stealth] ERROR`` is intentionally not handled as an ordinary retry.
      On the first HTTP 400/422 response containing that marker, one recovery
      request is made using a copied history with a final user message reading
      ``continue your work``. If that recovery receives the same stealth error,
      it is surfaced instead of entering the standard retry loop.

    * 404 Not Found is normally permanent. A narrow provider-specific
      exception is made when the structured downstream code is
      ``upstream_404``; this represents a router/provider upstream route
      failure rather than an invalid user endpoint.

    HTTP 200 responses are also validated. A response with no non-whitespace
    assistant text and no tool calls is considered a suspicious empty response
    and is retried with the normal backoff policy.

    The ``finish_reason`` of every decoded HTTP-200 response is logged, even
    when the response is empty/malformed and will be retried.

    Transport-level failures such as timeouts, temporary DNS errors,
    refused connections, dropped connections, and connection resets
    are also retried. TLS certificate verification failures are not.

    If a provider supplies Retry-After information, that value is
    honored as a minimum wait time in addition to the normal
    exponential backoff.

    ``max_attempts`` counts ordinary attempts, including the initial request.
    For example, ``max_attempts=3`` permits one initial request followed by at
    most two ordinary retries. The one-time ``continue your work`` stealth
    recovery is a changed request and does not consume an additional ordinary
    retry slot.

    Args:
        context:
            Current execution context containing the selected model
            configuration and authentication information.

        messages:
            Conversation messages to include after generated system
            context.

        tools:
            Tools exposed to the model. Memory tools may additionally
            contribute transient system context.

        reasoning_effort:
            Optional model-specific reasoning effort setting.

        request_timeout:
            Timeout, in seconds, for each individual HTTP request.

        max_attempts:
            Maximum total number of ordinary attempts, including the initial
            request. Defaults to the model retry configuration.

        initial_backoff:
            Initial retry delay in seconds before jitter.

        max_backoff:
            Maximum exponential backoff base in seconds. A
            server-provided Retry-After value may exceed this value.

        retry_interrupt:
            Optional callback checked during retry backoff. Returning true
            aborts the obsolete request so queued steering can rebuild it.

        sys_prompt:
            Optional system prompt for establishing constant system prompts.

    Returns:
        The decoded JSON response from the model API.

    Raises:
        RuntimeError:
            If a non-retryable HTTP/network error occurs or all retry
            attempts are exhausted.

        ValueError:
            If retry/request configuration is invalid.
    """
    model = _resolve_model_snapshot(context, model_config)
    retry_config = model.retry
    if max_attempts is None:
        max_attempts = retry_config.max_attempts
    if request_timeout is None:
        request_timeout = retry_config.request_timeout
    if initial_backoff is None:
        initial_backoff = retry_config.initial_backoff
    if max_backoff is None:
        max_backoff = retry_config.max_backoff
    if sys_prompt is None:
        sys_prompt = system_prompt(context)

    if max_attempts < 1:
        raise ValueError(
            "max_attempts must be at least 1."
        )

    if request_timeout <= 0:
        raise ValueError(
            "request_timeout must be greater than 0."
        )

    if initial_backoff < 0:
        raise ValueError(
            "initial_backoff cannot be negative."
        )

    if max_backoff < 0:
        raise ValueError(
            "max_backoff cannot be negative."
        )

    if initial_backoff > max_backoff:
        raise ValueError(
            "initial_backoff cannot exceed max_backoff."
        )

    system_messages: list[
        ChatCompletionSystemMessageParam
    ] = [
        {
            "role": "system",
            "content": sys_prompt,
        }
    ]

    memory_context = build_memory_context(
        tools
    )

    messages_to_merge: list[ChatMessage] = [
        *system_messages,
        *messages,
    ]

    request_messages = merge_consecutive_roles(
        messages_to_merge
    )

    request_messages = insert_memory_context(
        request_messages,
        memory_context,
    )

    request_messages = normalize_message_content(
        request_messages
    )

    validate_tool_history(
        request_messages
    )

    payload: dict[str, Any] = {
        "model": model.id,
        "max_tokens": _model_max_output_tokens(model),
        "messages": request_messages,
    }

    if reasoning_effort is not None:
        payload[
            "reasoning_effort"
        ] = reasoning_effort

    tool_definitions = [
        tool.get_as_tool()
        for tool in tools.values()
    ]

    if tool_definitions:
        payload[
            "tools"
        ] = tool_definitions

    url = chat_completions_url(
        model.host
    )

    active_request_messages = list(
        request_messages
    )
    stealth_continue_used = False
    recovery_request_pending = False
    attempt = 1

    while attempt <= max_attempts:
        payload[
            "messages"
        ] = active_request_messages

        request_data = json.dumps(
            payload
        ).encode(
            "utf-8"
        )

        request = urllib.request.Request(
            url,
            data=request_data,
            headers={
                "Content-Type": (
                    "application/json"
                ),
                "Authorization": (
                    f"Bearer {_model_api_key(model)}"
                ),
            },
            method="POST",
        )

        if recovery_request_pending:
            request_label = "stealth continuation recovery"
            recovery_request_pending = False
        else:
            request_label = "model request"

        _debug_print(
            f"⏺ Starting {request_label} "
            f"(attempt {attempt}/{max_attempts}, "
            f"model={model.id}, "
            f"timeout={request_timeout:.1f}s)"
        )

        started_at = time.monotonic()

        try:
            with urllib.request.urlopen(
                request,
                timeout=request_timeout,
            ) as response:
                raw_response = response.read().decode(
                    "utf-8",
                    errors="replace",
                )
                response_status = getattr(
                    response,
                    "status",
                    200,
                )

            elapsed = time.monotonic() - started_at
            _debug_print(
                f"⏺ Model HTTP {response_status} received "
                f"in {elapsed:.2f}s"
            )

            try:
                decoded = json.loads(raw_response)

                decoded = normalize_model_response(
                    decoded,
                    tools=tools,
                    model_id=model.id
                )

                _log_finish_reasons(decoded)

                if _has_usable_choice(decoded):
                    return decoded
            except json.JSONDecodeError as error:
                _retry_after_error(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    initial_backoff=initial_backoff,
                    max_backoff=max_backoff,
                    error=error,
                    reason="received an invalid JSON response",
                    interrupt=retry_interrupt,
                )
                attempt += 1
                continue

            _log_finish_reasons(
                decoded
            )

            if _has_usable_choice(decoded):
                return decoded

            diagnostic = _choice_output_diagnostic(
                decoded
            )
            print(
                f"{YELLOW}"
                f"⚠ Model API returned HTTP 200 without usable "
                f"assistant output ({diagnostic})."
                f"{RESET}"
            )

            response_error = RuntimeError(
                "Model API returned HTTP 200 without a usable choice."
            )
            _retry_after_error(
                attempt=attempt,
                max_attempts=max_attempts,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                error=response_error,
                reason=(
                    "returned an empty or malformed response "
                    f"({diagnostic})"
                ),
                interrupt=retry_interrupt,
            )
            attempt += 1
            continue

        except urllib.error.HTTPError as error:
            try:
                body = error.read().decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                body = ""

            detail = _http_error_detail(
                body
            )

            gmicloud_balance_ignored = (
                _is_gmicloud_insufficient_balance_error(
                    error.code,
                    body,
                )
            )

            if gmicloud_balance_ignored:
                print(
                    f"{YELLOW}"
                    f"⏺ Ignoring GMICloud HTTP {error.code} "
                    f"'Insufficient balance' response; "
                    f"continuing with retry policy."
                    f"{RESET}"
                )
            else:
                print(
                    f"{RED}"
                    f"✖ Model API returned HTTP "
                    f"{error.code}: "
                    f"{detail or error.reason}"
                    f"{RESET}"
                )

            if _is_stealth_continue_work_error(
                error.code,
                body,
            ):
                if stealth_continue_used:
                    raise RuntimeError(
                        f"Model API returned HTTP "
                        f"{error.code} after the one-time "
                        f"'continue your work' recovery: "
                        f"{detail or error.reason}"
                    ) from error

                if (
                    retry_interrupt is not None
                    and retry_interrupt()
                ):
                    raise ModelRequestInterrupted(
                        "Model recovery interrupted by user steering."
                    ) from error

                active_request_messages = (
                    _append_continue_work_message(
                        active_request_messages
                    )
                )
                active_request_messages = (
                    normalize_message_content(
                        active_request_messages
                    )
                )
                validate_tool_history(
                    active_request_messages
                )

                stealth_continue_used = True
                recovery_request_pending = True

                print(
                    f"{YELLOW}"
                    f"⏺ Stealth provider failure detected. "
                    f"Appending user message "
                    f"'continue your work' and issuing one "
                    f"continuation recovery request."
                    f"{RESET}"
                )

                # This is a changed request, not a retry of the failed payload,
                # so it deliberately does not consume another ordinary retry
                # slot or wait for exponential backoff.
                continue

            if _should_retry_http_error(
                error.code,
                body,
            ):
                retry_after = _get_retry_after(
                    error,
                    body,
                )

                reason = _http_retry_reason(
                    error.code,
                    body,
                )

                if detail:
                    reason += f": {detail[:300]}"

                _retry_after_error(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    initial_backoff=initial_backoff,
                    max_backoff=max_backoff,
                    error=error,
                    reason=reason,
                    retry_after=retry_after,
                    interrupt=retry_interrupt,
                )

                attempt += 1
                continue

            raise RuntimeError(
                f"Model API returned HTTP "
                f"{error.code}: "
                f"{detail or error.reason}"
            ) from error

        except (
            TimeoutError,
            socket.timeout,
        ) as error:
            _retry_after_error(
                attempt=attempt,
                max_attempts=max_attempts,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                error=error,
                reason="timed out",
                interrupt=retry_interrupt,
            )
            attempt += 1
            continue

        except urllib.error.URLError as error:
            if _is_retryable_url_error(
                error
            ):
                _retry_after_error(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    initial_backoff=initial_backoff,
                    max_backoff=max_backoff,
                    error=error,
                    reason=(
                        "hit a temporary "
                        "network error"
                    ),
                    interrupt=retry_interrupt,
                )
                attempt += 1
                continue

            raise RuntimeError(
                "Could not connect to model API: "
                f"{error.reason}"
            ) from error

        # Some socket/HTTP failures can escape urllib without being
        # wrapped in URLError. ConnectionError covers common cases such
        # as connection reset, refused, and aborted connections.
        except ConnectionError as error:
            _retry_after_error(
                attempt=attempt,
                max_attempts=max_attempts,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                error=error,
                reason=(
                    "lost the network "
                    "connection"
                ),
                interrupt=retry_interrupt,
            )
            attempt += 1
            continue

    # The loop should only terminate by returning a response or by
    # _retry_after_error() raising after the final allowed ordinary attempt.
    raise RuntimeError(
        "Model API request failed unexpectedly."
    )


def build_memory_context(
    tools: dict[str, Tool],
) -> str | None:
    """
    Build durable conversation-memory context for the next model request.

    Memory tools own their state and expose an LLM-readable
    representation through ``format_for_llm``.
    """
    sections: list[str] = []

    for tool in tools.values():
        if not isinstance(
            tool,
            MemoryTool,
        ):
            continue

        section = tool.format_for_llm().strip()

        if section:
            sections.append(
                section
            )

    if not sections:
        return None

    return "\n\n".join(
        (
            "# Conversation Memory",
            (
                "The following state is owned by this conversation and "
                "survives agent turns and dropped older messages. Treat it "
                "as active working memory. Update it "
                "through the corresponding memory tools when it becomes "
                "completed, stale, invalid, or otherwise changes."
            ),
            *sections,
        )
    )

def insert_memory_context(
    messages: list[ChatMessage],
    memory_context: str | None,
) -> list[ChatMessage]:
    """
    Insert mutable conversation memory late in the prompt to maximize
    reusable prompt-cache prefix length.

    Preference:

    1. Immediately before the most recent assistant tool-call message.
    2. Otherwise immediately before the most recent user message.
    3. Otherwise at the end.
    """
    if not memory_context:
        return list(messages)

    memory_message = cast(
        ChatMessage,
        {
            "role": "system",
            "content": memory_context,
        },
    )

    result = list(messages)

    # Never insert between an assistant tool call and its tool results.
    # Instead place memory before the entire latest tool exchange.
    for index in range(
        len(result) - 1,
        -1,
        -1,
    ):
        message = result[index]

        if (
            message.get("role") == "assistant"
            and isinstance(
                message.get("tool_calls"),
                list,
            )
            and message.get("tool_calls")
        ):
            result.insert(
                index,
                memory_message,
            )

            return result

    # No historical tool exchange. Keep mutable memory near the end,
    # but before the current user turn.
    for index in range(
        len(result) - 1,
        -1,
        -1,
    ):
        if result[index].get("role") == "user":
            result.insert(
                index,
                memory_message,
            )

            return result

    result.append(
        memory_message
    )

    return result

def normalize_message_content(
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    """
    Normalize OpenAI-compatible message content for stricter providers.

    Some providers reject assistant tool-call messages whose content is
    null, despite the OpenAI protocol permitting omitted/null content
    when tool_calls are present.
    """
    normalized: list[ChatMessage] = []

    for index, message in enumerate(messages):
        current = cast(
            dict[str, Any],
            dict(message),
        )

        content = current.get(
            "content"
        )

        role = current.get(
            "role"
        )

        if content is None:
            if (
                role == "assistant"
                and current.get("tool_calls")
            ):
                current["content"] = ""
            else:
                raise ValueError(
                    f"Message {index} with role {role!r} "
                    "has null or missing content."
                )

        elif not isinstance(
            content,
            (str, list),
        ):
            raise ValueError(
                f"Message {index} with role {role!r} "
                "has invalid content type "
                f"{type(content).__name__!r}."
            )

        normalized.append(
            cast(
                ChatMessage,
                current,
            )
        )

    return normalized


