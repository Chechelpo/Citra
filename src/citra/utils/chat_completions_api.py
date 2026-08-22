from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import random
import socket
import ssl
import time
from typing import Any, Callable, cast
import urllib.error
import urllib.request

from openai.types.chat import ChatCompletionSystemMessageParam

from ..agent import ChatMessage
from ..context import ExecutionContext
from ..tools.session_memory import MemoryTool
from ..tools.tool import Tool
from .api import chat_completions_url
from .prompt import build_system_prompt
from .terminal import (
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

# Despite the historical name, this is used as the maximum number of
# total request attempts, including the initial request.
DEFAULT_MAX_RETRIES: int = 12


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
    Decide whether an HTTP response should be retried.

    Retry policy
    ------------

    * Most HTTP 4xx responses are NOT retried. They normally indicate
      that our request, credentials, endpoint, payload, or permissions
      are invalid and retrying the same request will not fix them.

    * A small set of 4xx responses which commonly represent temporary
      conditions are retried.

    * HTTP 429 is retried only when RETRY_ON_RATE_LIMIT is enabled.

    * All HTTP 5xx responses are retried because they represent a
      server-side or upstream failure rather than a malformed client
      request.

    Unknown 4xx responses deliberately default to non-retryable. This
    is safer than repeatedly submitting a request the server has
    explicitly rejected.
    """
    if status == 429:
        return RETRY_ON_RATE_LIMIT

    if status in _RETRYABLE_CLIENT_HTTP_STATUS_CODES:
        return True

    if 500 <= status <= 599:
        return True

    return False


def _http_retry_reason(
    status: int,
) -> str:
    """
    Return a human-readable reason for a retryable HTTP status.
    """
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
    Extract the most useful error text from an OpenAI-compatible
    HTTP error response.

    Routers such as OpenRouter may expose a generic top-level message
    while placing the downstream provider's actual error in
    error.metadata.raw.
    """
    if not body:
        return ""

    try:
        data = json.loads(body)
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return body.strip()

    if not isinstance(data, dict):
        return body.strip()

    error_data = data.get("error")

    if not isinstance(error_data, dict):
        return body.strip()

    metadata = error_data.get("metadata")

    if isinstance(metadata, dict):
        raw = metadata.get("raw")
        provider = (
            metadata.get("provider_name")
            or metadata.get("provider")
        )

        if raw:
            raw_text = str(raw).strip()

            # Sometimes metadata.raw is itself serialized JSON.
            try:
                raw_data = json.loads(raw_text)
            except (
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                raw_data = None

            if isinstance(raw_data, dict):
                nested_error = raw_data.get("error")

                if isinstance(nested_error, dict):
                    nested_message = nested_error.get(
                        "message"
                    )

                    if nested_message:
                        raw_text = str(
                            nested_message
                        )

                elif raw_data.get("message"):
                    raw_text = str(
                        raw_data["message"]
                    )

            if provider:
                return (
                    f"[{provider}] "
                    f"{raw_text}"
                )

            return raw_text

    message = error_data.get("message")

    if message:
        return str(message)

    return body.strip()

def _should_retry_http_error(
    status: int,
    body: str,
) -> bool:
    """
    Decide whether this specific HTTP failure should be retried.
    """
    if status != 400:
        return _should_retry_http_status(status)

    fields = _http_error_fields(body)
    codes = {
        value.casefold().replace("-", "_").replace(" ", "_")
        for key, value in fields
        if key.endswith("code") or key.endswith("type")
    }

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
    detail = " ".join(value for _, value in fields).casefold()

    permanent_markers = (
        "authentication failed",
        "context length",
        "context_length_exceeded",
        "invalid api key",
        "invalid request",
        "invalid tool",
        "maximum context",
        "permission denied",
        "unprocessable",
    )
    if any(marker in detail for marker in permanent_markers):
        return False

    if codes & transient_codes:
        return True

    # A router may map an upstream transient provider failure onto
    # HTTP 400 even though the user's request itself is valid.
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
        "[stealth] error",
        "temporarily unavailable",
        "overloaded",
    )

    return any(
        marker in detail
        for marker in transient_provider_markers
    )


def _http_error_fields(body: str) -> list[tuple[str, str]]:
    """Extract classifier fields without trusting one preferred message.

    OpenRouter often places the useful provider signal in ``metadata.raw``
    while keeping a generic message at ``error.message``. Classification must
    inspect both, plus structured error/type/code fields.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return [("body", body.strip())] if body.strip() else []

    fields: list[tuple[str, str]] = []

    def visit(value: Any, path: str, depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if isinstance(child, (dict, list)):
                    visit(child, child_path, depth + 1)
                elif child is not None and str(key).casefold() in {
                    "code",
                    "error_type",
                    "message",
                    "provider_code",
                    "provider_name",
                    "raw",
                    "type",
                }:
                    fields.append((child_path.casefold(), str(child)))
        elif isinstance(value, list):
            for index, child in enumerate(value[:20]):
                visit(child, f"{path}[{index}]", depth + 1)

    visit(data, "")
    return fields


def _has_usable_choice(value: Any) -> bool:
    """Check the minimum response shape needed by the agent protocol."""
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

    if content is not None and not isinstance(content, str):
        return False

    tool_calls = message.get("tool_calls")

    return tool_calls is None or isinstance(tool_calls, list)


def call_api(
    context: ExecutionContext,
    messages: list[ChatMessage],
    tools: dict[str, Tool],
    reasoning_effort: str | None = None,
    *,
    request_timeout: float | None = None,
    max_attempts: int | None = None,
    initial_backoff: float | None = None,
    max_backoff: float | None = None,
    retry_interrupt: Callable[[], bool] | None = None,
    sys_prompt: str | None = None
) -> dict[str, Any]:
    """
    Perform one OpenAI-compatible Chat Completions request.

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
    and fail immediately. Examples include:

    * 401 Unauthorized
    * 403 Forbidden
    * 404 Not Found
    * 405 Method Not Allowed
    * 413 Content Too Large
    * 415 Unsupported Media Type
    * 422 Unprocessable Content

    Retrying these would normally submit the same invalid request
    repeatedly without any chance of recovery.

    * 400 Bad Request is retried only when structured error codes or provider
      metadata identify a transient upstream/router failure (including the
      OpenRouter ``[Stealth] ERROR`` form). Invalid requests and context-length
      failures fail immediately.

    Transport-level failures such as timeouts, temporary DNS errors,
    refused connections, dropped connections, and connection resets
    are also retried. TLS certificate verification failures are not.

    If a provider supplies Retry-After information, that value is
    honored as a minimum wait time in addition to the normal
    exponential backoff.

    ``max_attempts`` counts the initial request. For example,
    ``max_attempts=3`` permits one initial request followed by at most
    two retries.

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
            Maximum total number of attempts, including the initial
            request. Defaults to DEFAULT_MAX_RETRIES.

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
    retry_config = getattr(
        context.model_config,
        "retry",
        None,
    )
    if max_attempts is None:
        max_attempts = getattr(
            retry_config,
            "max_attempts",
            DEFAULT_MAX_RETRIES,
        )
    if request_timeout is None:
        request_timeout = getattr(
            retry_config,
            "request_timeout",
            120.0,
        )
    if initial_backoff is None:
        initial_backoff = getattr(
            retry_config,
            "initial_backoff",
            1.0,
        )
    if max_backoff is None:
        max_backoff = getattr(
            retry_config,
            "max_backoff",
            30.0,
        )
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

    model = context.model_config

    system_messages: list[
        ChatCompletionSystemMessageParam
    ] = [
        {
            "role": "system",
            "content": sys_prompt
        }
    ]

    memory_context = build_memory_context(
        tools
    )

    if memory_context:
        system_messages.append(
            {
                "role": "system",
                "content": memory_context,
            }
        )

    messages_to_merge: list[
        ChatMessage
    ] = [
        *system_messages,
        *messages,
    ]

    request_messages = merge_consecutive_roles(
        messages_to_merge
    )
    validate_tool_history(
        request_messages
    )
    payload: dict[str, Any] = {
        "model": model.id,
        "max_tokens": model.max_tokens,
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

    request_data = json.dumps(
        payload
    ).encode(
        "utf-8"
    )

    url = chat_completions_url(
        model.host
    )

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        request = urllib.request.Request(
            url,
            data=request_data,
            headers={
                "Content-Type": (
                    "application/json"
                ),
                "Authorization": (
                    f"Bearer {model.api_key}"
                ),
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=request_timeout,
            ) as response:
                raw_response = response.read().decode(
                    "utf-8",
                    errors="replace",
                )

            try:
                decoded = json.loads(raw_response)
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
                continue

            if _has_usable_choice(decoded):
                return decoded

            response_error = RuntimeError(
                "Model API returned HTTP 200 without a usable choice."
            )
            _retry_after_error(
                attempt=attempt,
                max_attempts=max_attempts,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                error=response_error,
                reason="returned an empty or malformed response",
                interrupt=retry_interrupt,
            )
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

            # Always show the HTTP error, whether or not it will
            # subsequently be retried.
            print(
                f"{RED}"
                f"✖ Model API returned HTTP "
                f"{error.code}: "
                f"{detail or error.reason}"
                f"{RESET}"
            )

            if _should_retry_http_error(
                error.code,
                body,
            ):
                retry_after = _get_retry_after(
                    error,
                    body,
                )

                reason = _http_retry_reason(
                    error.code
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
            continue

    # The loop should only terminate by returning a response or by
    # _retry_after_error() raising after the final allowed attempt.
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
