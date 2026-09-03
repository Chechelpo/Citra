from __future__ import annotations

import json
import math
import re
from typing import Any

from ...tools.tool import Tool


# ============================================================================
# Public API
# ============================================================================


class ModelResponseNormalizationError(ValueError):
    """A model emitted recognizable protocol data that could not be parsed."""


_KIMI_K2_MODEL_ID_RE = re.compile(
    r"(?i)"
    r"(?<![a-z0-9])"
    r"kimi[\s._/-]*k2"
    r"(?:[\s._/-]*\d+)*"
)


def normalize_model_response(
    response: dict[str, Any],
    *,
    model_id: str,
    tools: dict[str, Tool],
) -> dict[str, Any]:
    """
    Dispatch response normalization by model family.

    Unknown models pass through untouched.
    """

    if _KIMI_K2_MODEL_ID_RE.search(model_id):
        return normalize_kimi_tool_calls(
            response,
            tools=tools,
        )

    # Future:
    #
    # if _LFM2_MODEL_ID_RE.search(model_id):
    #     return normalize_lfm2_tool_calls(response, tools=tools)
    #
    # if _SOME_MODEL_RE.search(model_id):
    #     return normalize_some_model(response, tools=tools)

    return response

# ---------------------------------------------------------------------------
# Kimi protocol tokens
# ---------------------------------------------------------------------------

_SECTION_BEGIN = "<|tool_calls_section_begin|>"
_SECTION_END = "<|tool_calls_section_end|>"

_CALL_BEGIN = "<|tool_call_begin|>"
_CALL_END = "<|tool_call_end|>"

_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"

_IM_END = "<|im_end|>"


_FULL_CALL_ID_RE = re.compile(
    r"functions\.([A-Za-z0-9_.-]+):(\d+)"
)

_MARKED_CALL_ID_RE = re.compile(
    r"(?:functions\.)?([A-Za-z0-9_.-]+):(\d+)"
)

_BARE_CALL_ID_RE = re.compile(
    r"(\d+)"
)

_KIMI_MODEL_RE = re.compile(
    r"(?i)(?:^|[/_.:\-\s])kimi[/_.:\-\s]*k2"
)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def normalize_model_response(
    response: dict[str, Any],
    *,
    model_id: str,
    tools: dict[str, Tool],
) -> dict[str, Any]:
    """
    Normalize model-specific response quirks.

    Unknown model families pass through unchanged.
    """

    if _KIMI_MODEL_RE.search(model_id):
        return normalize_kimi_tool_calls(
            response,
            tools=tools,
        )

    return response


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


def _tool_registry(
    tools: dict[str, Tool],
) -> tuple[
    dict[str, str],
    dict[str, dict[str, Any]],
]:
    """
    Return:

        aliases:
            exposed/dict names -> canonical function name

        schemas:
            canonical function name -> parameter schema
    """

    aliases: dict[str, str] = {}
    schemas: dict[str, dict[str, Any]] = {}

    for key, tool in tools.items():
        key_name = str(key)
        canonical_name = key_name
        schema: dict[str, Any] = {}

        spec = tool.get_as_tool()

        if isinstance(spec, dict):
            function = spec.get("function")

            if isinstance(function, dict):
                name = function.get("name")

                if isinstance(name, str) and name:
                    canonical_name = name

                parameters = function.get("parameters")

                if isinstance(parameters, dict):
                    schema = parameters

            else:
                name = spec.get("name")

                if isinstance(name, str) and name:
                    canonical_name = name

                parameters = spec.get("parameters")

                if isinstance(parameters, dict):
                    schema = parameters

        aliases[key_name] = canonical_name
        aliases[canonical_name] = canonical_name

        schemas[canonical_name] = schema

    return aliases, schemas


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _skip_ws(
    text: str,
    position: int,
) -> int:
    while (
        position < len(text)
        and text[position].isspace()
    ):
        position += 1

    return position


def _find_first_tool_start(
    text: str,
    aliases: dict[str, str],
) -> int | None:
    """
    Find the earliest credible Kimi tool-call start.

    Prefer native markers when available, but also support providers such
    as NanoGPT that stripped the special tokens and left:

        functions.read:0{...}
    """

    candidates: list[int] = []

    for marker in (
        _SECTION_BEGIN,
        _CALL_BEGIN,
    ):
        position = text.find(marker)

        if position != -1:
            candidates.append(position)

    for match in _FULL_CALL_ID_RE.finditer(text):
        name = match.group(1)

        if name not in aliases:
            continue

        position = _skip_ws(
            text,
            match.end(),
        )

        if text.startswith(
            _ARGUMENT_BEGIN,
            position,
        ):
            position = _skip_ws(
                text,
                position
                + len(_ARGUMENT_BEGIN),
            )

        # Kimi tools use object-shaped arguments.
        if (
            position < len(text)
            and text[position] == "{"
        ):
            candidates.append(
                match.start()
            )

    if not candidates:
        return None

    return min(candidates)


def _strict_json_object_end(
    text: str,
    position: int,
) -> int | None:
    """
    Use JSON decoding as a boundary detector when possible.

    Failure is NOT considered a protocol error. Kimi can emit malformed
    JSON arguments; those must still be exposed as raw tool arguments.
    """

    try:
        value, end = json.JSONDecoder().raw_decode(
            text,
            position,
        )
    except (
        json.JSONDecodeError,
        RecursionError,
    ):
        return None

    if not isinstance(value, dict):
        return None

    return end


def _next_known_call_header(
    text: str,
    start: int,
    aliases: dict[str, str],
) -> int | None:
    """
    Locate the next stripped Kimi call:

        functions.<known-tool>:N{

    Unknown function-like text inside arguments is ignored.
    """

    for match in _FULL_CALL_ID_RE.finditer(
        text,
        start,
    ):
        if match.group(1) not in aliases:
            continue

        position = _skip_ws(
            text,
            match.end(),
        )

        if text.startswith(
            _ARGUMENT_BEGIN,
            position,
        ):
            position = _skip_ws(
                text,
                position
                + len(_ARGUMENT_BEGIN),
            )

        if (
            position < len(text)
            and text[position] == "{"
        ):
            return match.start()

    return None


def _native_argument_end(
    text: str,
    arguments_start: int,
) -> int | None:
    """
    Find the structural <|tool_call_end|> for malformed native calls.

    rfind() is intentional: if literal marker-looking text appears inside
    the argument payload, the real structural end marker should be the
    last one before the next call or the section terminator.
    """

    next_call = text.find(
        _CALL_BEGIN,
        arguments_start,
    )

    section_end = text.find(
        _SECTION_END,
        arguments_start,
    )

    limits = [
        value
        for value in (
            next_call,
            section_end,
        )
        if value != -1
    ]

    limit = (
        min(limits)
        if limits
        else len(text)
    )

    position = text.rfind(
        _CALL_END,
        arguments_start,
        limit,
    )

    if position == -1:
        return None

    return position


# ---------------------------------------------------------------------------
# Bare-counter tool-name inference
# ---------------------------------------------------------------------------


def _infer_tool_name(
    raw_arguments: str,
    schemas: dict[str, dict[str, Any]],
) -> str | None:
    """
    Kimi occasionally emits:

        <|tool_call_begin|>17<|tool_call_argument_begin|>{...}

    With a named call we never need to JSON-decode arguments.

    A bare counter is different: the function name is missing, so argument
    keys are the only way to infer the tool. Fail on ambiguity.
    """

    if len(schemas) == 1:
        return next(iter(schemas))

    try:
        arguments = json.loads(
            raw_arguments
        )
    except (
        json.JSONDecodeError,
        TypeError,
    ):
        return None

    if not isinstance(arguments, dict):
        return None

    keys = set(arguments)

    candidates: list[
        tuple[
            tuple[int, int, int],
            str,
        ]
    ] = []

    for name, schema in schemas.items():
        properties = schema.get(
            "properties",
            {},
        )

        if not isinstance(properties, dict):
            continue

        property_keys = set(properties)

        overlap = len(
            keys & property_keys
        )

        extra = len(
            keys - property_keys
        )

        if overlap == 0:
            continue

        score = (
            overlap - extra,
            overlap,
            -extra,
        )

        candidates.append(
            (
                score,
                name,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        reverse=True
    )

    if (
        len(candidates) > 1
        and candidates[0][0]
        == candidates[1][0]
    ):
        return None

    return candidates[0][1]


# ---------------------------------------------------------------------------
# Kimi grammar
# ---------------------------------------------------------------------------


def _parse_kimi_suffix(
    text: str,
    start: int,
    *,
    aliases: dict[str, str],
    schemas: dict[str, dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """
    Parse Kimi calls from either:

      Native:
        <|tool_calls_section_begin|>
        <|tool_call_begin|>
        functions.read:0
        <|tool_call_argument_begin|>
        {...}
        <|tool_call_end|>
        <|tool_calls_section_end|>

      Stripped:
        functions.read:0{...}functions.tree:1{...}

    Malformed argument JSON is preserved verbatim.
    """

    position = _skip_ws(
        text,
        start,
    )

    if text.startswith(
        _SECTION_BEGIN,
        position,
    ):
        position = _skip_ws(
            text,
            position + len(_SECTION_BEGIN),
        )

    calls: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    while position < len(text):
        position = _skip_ws(
            text,
            position,
        )

        if text.startswith(
            _SECTION_END,
            position,
        ):
            position = _skip_ws(
                text,
                position + len(_SECTION_END),
            )
            break

        marked = text.startswith(
            _CALL_BEGIN,
            position,
        )

        if marked:
            position = _skip_ws(
                text,
                position + len(_CALL_BEGIN),
            )

        function_name: str | None
        emitted_id: str | None = None

        # Standard Kimi form.
        match = _FULL_CALL_ID_RE.match(
            text,
            position,
        )

        if match is not None:
            raw_name = match.group(1)
            call_index = match.group(2)

            function_name = aliases.get(
                raw_name
            )

            if function_name is None:
                return None

            emitted_id = match.group(0)
            position = match.end()

        # Slightly malformed native form, but only tolerate it when a
        # <|tool_call_begin|> marker proves this really is a call.
        elif marked:
            short_match = (
                _MARKED_CALL_ID_RE.match(
                    text,
                    position,
                )
            )

            if short_match is not None:
                raw_name = (
                    short_match.group(1)
                )

                call_index = (
                    short_match.group(2)
                )

                function_name = aliases.get(
                    raw_name
                )

                if function_name is None:
                    return None

                position = short_match.end()

            else:
                bare_match = (
                    _BARE_CALL_ID_RE.match(
                        text,
                        position,
                    )
                )

                if bare_match is None:
                    return None

                call_index = (
                    bare_match.group(1)
                )

                function_name = None
                position = bare_match.end()

        else:
            break

        position = _skip_ws(
            text,
            position,
        )

        if text.startswith(
            _ARGUMENT_BEGIN,
            position,
        ):
            position = _skip_ws(
                text,
                position + len(_ARGUMENT_BEGIN),
            )

        if (
            position >= len(text)
            or text[position] != "{"
        ):
            return None

        arguments_start = position

        # ---------------------------------------------------------------
        # 1. Best case: actual valid JSON.
        # ---------------------------------------------------------------

        json_end = _strict_json_object_end(
            text,
            arguments_start,
        )

        arguments_end: int
        next_position: int

        if json_end is not None:
            probe = _skip_ws(
                text,
                json_end,
            )

            if text.startswith(
                _CALL_END,
                probe,
            ):
                arguments_end = json_end

                next_position = (
                    probe
                    + len(_CALL_END)
                )

            elif (
                text.startswith(
                    _CALL_BEGIN,
                    probe,
                )
                or text.startswith(
                    _SECTION_END,
                    probe,
                )
                or _FULL_CALL_ID_RE.match(
                    text,
                    probe,
                )
                or probe == len(text)
            ):
                arguments_end = json_end
                next_position = probe

            else:
                # JSON parsed, but whatever follows doesn't look like Kimi
                # protocol. Avoid turning arbitrary prose into execution.
                return None

        # ---------------------------------------------------------------
        # 2. Malformed JSON, but native delimiters survived.
        #
        # Native delimiters are authoritative. Do NOT require JSON validity.
        # ---------------------------------------------------------------

        else:
            native_end = _native_argument_end(
                text,
                arguments_start,
            )

            if native_end is not None:
                arguments_end = native_end
                next_position = (
                    native_end
                    + len(_CALL_END)
                )
            else:
                next_header = _next_known_call_header(
                    text,
                    arguments_start + 1,
                    aliases,
                )

                section_end = text.find(
                    _SECTION_END,
                    arguments_start,
                )

                boundaries = [
                    value
                    for value in (
                        next_header,
                        (
                            section_end
                            if section_end != -1
                            else None
                        ),
                    )
                    if value is not None
                ]

                if not boundaries:
                    raise ModelResponseNormalizationError(
                        "Kimi emitted incomplete tool arguments: "
                        "invalid JSON with no terminating protocol boundary."
                    )

                arguments_end = min(boundaries)
                next_position = arguments_end

        raw_arguments = text[
            arguments_start:
            arguments_end
        ].strip()

        if not raw_arguments.startswith("{"):
            return None

        # Bare counter requires inference. Named calls don't.
        if function_name is None:
            function_name = _infer_tool_name(
                raw_arguments,
                schemas,
            )

            if function_name is None:
                # Can't safely execute an unidentified tool.
                return None

        call_id = (
            emitted_id
            if emitted_id is not None
            else (
                f"functions."
                f"{function_name}:"
                f"{call_index}"
            )
        )

        if call_id in seen_ids:
            return None

        seen_ids.add(
            call_id
        )

        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function_name,

                    # IMPORTANT:
                    # Preserve the model's argument string byte-for-byte
                    # (apart from surrounding whitespace).
                    #
                    # Do NOT json.loads/json.dumps here.
                    "arguments": raw_arguments,
                },
            }
        )

        position = _skip_ws(
            text,
            next_position,
        )

        if text.startswith(
            _SECTION_END,
            position,
        ):
            position = _skip_ws(
                text,
                position + len(_SECTION_END),
            )
            break

        if (
            text.startswith(
                _CALL_BEGIN,
                position,
            )
            or _FULL_CALL_ID_RE.match(
                text,
                position,
            )
        ):
            continue

        if position == len(text):
            break

        return None

    # Only harmless Kimi/assistant trailers may remain.
    position = _skip_ws(
        text,
        position,
    )

    for trailer in (
        "</think>",
        "</thinking>",
        _IM_END,
    ):
        if text.startswith(
            trailer,
            position,
        ):
            position = _skip_ws(
                text,
                position + len(trailer),
            )

    if position != len(text):
        return None

    return calls or None


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------


def normalize_kimi_tool_calls(
    response: dict[str, Any],
    *,
    tools: dict[str, Tool],
) -> dict[str, Any]:
    aliases, schemas = _tool_registry(
        tools
    )

    if not aliases:
        return response

    choices = response.get(
        "choices"
    )

    if not isinstance(choices, list):
        return response

    for choice in choices:
        if not isinstance(choice, dict):
            continue

        message = choice.get(
            "message"
        )

        if not isinstance(message, dict):
            continue

        # Provider already normalized it.
        existing = message.get(
            "tool_calls"
        )

        if (
            isinstance(existing, list)
            and existing
        ):
            continue

        # Different providers leak K2.6 native output into different fields.
        for field in (
            "content",
            "reasoning_content",
            "reasoning",
        ):
            text = message.get(field)

            if (
                not isinstance(text, str)
                or not text
            ):
                continue

            start = _find_first_tool_start(
                text,
                aliases,
            )

            if start is None:
                continue

            calls = _parse_kimi_suffix(
                text,
                start,
                aliases=aliases,
                schemas=schemas,
            )

            if not calls:
                continue

            message["tool_calls"] = calls

            # Preserve any real assistant/reasoning text before the calls.
            message[field] = (
                text[:start].rstrip()
            )

            choice["finish_reason"] = (
                "tool_calls"
            )

            break

    return response
