from __future__ import annotations

"""
Model-specific conversation-history policies mantained by LLMs.

The policy layer decides whether model reasoning metadata should survive when
Citra replays conversation history.

Source notes (verified 2026-08-27)
----------------------------------

Z.AI / GLM 5.x
~~~~~~~~~~~~~~
Sources:
- Z.AI Developer Document: "Thinking Mode"
- Z.AI Developer Document: "Chat Completion"

Relevant behavior:
- GLM supports interleaved thinking during tool use.
- When tools are used, reasoning_content should be returned with the assistant
  tool-call message so the model can continue reasoning after the tool result.
- The standard Chat Completion API defaults thinking.clear_thinking=True, so
  prior-turn reasoning is ignored by the server by default.
- Preserved Thinking uses clear_thinking=False and requires the complete,
  unmodified reasoning_content history in its original order.
- The Coding Plan endpoint enables Preserved Thinking by default.

Therefore this history layer MUST NOT destructively remove GLM reasoning.
The request's clear_thinking setting should decide whether prior reasoning is
actually consumed by GLM.


DeepSeek V4
~~~~~~~~~~~
Source:
- DeepSeek API Docs: "Thinking Mode"

Relevant behavior:
- reasoning_content from an ordinary assistant turn without tool calls may be
  omitted on later turns; if supplied, DeepSeek ignores it.
- If an assistant turn performs a tool call, its reasoning_content MUST be
  included in subsequent requests.
- DeepSeek documents that failing to replay required tool-call reasoning can
  produce an HTTP 400 response.

Therefore preserving reasoning is the safe policy. Removing reasoning based
only on current_turn is incorrect because tool-call reasoning can remain
mandatory in later user turns.


Meta Muse Spark
~~~~~~~~~~~~~~~
Sources:
- Meta Model API cookbook: "Reasoning / thinking tokens"
- OpenRouter documentation: "Reasoning Tokens"

Relevant behavior:
- On Meta's public Chat Completions endpoint, Muse Spark's chain of thought is
  internal and reasoning_content is currently empty.
- Meta's Responses API carries replayable reasoning as encrypted reasoning
  state / previous_response_id rather than plaintext Chat Completion history.
- Gateways such as OpenRouter may expose replayable reasoning metadata through
  reasoning, reasoning_content, or reasoning_details and recommend preserving
  reasoning_details unmodified across reasoning/tool continuations.

Therefore this layer should not delete Muse Spark reasoning metadata. On the
direct Meta Chat Completions API this is normally a no-op; on compatible
gateways it avoids deleting state that may be required for continuation.


MiniMax M3
~~~~~~~~~~
Sources (verified 2026-08-28):
- MiniMax API Docs: "OpenAI SDK"
- OpenRouter documentation: "Reasoning Tokens"

Relevant behavior:
- MiniMax M3 supports interleaved thinking during tool use.
- In multi-turn function-call conversations, MiniMax requires the complete
  assistant response to be appended to history to preserve reasoning-chain
  continuity.
- With MiniMax's native OpenAI-compatible API and reasoning_split=False,
  thinking is embedded in content as <think>...</think> and must be preserved
  completely.
- With reasoning_split=True, thinking is separated into reasoning_content and
  reasoning_details, and reasoning_details must be preserved completely.
- OpenRouter likewise recommends replaying reasoning_details unmodified when
  continuing reasoning/tool interactions.

Therefore this history layer MUST NOT destructively remove MiniMax M3
reasoning metadata or rewrite assistant content. The request/transport layer
should decide whether thinking is enabled and which wire representation is
accepted.


Kimi K2
~~~~~~~
The existing Citra behavior is retained:
- preserve reasoning inside the current turn;
- remove known reasoning fields from completed historical assistant turns.


Transport-layer note
--------------------
This module is deliberately about HISTORY semantics, not wire-schema
normalization. A provider adapter may still need to translate or omit fields
that its particular endpoint does not accept. In particular:
- DeepSeek direct primarily uses reasoning_content.
- OpenRouter commonly uses reasoning_details, with reasoning and
  reasoning_content supported as aliases/forms.
- Meta Responses uses its own reasoning-item representation.
- MiniMax's native OpenAI-compatible API can embed thinking in content or,
  with reasoning_split=True, expose reasoning_content/reasoning_details.

Do not normalize one representation into another here; reasoning_details may
contain structured or encrypted provider state that must remain byte-for-byte
unchanged.
"""

import re

from collections.abc import Callable, Iterable

from citra.agent.chat_message import ChatMessage


HistoryPolicy = Callable[
    [list[ChatMessage], bool],
    list[ChatMessage],
]


_REASONING_FIELDS = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
)


def _default_policy(
    messages: list[ChatMessage],
    current_turn: bool,
) -> list[ChatMessage]:
    """Handle default policy."""
    del current_turn
    return list(messages)


def _strip_assistant_fields(
    messages: list[ChatMessage],
    fields: Iterable[str],
) -> list[ChatMessage]:
    """
    Return a shallow copy of the message list with selected fields removed
    from assistant messages.

    Individual message mappings are copied before modification, so the stored
    conversation is never mutated by a history policy.
    """
    fields = tuple(fields)
    result: list[ChatMessage] = []
    for message in messages:
        current = dict(message)
        if current.get("role") == "assistant":
            for field in fields:
                current.pop(field, None)
        result.append(current)  # type: ignore[arg-type]
    return result


def _kimi_k2_policy(
    messages: list[ChatMessage],
    current_turn: bool,
) -> list[ChatMessage]:
    """
    Keep Kimi K2 reasoning while the active turn is still running, but remove
    reasoning metadata once the turn becomes ordinary conversation history.
    """
    if current_turn:
        return list(messages)
    return _strip_assistant_fields(
        messages,
        _REASONING_FIELDS,
    )


def _glm_5_policy(
    messages: list[ChatMessage],
    current_turn: bool,
) -> list[ChatMessage]:
    """
    Preserve GLM 5.x reasoning metadata.

    Z.AI already provides `thinking.clear_thinking` to control whether
    cross-turn reasoning is consumed:

        clear_thinking=True
            Standard API default. Historical reasoning is ignored/cleared by
            the provider.

        clear_thinking=False
            Preserved Thinking. Complete, unmodified historical
            reasoning_content is required.

    Interleaved tool use also requires reasoning_content to be replayed with
    the assistant tool-call message.

    Removing reasoning here would make Preserved Thinking impossible and can
    break an interleaved tool continuation, while preserving it remains safe
    when clear_thinking=True because the provider performs the clearing.
    """
    del current_turn
    return list(messages)


def _deepseek_v4_policy(
    messages: list[ChatMessage],
    current_turn: bool,
) -> list[ChatMessage]:
    """
    Preserve all DeepSeek V4 reasoning metadata.

    DeepSeek V4's direct API uses `reasoning_content`.

    For a normal non-tool turn, historical reasoning_content may be sent back
    and is ignored.

    For a tool-call turn, however, DeepSeek requires the assistant's
    reasoning_content to remain in context for subsequent requests, including
    later user turns. Dropping it can produce HTTP 400.

    We intentionally do not remove reasoning/reasoning_details either because
    OpenAI-compatible gateways may use those representations for the same
    model.
    """
    del current_turn
    return list(messages)


def _muse_spark_policy(
    messages: list[ChatMessage],
    current_turn: bool,
) -> list[ChatMessage]:
    """
    Preserve Muse Spark reasoning metadata if a transport supplied it.

    Meta's direct Chat Completions endpoint currently keeps chain-of-thought
    internal, so there is normally no non-empty reasoning_content to replay.

    Meta's Responses API uses replayable encrypted reasoning state instead.

    OpenAI-compatible gateways may expose `reasoning_details` or related
    fields, and those fields can carry continuation state. Since model_id by
    itself does not reliably identify the transport, preserving available
    metadata is safer than destructively stripping it here.
    """
    del current_turn
    return list(messages)


def _minimax_m3_policy(
    messages: list[ChatMessage],
    current_turn: bool,
) -> list[ChatMessage]:
    """
    Preserve all MiniMax M3 reasoning state and assistant content.

    MiniMax M3 uses interleaved thinking for tool workflows. Its native
    OpenAI-compatible API requires the complete assistant response to remain
    in multi-turn function-call history so the reasoning chain can continue.

    Depending on `reasoning_split`, thinking may be embedded directly in the
    assistant `content` as <think>...</think> or exposed separately through
    `reasoning_content` / `reasoning_details`. OpenAI-compatible gateways may
    also carry replayable state in `reasoning_details`.

    Because model_id does not identify the transport or output representation,
    the history layer must preserve the complete message. Transport adapters
    remain responsible for endpoint-specific schema normalization.
    """
    del current_turn
    return list(messages)


# Model IDs appear in several provider-qualified forms, for example:
#
#   z-ai/glm-5.2
#   openrouter/z-ai/glm-5.3
#   deepseek/deepseek-v4-pro
#   meta/muse-spark-1.2
#   minimax/minimax-m3
#   openrouter/minimax/minimax-m3-20260531
#   moonshot/kimi-k2
#
# Accept common provider/model separators while still requiring a model-family
# boundary so unrelated identifiers do not accidentally match.
_MODEL_SEP = r"[/_.:\-\s]"


_KIMI_K2_RE = re.compile(
    rf"(?i)"
    rf"(?:^|{_MODEL_SEP})"
    rf"kimi{_MODEL_SEP}*k2"
    rf"(?=$|{_MODEL_SEP})"
)


_GLM_5_RE = re.compile(
    rf"(?i)"
    rf"(?:^|{_MODEL_SEP})"
    rf"glm{_MODEL_SEP}*5"
    rf"(?:[._-]\d+)*"
    rf"(?=$|{_MODEL_SEP})"
)


_DEEPSEEK_V4_RE = re.compile(
    rf"(?i)"
    rf"(?:^|{_MODEL_SEP})"
    rf"deepseek{_MODEL_SEP}*v?4"
    rf"(?=$|{_MODEL_SEP})"
)


_MUSE_SPARK_RE = re.compile(
    rf"(?i)"
    rf"(?:^|{_MODEL_SEP})"
    rf"muse{_MODEL_SEP}*spark"
    rf"(?=$|{_MODEL_SEP})"
)


_MINIMAX_M3_RE = re.compile(
    rf"(?i)"
    rf"(?:^|{_MODEL_SEP})"
    rf"minimax{_MODEL_SEP}*m3"
    rf"(?=$|{_MODEL_SEP})"
)


def resolve_history_policy(
    model_id: str,
) -> HistoryPolicy:
    """
    Resolve the conversation-history policy for a model identifier.

    Matching is intentionally based on model family rather than one exact
    release so dated/provider-qualified model IDs continue to work.
    """
    if _KIMI_K2_RE.search(model_id):
        return _kimi_k2_policy
    if _GLM_5_RE.search(model_id):
        return _glm_5_policy
    if _DEEPSEEK_V4_RE.search(model_id):
        return _deepseek_v4_policy
    if _MUSE_SPARK_RE.search(model_id):
        return _muse_spark_policy
    if _MINIMAX_M3_RE.search(model_id):
        return _minimax_m3_policy
    return _default_policy


def apply_history_policy(
    messages: list[ChatMessage],
    *,
    model_id: str,
    current_turn: bool,
) -> list[ChatMessage]:
    """
    Apply the model-specific history policy without mutating the caller's
    message-list object.
    """
    return resolve_history_policy(model_id)(
        messages,
        current_turn,
    )