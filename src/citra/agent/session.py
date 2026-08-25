"""
Conversation/session state for one running Citra session.

Conversation history is stored as protocol-safe ``MessageGroup`` objects.

Each group contains one logical OpenAI message unit:
- a normal user or assistant message; or
- an assistant tool-call message together with all corresponding
  ``role="tool"`` results.

Only the agent worker should mutate conversation history.

Other threads, particularly the terminal/UI thread, may submit steering
instructions through ``steering``.
"""
from __future__ import annotations

import json
from citra.utils.model_tokenizer import tokenize

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from .steering import SteeringInbox
from .conversation_memory import ConversationMemory


__all__ = [
    "AgentSession",
    "ChatMessage",
    "MessageGroup",
]


ChatMessage = ChatCompletionMessageParam


@dataclass(frozen=True)
class CachedToolResult:
    generation: int
    result: str


class ToolCallCache:
    """Turn-local cache for deterministic model-facing tool results."""

    def __init__(self) -> None:
        self.generation = 0
        self._entries: dict[str, CachedToolResult] = {}

    def begin_turn(self) -> None:
        self.generation = 0
        self._entries.clear()

    def invalidate(self) -> None:
        self.generation += 1

    def get(
        self,
        tool_id: str,
        arguments: dict[str, object],
    ) -> CachedToolResult | None:
        entry = self._entries.get(
            self._key(tool_id, arguments)
        )
        if entry is None or entry.generation != self.generation:
            return None
        return entry

    def put(
        self,
        tool_id: str,
        arguments: dict[str, object],
        result: str,
    ) -> None:
        self._entries[self._key(tool_id, arguments)] = CachedToolResult(
            generation=self.generation,
            result=result,
        )

    @staticmethod
    def _key(
        tool_id: str,
        arguments: dict[str, object],
    ) -> str:
        encoded = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"{tool_id}:{encoded}"

@dataclass
class MessageGroup:
    """
    Protocol-safe group of OpenAI-compatible messages.

    A group is never split when selecting recent conversation context.
    """

    messages: list[ChatMessage] = field(
        default_factory=list
    )

    def to_messages(self) -> list[ChatMessage]:
        """
        Return the OpenAI-compatible messages contained in this group.
        """
        return list(self.messages)



@dataclass
class AgentSession:
    """
    Persistent state for one Citra conversation.

    ``message_groups`` stores conversation history as protocol-safe
    groups.

    Normal user and assistant messages occupy one group each.

    Assistant messages containing tool calls remain grouped with all
    corresponding ``role="tool"`` result messages.

    ``steering`` contains user corrections submitted while an agent turn
    is already running. Steering remains separate from conversation
    history until the agent reaches a protocol-safe insertion boundary.
    """

    message_groups: list[MessageGroup] = field(
        default_factory=list
    )

    steering: SteeringInbox = field(
        default_factory=SteeringInbox
    )

    memory: ConversationMemory = field(
        default_factory=ConversationMemory
    )

    turn_number: int = 0


    tool_cache: ToolCallCache = field(
        default_factory=ToolCallCache
    )

    _turn_start_group_index: int = field(
        default=0,
        init=False,
        repr=False,
    )

    _turn_cache_context_complete: bool = field(
        default=True,
        init=False,
        repr=False,
    )

    def begin_turn(self) -> int:
        """Advance and return the durable conversation turn number."""
        self.turn_number += 1
        self.tool_cache.begin_turn()
        self._turn_start_group_index = len(self.message_groups)
        self._turn_cache_context_complete = True
        return self.turn_number

    @property
    def can_reuse_tool_cache(self) -> bool:
        """Whether earlier tool results from this turn remain in model context."""
        return self._turn_cache_context_complete

    def get_messages(self) -> list[ChatMessage]:
        """
        Return the complete conversation history in OpenAI-compatible
        flat message format.
        """
        return [
            message
            for group in self.message_groups
            for message in group.messages
        ]

    def get_last_messages_up_to_tokenLength(
        self,
        model_id: str,
        length: int,
    ) -> list[ChatMessage]:
        """
        Return the most recent messages fitting within ``length`` tokens.

        Protocol-safe message groups are never split.
        """
        if length < 0:
            raise ValueError(
                "'length' must be zero or greater."
            )

        if length == 0:
            self._turn_cache_context_complete = (
                len(self.message_groups) <= self._turn_start_group_index
            )
            return []

        selected: list[MessageGroup] = []
        used_tokens = 0

        for group in reversed(self.message_groups):
            group_text = json.dumps(
                group.messages,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            tokens = tokenize(
                model_id,
                group_text,
            )

            if used_tokens + tokens > length:
                break

            selected.append(group)
            used_tokens += tokens

        selected.reverse()

        earliest_selected_index = (
            len(self.message_groups) - len(selected)
        )
        self._turn_cache_context_complete = (
            earliest_selected_index <= self._turn_start_group_index
        )

        return [
            message
            for group in selected
            for message in group.messages
        ]

    def add_user_message(
        self,
        content: str,
    ) -> None:
        """
        Append a normal user message as a new message group.

        Only use this when no assistant tool-call sequence is awaiting
        tool-result messages.
        """
        content = content.strip()

        if not content:
            return

        message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": content,
        }

        self.message_groups.append(
            MessageGroup(
                messages=[message],
            )
        )

    def add_assistant_message(
        self,
        message: ChatCompletionAssistantMessageParam,
    ) -> None:
        """
        Append an assistant message as a new message group.

        If the message contains tool calls, subsequent tool results are
        appended to this same group by ``add_tool_result``.
        """
        self.message_groups.append(
            MessageGroup(
                messages=[message],
            )
        )

    def add_tool_result(
        self,
        tool_call_id: str,
        result: str,
    ) -> None:
        """
        Append a tool result to the active assistant tool-call group.
        """
        if not self.message_groups:
            raise ValueError(
                "Cannot add a tool result without a preceding "
                "assistant tool-call message."
            )

        group = self.message_groups[-1]

        if not group.messages:
            raise ValueError(
                "Cannot add a tool result to an empty message group."
            )

        message = group.messages[0]

        if message["role"] != "assistant":
            raise ValueError(
                "Tool result must follow an assistant message."
            )

        assistant_message = cast(
            ChatCompletionAssistantMessageParam,
            message,
        )

        tool_calls = assistant_message.get("tool_calls")

        if not tool_calls:
            raise ValueError(
                "Tool result must follow an assistant message "
                "containing tool calls."
            )

        expected_ids = {
            tool_call["id"]
            for tool_call in tool_calls
        }

        if tool_call_id not in expected_ids:
            raise ValueError(
                f"Unknown tool call ID: {tool_call_id}"
            )

        existing_ids = {
            message["tool_call_id"]
            for message in group.messages[1:]
            if message["role"] == "tool"
        }

        if tool_call_id in existing_ids:
            raise ValueError(
                f"Tool result for '{tool_call_id}' "
                "has already been added."
            )

        tool_message: ChatCompletionToolMessageParam = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }

        group.messages.append(tool_message)

    def has_pending_tool_results(self) -> bool:
        """
        Return whether the latest assistant tool-call group is still
        missing one or more tool results.
        """
        if not self.message_groups:
            return False

        group = self.message_groups[-1]

        if not group.messages:
            return False

        message = group.messages[0]

        if message["role"] != "assistant":
            return False

        assistant_message = cast(
            ChatCompletionAssistantMessageParam,
            message,
        )

        tool_calls = assistant_message.get("tool_calls")

        if not tool_calls:
            return False

        expected_ids = {
            tool_call["id"]
            for tool_call in tool_calls
        }

        received_ids = {
            message["tool_call_id"]
            for message in group.messages[1:]
            if message["role"] == "tool"
        }

        return expected_ids != received_ids

    def queue_steering(
        self,
        content: str,
    ) -> bool:
        """
        Queue a user correction for the earliest safe model-call boundary.
        """
        return self.steering.push(
            content
        )

    def flush_steering(self) -> int:
        """
        Move all pending steering messages into conversation history.

        Steering may only be flushed at a protocol-safe boundary.

        If the most recent assistant message contains tool calls, every
        corresponding tool result must already have been added.
        """
        if self.has_pending_tool_results():
            raise RuntimeError(
                "Cannot flush steering while tool results are pending."
            )

        pending = self.steering.drain()

        for content in pending:
            self.add_user_message(
                content
            )

        return len(pending)

    def clear_history(
        self,
        *,
        clear_memory: bool = True,
    ) -> None:
        """
        Clear conversation history and pending steering instructions.
        """
        self.message_groups.clear()
        self.steering.clear()
        self.tool_cache.begin_turn()
        self._turn_start_group_index = 0
        self._turn_cache_context_complete = True

        if clear_memory:
            self.memory.clear()
