"""Compact cross-turn handoff state for unfinished work."""

from dataclasses import dataclass
from typing import Any, override

from citra.agent import AgentSession
from citra.context import ExecutionContext
from citra.utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)

from .memory_tool import MemoryTool


@dataclass(frozen=True)
class CheckpointExtract:
    content: str
    next_step: str | None
    turn: int


class CheckpointTool(MemoryTool[CheckpointExtract]):
    """Keep one authoritative resume point across agent turns."""

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="checkpoint",
            description=(
                "Set or clear the conversation's compact handoff checkpoint. "
                "Set it before ending or when work may be interrupted; record "
                "what is already true and the exact next step. It survives "
                "agent-turn boundaries and chat-history truncation."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Checkpoint operation.",
                            enum=("set", "clear"),
                        ),
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description="Concise current state for action=set.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="next_step",
                        schema=JsonSchema.string(
                            description="Concrete next action for action=set.",
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    def __init__(
        self,
        context: ExecutionContext,
        session: AgentSession,
    ) -> None:
        super().__init__(context=context, session=session, definition=self.DEFINITION)
        self._checkpoint: CheckpointExtract | None = None

    @property
    @override
    def heading(self) -> str:
        return "Handoff checkpoint"

    @override
    def get_extracts(self) -> list[CheckpointExtract]:
        return [] if self._checkpoint is None else [self._checkpoint]

    @override
    def format_extract(self, extract: CheckpointExtract) -> str:
        text = f"- State (turn {extract.turn}): {extract.content}"
        if extract.next_step:
            text += f"\n  - Next: {extract.next_step}"
        return text

    @override
    def should_offer_documentation(self) -> bool:
        return False

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        action = arguments["action"]
        if action == "clear":
            if "content" in arguments or "next_step" in arguments:
                raise ValueError("'content' and 'next_step' are invalid for clear.")
            self._checkpoint = None
            return "Cleared handoff checkpoint."
        if action != "set":
            raise ValueError(f"Unsupported checkpoint action: {action}")
        content = str(arguments.get("content") or "").strip()
        if not content:
            raise ValueError("'content' is required for checkpoint action 'set'.")
        next_step_raw = arguments.get("next_step")
        next_step = str(next_step_raw).strip() if next_step_raw is not None else None
        self._checkpoint = CheckpointExtract(
            content=content,
            next_step=next_step or None,
            turn=self.session.turn_number,
        )
        return "Updated handoff checkpoint."

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        action = arguments.get("action", "?")
        parts = [f"action={action}"]

        content = arguments.get("content")
        if content is not None:
            parts.append(f"content={self._truncate(str(content))}")

        next_step = arguments.get("next_step")
        if next_step is not None:
            parts.append(f"next_step={self._truncate(str(next_step))}")

        return " | ".join(parts)

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= 80:
            return value
        return value[:80] + "..."

