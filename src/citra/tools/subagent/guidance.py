"""
The subagent's only bridge to the orchestrator: a single ``request_guidance``
tool that appends a question to the shared transcript and blocks until the
orchestrator responds.

The orchestrator's ``SubagentTool.poll`` exposes any pending guidance
requests as part of the per-subagent digest; the orchestrator's model
answers the question through the next call to ``SubagentTool.poll`` with
an ``answer`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

from ...context import ExecutionContext
from ..tool import Tool, ToolDefinition
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)


@dataclass(frozen=True)
class SubagentGuidanceBridge:
    """Per-worker binding injected into the nested execution context.

    Normal tools are constructed by :class:`ToolRegistry` with only an
    ``ExecutionContext``.  Keeping the worker id and supervisor on this small
    context service lets ``RequestGuidanceTool`` follow that constructor
    contract instead of requiring an incompatible bespoke factory.
    """

    subagent_id: str
    supervisor: Any

    def request_guidance(self, question: str) -> str:
        return self.supervisor.request_guidance(
            self.subagent_id,
            question,
        )


class RequestGuidanceTool(Tool):
    """
    Subagent-side tool for asking the orchestrator a question.

    The tool is constructed with a reference to the
    ``SubagentSupervisor`` so it can append a ``guidance-request`` entry
    and block on the orchestrator's reply.
    """

    TOOL_ID = "request_guidance"

    INVALIDATES_TOOL_CACHE = False

    # The tool is model-facing for the subagent only. The orchestrator
    # never sees it because the subagent's mode does not register the
    # ``subagent`` tool, and the orchestrator's mode does not register
    # ``request_guidance``.

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="request_guidance",
            description=(
                "Ask the orchestrator for guidance on a single, "
                "self-contained question. The call blocks until the "
                "orchestrator responds. Use this when the task or the "
                "environment is ambiguous enough that continuing without "
                "an answer would risk wasted or wrong work. Do not use it "
                "for routine progress updates."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="question",
                        schema=JsonSchema.string(
                            description=(
                                "One clear, self-contained question for "
                                "the orchestrator. Include the exact "
                                "context the orchestrator needs to answer."
                            ),
                        ),
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        super().__init__(context=context)
        bridge = getattr(context, "subagents", None)
        if not isinstance(bridge, SubagentGuidanceBridge):
            raise RuntimeError(
                "ExecutionContext is missing its subagent guidance bridge."
            )
        self.__bridge = bridge

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        del context
        return (
            ToolDefinition(
                definition=cls.DEFINITION,
            ),
        )

    @property
    def subagent_id(self) -> str:
        return self.__bridge.subagent_id

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        question = str(arguments.get("question") or "").strip()
        if not question:
            raise ValueError(
                "request_guidance requires a non-empty 'question'."
            )

        return self.__bridge.request_guidance(question)


# ---------------------------------------------------------------------------
# Transcript entry helpers (shared with the supervisor module)
# ---------------------------------------------------------------------------

def _request_entry(
    subagent_id: str,
    question: str,
) -> dict[str, Any]:
    return {
        "role": "user",
        "content": (
            f"[subagent:{subagent_id}] asks:\n\n{question}"
        ),
        "kind": "guidance-request",
        "subagent_id": subagent_id,
    }


def _response_entry(
    subagent_id: str,
    response: str,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": (
            f"[orchestrator -> subagent:{subagent_id}]\n\n{response}"
        ),
        "kind": "guidance-response",
        "subagent_id": subagent_id,
    }
