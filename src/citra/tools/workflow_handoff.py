"""Typed controller handoff used by serial workflow phases."""

from __future__ import annotations

from typing import Any, override

from citra.context import ExecutionContext
from citra.utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)

from .tool import Tool, ToolDefinition


class WorkflowHandoffTool(Tool):
    """Submit one validated phase summary and transition request."""

    TOOL_ID = "workflow_handoff"
    INVALIDATES_TOOL_CACHE = False
    MAX_OUTPUT_TOKENS = 512

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="workflow_handoff",
            description=(
                "Complete the active serial workflow phase by submitting its "
                "explicit factual handoff and requested next phase. The "
                "workflow controller validates permitted transitions."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="summary",
                        schema=JsonSchema.string(
                            description=(
                                "Self-contained phase output for the next "
                                "isolated role; include relevant paths, results, "
                                "risks, and unresolved issues."
                            )
                        ),
                    ),
                    JsonProperty(
                        name="next_step",
                        schema=JsonSchema.string(
                            description=(
                                "Requested next workflow step. It must be one "
                                "of the active phase's allowed transitions."
                            )
                        ),
                    ),
                ),
                additional_properties=False,
            ),
        )
    )

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        """Handle definitions for context."""
        del context
        return (ToolDefinition(definition=cls.DEFINITION),)

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        """Execute the execute operation."""
        run = self.context.workflow_runtime.require_active_run()
        handoff = run.submit_handoff(
            summary=str(arguments["summary"]),
            next_step=str(arguments["next_step"]),
        )
        return (
            f"Recorded {handoff.step_id!r} handoff; controller transition "
            f"target={handoff.next_step!r}."
        )

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        """Handle format call log."""
        return f"next_step={arguments.get('next_step', '?')}"


__all__ = ["WorkflowHandoffTool"]
