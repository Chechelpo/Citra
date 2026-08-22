"""Protocol-safe agent loop independent of workspace and REPL lifecycle."""
from __future__ import annotations

from citra.utils.prompt import build_system_prompt

from typing import Any, Callable, cast

from openai.types.chat import ChatCompletionMessageFunctionToolCallParam

from .response import execute_tool_call, get_assistant_message
from .session import AgentSession
from ..cli.rendering import (
    memory_tool_for_call,
    render_assistant_text,
    render_memory_change,
    render_tool_call_result,
    render_tool_call_start,
)
from ..context import ExecutionContext
from ..tools.default_registry import TOOL_REGISTRY
from ..tools.session_memory import TodoTool
from ..utils.chat_completions_api import (
    ModelRequestInterrupted,
    build_memory_context,
    call_api,
)
from ..utils.terminal import RESET, YELLOW


ApiCall = Callable[..., dict]
_CANCELLED_BY_STEERING = (
    "cancelled: user steering instructions were received before this tool call executed"
)


class AgentRunner:
    """Run model/tool cycles against lifecycle-scoped services."""

    def __init__(
        self,
        context: ExecutionContext,
        session: AgentSession,
        *,
        api_call: ApiCall = call_api,
    ) -> None:
        self.context = context
        self.session = session
        self.api_call = api_call

    def run_turn(self) -> None:
        self.session.begin_turn()
        prompt:str = build_system_prompt(self.context)
        while True:
            self.session.flush_steering()
            tools = TOOL_REGISTRY.instantiate(self.context, self.session)
            api_arguments: dict[str, Any] = {
                "context": self.context,
                "messages": self.session.get_last_n_messages(
                    self.context.config.message_context.uncompressed_messages
                ),
                "tools": tools,
            }

            reasoning_effort = self.context.config.model.reasoning_effort
            if reasoning_effort is not None:
                api_arguments["reasoning_effort"] = reasoning_effort

            if self.api_call is call_api:
                api_arguments["retry_interrupt"] = self.session.steering.has_pending
                
            api_arguments["sys_prompt"] = prompt
            try:
                response = self.api_call(**api_arguments)
            except ModelRequestInterrupted:
                # No assistant message was accepted, so this is a safe place
                # to flush steering and rebuild the request immediately.
                continue

            assistant = get_assistant_message(response)
            text = assistant.get("content")
            if isinstance(text, str) and text:
                render_assistant_text(text)
            tool_calls = cast(
                list[ChatCompletionMessageFunctionToolCallParam],
                assistant.get("tool_calls") or [],
            )
            
            self.session.add_assistant_message(assistant)
            if not tool_calls:
                # Steering may have arrived while a final-looking response was
                # in flight. Treat that response as an intermediate message
                # and give the model the correction before ending the turn.
                if self.session.steering.has_pending():
                    continue
                todo_tool = tools.get("todo")
                if (
                    isinstance(todo_tool, TodoTool)
                    and todo_tool.has_outstanding_todos()
                ):
                    self.session.add_user_message(
                        "Continue: valid conversation TODOs remain outstanding. "
                        "Complete them, or remove only entries that are truly stale "
                        "or invalid, before returning a final answer."
                    )
                    continue
                return
            cancel_remaining = False
            for tool_call in tool_calls:
                call_id = tool_call.get("id")
                if not call_id:
                    raise RuntimeError("Model returned a tool call without an id.")
                if not cancel_remaining and self.session.steering.has_pending():
                    cancel_remaining = True
                    print(
                        f"\n{YELLOW}⏺ Steering received. "
                        f"Cancelling remaining tool calls.{RESET}"
                    )
                render_tool_call_start(tool_call)
                memory_tool = memory_tool_for_call(tools, tool_call)
                memory_before = build_memory_context(tools) if memory_tool else None
                result = (
                    _CANCELLED_BY_STEERING
                    if cancel_remaining
                    else execute_tool_call(tools, tool_call)
                )
                render_tool_call_result(result)
                self.session.add_tool_result(call_id, result)
                if not cancel_remaining and memory_tool is not None:
                    render_memory_change(tools, memory_before)


def run_agent_turn(
    session: AgentSession,
    context: ExecutionContext,
    *,
    api_call: ApiCall = call_api,
) -> None:
    """Compatibility function using an already lifecycle-owned context."""
    AgentRunner(context, session, api_call=api_call).run_turn()
