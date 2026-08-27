"""Protocol-safe agent loop independent of workspace and REPL lifecycle."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageFunctionToolCallParam

from citra.utils.prompt import build_system_prompt

from ..cli.rendering import (
    memory_tool_for_call,
    render_assistant_text,
    render_memory_change,
    render_tool_call_result,
    render_tool_call_start,
)
from ..context import ExecutionContext
from ..tools.default_registry import TOOL_REGISTRY
from ..tools.enable_tools import EnableTools
from ..tools.session_memory import TodoTool
from ..utils.chat_completions_api import (
    ModelRequestInterrupted,
    build_memory_context,
    call_api,
)
from ..utils.terminal import RESET, YELLOW
from .response import execute_tool_call, get_assistant_message
from .session import AgentSession


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
        ensure_active = getattr(self.context, "ensure_active", None)
        if callable(ensure_active):
            ensure_active()
        turn_number = self.session.begin_turn()
        mode = getattr(self.context, "mode", None)
        get_task_steering = getattr(mode, "get_task_steering", None)
        steering = (
            get_task_steering(
                turn_number - 1,
                self.context,
            )
            if callable(get_task_steering)
            else None
        )
        if steering is not None and not isinstance(steering, str):
            raise TypeError("Mode task steering must be a string or None")
        if steering:
            self.session.add_user_message(steering)
        prompt: str = (
            build_system_prompt(self.context)
            if hasattr(self.context, "workspace")
            else ""
        )
        core_tool_ids, deferred_catalog = _configured_tools(self.context)
        enabled_tool_ids: set[str] = set()

        while True:
            if self._runtime_is_closing():
                return
            self.session.flush_steering()

            # Keep the tool schema order monotonic for prompt-cache locality:
            # core tools and the loader stay fixed, deferred tools append only.
            tools = TOOL_REGISTRY.instantiate(
                self.context,
                self.session,
                tool_ids=core_tool_ids,
            )
            tools["enable_tools"] = EnableTools(
                context=self.context,
                available_tools=deferred_catalog,
                enabled_tool_ids=enabled_tool_ids,
            )
            tools.update(
                TOOL_REGISTRY.instantiate(
                    self.context,
                    self.session,
                    tool_ids=enabled_tool_ids,
                )
            )

            model_value = self.context.config.model
            model_config = model_value() if callable(model_value) else model_value
            model_id = str(getattr(model_config, "id", "test-model"))
            max_input_tokens = int(
                getattr(model_config, "max_input_tokens", 1_000_000)
            )

            api_arguments: dict[str, Any] = {
                "context": self.context,
                "messages": self.session.get_last_messages_up_to_tokenLength(
                    model_id=model_id,
                    length=max_input_tokens,
                ),
                "tools": tools,
            }

            reasoning_effort = model_config.reasoning_effort
            if reasoning_effort is not None:
                api_arguments["reasoning_effort"] = reasoning_effort

            if self.api_call is call_api:
                # Freeze the active model for the entire HTTP request and all
                # of its retries. A config change only affects the next cycle.
                api_arguments["model_config"] = model_config
                api_arguments["retry_interrupt"] = self.session.steering.has_pending

            if prompt:
                api_arguments["sys_prompt"] = prompt
            try:
                response = self.api_call(**api_arguments)
            except ModelRequestInterrupted:
                # No assistant message was accepted, so this is a safe place
                # to flush steering and rebuild the request immediately.
                continue

            if self._runtime_is_closing():
                return

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
                if self._runtime_is_closing():
                    return
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
                    else execute_tool_call(
                        tools,
                        tool_call,
                        session=self.session,
                    )
                )
                render_tool_call_result(result)
                self.session.add_tool_result(call_id, result)
                if not cancel_remaining and memory_tool is not None:
                    render_memory_change(tools, memory_before)

    def _runtime_is_closing(self) -> bool:
        workspace = getattr(self.context, "workspace", None)
        return bool(getattr(workspace, "is_closing", False))


def _configured_tools(
    context: ExecutionContext,
) -> tuple[set[str], dict[str, str]]:
    """Apply runtime-mode exclusions before exposing any tool schemas."""
    disabled_tool_ids = set(
        getattr(
            getattr(context, "workspace", None),
            "disabled_tool_ids",
            (),
        )
    )
    core_tool_ids = set(TOOL_REGISTRY.core_tool_ids) - disabled_tool_ids
    deferred_catalog = {
        tool_id: summary
        for tool_id, summary in TOOL_REGISTRY.deferred_catalog.items()
        if tool_id not in disabled_tool_ids
    }
    return core_tool_ids, deferred_catalog


def run_agent_turn(
    session: AgentSession,
    context: ExecutionContext,
    *,
    api_call: ApiCall = call_api,
) -> None:
    """Compatibility function using an already lifecycle-owned context."""
    AgentRunner(context, session, api_call=api_call).run_turn()
