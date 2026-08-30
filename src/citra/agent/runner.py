"""Protocol-safe agent loop independent of workspace and REPL lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
from ..tools.enable_tools import EnableTools
from ..tools.session_memory import RequirementTool, TodoTool
from ..tools.tool_registry import ToolRegistry
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


@dataclass(frozen=True)
class AgentRunEvent:
    """One observable model-loop event emitted at a protocol-safe boundary."""

    kind: str
    role: str
    content: str
    tool: str | None = None


AgentEventSink = Callable[[AgentRunEvent], None]


class AgentRunner:
    """Run model/tool cycles against lifecycle-scoped services."""

    def __init__(
        self,
        context: ExecutionContext,
        session: AgentSession,
        *,
        api_call: ApiCall = call_api,
        event_sink: AgentEventSink | None = None,
        render_output: bool = True,
    ) -> None:
        self.context = context
        self.session = session
        self.api_call = api_call
        self.event_sink = event_sink
        self.render_output = render_output

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

        tool_registry = ToolRegistry(toolset=self.context.mode.tool_set)
        core_tool_ids, deferred_catalog = _configured_tools(
            self.context,
            tool_registry,
        )
        enabled_tool_ids: set[str] = set()

        while True:
            if self._runtime_is_closing():
                return
            self.session.flush_steering()

            # Keep the tool schema order monotonic for prompt-cache locality:
            # core tools and the loader stay fixed, deferred tools append only.
            tools_by_id = tool_registry.instantiate(
                self.context,
                self.session,
                tool_ids=core_tool_ids,
            )
            if deferred_catalog:
                enable_tools = EnableTools(
                    context=self.context,
                    available_tools=deferred_catalog,
                    enabled_tool_ids=enabled_tool_ids,
                )
                tools_by_id[enable_tools.id] = enable_tools
            tools_by_id.update(
                tool_registry.instantiate(
                    self.context,
                    self.session,
                    tool_ids=enabled_tool_ids,
                )
            )
            # The model-facing name is selected from the current context and
            # can differ from the stable Citra ID. Build the dispatch map only
            # after every tool for this request has resolved its definition.
            tools = tool_registry.index_by_model_name(
                tools_by_id.values()
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
                self._emit(
                    AgentRunEvent(
                        kind="assistant",
                        role="assistant",
                        content=text,
                    )
                )
                if self.render_output:
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
                todo_tool = tools_by_id.get(TodoTool.TOOL_ID)
                requirement_tool = tools_by_id.get(RequirementTool.TOOL_ID)
                if (
                    isinstance(requirement_tool, RequirementTool)
                    and requirement_tool.has_unsatisfied_requirements()
                    and not self._is_serial_role_turn()
                ):
                    self.session.add_user_message(
                        "Continue: valid task requirements remain unsatisfied. "
                        "Satisfy them with verification evidence, or remove "
                        "only requirements that are truly obsolete or invalid."
                    )
                    continue
                if (
                    isinstance(todo_tool, TodoTool)
                    and todo_tool.has_outstanding_todos()
                    and not self._is_serial_role_turn()
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
                    if self.render_output:
                        print(
                            f"\n{YELLOW}⏺ Steering received. "
                            f"Cancelling remaining tool calls.{RESET}"
                        )
                function = tool_call["function"]
                tool_name = str(function.get("name") or "unknown")
                self._emit(
                    AgentRunEvent(
                        kind="tool-call",
                        role="assistant",
                        content=str(function.get("arguments") or "{}"),
                        tool=tool_name,
                    )
                )
                if self.render_output:
                    render_tool_call_start(tool_call)
                memory_tool = (
                    memory_tool_for_call(tools, tool_call)
                    if self.render_output
                    else None
                )
                memory_before = (
                    build_memory_context(tools)
                    if memory_tool
                    else None
                )
                result = (
                    _CANCELLED_BY_STEERING
                    if cancel_remaining
                    else execute_tool_call(
                        tools,
                        tool_call,
                        session=self.session,
                    )
                )
                self._emit(
                    AgentRunEvent(
                        kind="tool-result",
                        role="tool",
                        content=result,
                        tool=tool_name,
                    )
                )
                if self.render_output:
                    render_tool_call_result(result)
                self.session.add_tool_result(call_id, result)
                if not cancel_remaining and memory_tool is not None:
                    render_memory_change(tools, memory_before)

    def _emit(self, event: AgentRunEvent) -> None:
        sink = self.event_sink
        if sink is not None:
            sink(event)

    def _runtime_is_closing(self) -> bool:
        workspace = getattr(self.context, "workspace", None)
        return bool(getattr(workspace, "is_closing", False))

    def _is_serial_role_turn(self) -> bool:
        """Return whether TODOs may survive this isolated role boundary."""
        workflow = getattr(self.context, "workflow", None)
        run = getattr(self.context, "workflow_run", None)
        return bool(
            workflow is not None
            and getattr(workflow, "is_serial", False)
            and run is not None
        )


def _configured_tools(
    context: ExecutionContext,
    tool_registry: ToolRegistry,
) -> tuple[set[str], dict[str, str]]:
    """Apply runtime-mode exclusions before exposing any tool schemas."""
    disabled_tool_ids = set(
        getattr(
            getattr(context, "workspace", None),
            "disabled_tool_ids",
            (),
        )
    )

    core_tool_ids = set(tool_registry.core_tool_ids) - disabled_tool_ids
    deferred_catalog = {
        tool_id: summary
        for tool_id, summary in tool_registry.deferred_catalog(context).items()
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
