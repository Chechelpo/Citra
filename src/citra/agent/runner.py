"""Protocol-safe agent loop independent of workspace and REPL lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageFunctionToolCallParam

from citra.logging import Logger

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


_logger = Logger("agent_runner.py")


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
        """Initialize the instance."""
        self.context = context
        self.session = session
        self.api_call = api_call
        self.event_sink = event_sink
        self.render_output = render_output

        _logger.debug(
            "AgentRunner initialized",
            workflow=context.workflow.__class__.__name__,
        )

    def run_turn(self) -> None:
        """Execute the run turn operation."""
        self.context.ensure_active()

        turn_number = self.session.begin_turn()

        _logger.info(
            "Starting agent turn",
            turn=turn_number,
        )

        workflow = self.context.workflow
        steering = workflow.get_task_steering(
            turn_number - 1,
            self.context,
        )

        if steering is not None and not isinstance(steering, str):
            _logger.error("Workflow returned invalid steering instructions")
            raise TypeError("Workflow task steering must be a string or None")

        if steering:
            _logger.debug(
                "Applying workflow steering",
                length=len(steering),
            )
            self.session.add_user_message(steering)

        prompt: str = workflow.get_system_prompt(self.context)

        tool_registry = ToolRegistry(toolset=workflow.tool_set)

        core_tool_ids, deferred_catalog = _configured_tools(
            self.context,
            tool_registry,
        )

        _logger.debug(
            "Configured tools",
            core=len(core_tool_ids),
            deferred=len(deferred_catalog),
        )

        enabled_tool_ids: set[str] = set()

        while True:
            if self._runtime_is_closing():
                _logger.warning("Runtime closing; stopping agent loop")
                return

            self.session.flush_steering()

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

            tools = tool_registry.index_by_model_name(
                tools_by_id.values()
            )

            _logger.trace(
                "Resolved tools for model request",
                count=len(tools),
            )

            model_config = self.context.config.model()
            model_id = model_config.id
            max_input_tokens = model_config.max_input_tokens
            memory_services = self.session.memory.values()
            request_prompt = prompt
            if self.api_call is not call_api:
                memory_context = build_memory_context(memory_services)
                if memory_context:
                    request_prompt = "\n\n".join(
                        section
                        for section in (request_prompt, memory_context)
                        if section
                    )
                    _logger.debug(
                        "Projected retained memory into custom API prompt",
                        services=len(memory_services),
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
                api_arguments["model_config"] = model_config
                api_arguments["retry_interrupt"] = (
                    self.session.steering.has_pending
                )
                api_arguments["memory_services"] = memory_services

            if request_prompt:
                api_arguments["sys_prompt"] = request_prompt

            _logger.debug(
                "Calling model",
                model=model_id,
                tools=len(tools),
            )

            try:
                response = self.api_call(**api_arguments)

            except ModelRequestInterrupted:
                _logger.info(
                    "Model request interrupted by steering"
                )
                continue

            if self._runtime_is_closing():
                _logger.warning(
                    "Runtime closed after model response"
                )
                return

            assistant = get_assistant_message(response)

            text = assistant.get("content")

            if isinstance(text, str) and text:
                _logger.trace(
                    "Assistant returned text",
                    length=len(text),
                )

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

            _logger.debug(
                "Assistant response processed",
                tool_calls=len(tool_calls),
            )

            self.session.add_assistant_message(assistant)

            if not tool_calls:
                if self.session.steering.has_pending():
                    _logger.debug(
                        "Steering pending after final response; continuing"
                    )
                    continue

                todo_tool = tools_by_id.get(TodoTool.TOOL_ID)
                requirement_tool = tools_by_id.get(RequirementTool.TOOL_ID)

                if (
                    isinstance(requirement_tool, RequirementTool)
                    and requirement_tool.has_unsatisfied_requirements()
                    and not self._is_serial_role_turn()
                ):
                    _logger.info(
                        "Continuing due to unsatisfied requirements"
                    )

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
                    _logger.info(
                        "Continuing due to outstanding TODOs"
                    )

                    self.session.add_user_message(
                        "Continue: valid conversation TODOs remain outstanding. "
                        "Complete them, or remove only entries that are truly stale "
                        "or invalid, before returning a final answer."
                    )
                    continue

                _logger.info(
                    "Agent turn completed",
                    turn=turn_number,
                )

                return

            cancel_remaining = False

            for tool_call in tool_calls:
                if self._runtime_is_closing():
                    _logger.warning(
                        "Runtime closing during tool execution"
                    )
                    return

                call_id = tool_call.get("id")

                if not call_id:
                    _logger.error(
                        "Tool call missing id"
                    )
                    raise RuntimeError(
                        "Model returned a tool call without an id."
                    )

                if not cancel_remaining and self.session.steering.has_pending():
                    cancel_remaining = True

                    _logger.info(
                        "Cancelling remaining tools due to steering"
                    )

                    if self.render_output:
                        print(
                            f"\n{YELLOW}⏺ Steering received. "
                            f"Cancelling remaining tool calls.{RESET}"
                        )

                function = tool_call["function"]
                tool_name = str(function.get("name") or "unknown")

                _logger.debug(
                    "Executing tool call",
                    tool=tool_name,
                )

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

                _logger.debug(
                    "Tool call completed",
                    tool=tool_name,
                    result_length=len(result),
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

                self.session.add_tool_result(
                    call_id,
                    result,
                )

                if not cancel_remaining and memory_tool is not None:
                    render_memory_change(
                        tools,
                        memory_before,
                    )

    def _emit(self, event: AgentRunEvent) -> None:
        """Handle emit."""
        sink = self.event_sink

        if sink is not None:
            sink(event)

    def _runtime_is_closing(self) -> bool:
        """Handle runtime is closing."""
        return self.context.workspace.is_closing

    def _is_serial_role_turn(self) -> bool:
        """Return whether TODOs may survive this isolated role boundary."""
        runtime = self.context.workflow_runtime
        return runtime.workflow.is_serial and runtime.active_run is not None


def _configured_tools(
    context: ExecutionContext,
    tool_registry: ToolRegistry,
) -> tuple[set[str], dict[str, str]]:
    """Apply runtime workflow exclusions before exposing tool schemas."""

    disabled_tool_ids = set(context.workspace.disabled_tool_ids)

    core_tool_ids = set(tool_registry.core_tool_ids) - disabled_tool_ids

    deferred_catalog = {
        tool_id: summary
        for tool_id, summary in tool_registry.deferred_catalog(context).items()
        if tool_id not in disabled_tool_ids
    }

    _logger.trace(
        "Filtered configured tools",
        disabled=len(disabled_tool_ids),
        active=len(core_tool_ids),
    )

    return core_tool_ids, deferred_catalog


def run_agent_turn(
    session: AgentSession,
    context: ExecutionContext,
    *,
    api_call: ApiCall = call_api,
) -> None:
    """Compatibility function using an already lifecycle-owned context."""

    _logger.debug("Running compatibility agent turn")

    AgentRunner(
        context,
        session,
        api_call=api_call,
    ).run_turn()
