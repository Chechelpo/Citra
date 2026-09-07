"""Protocol-safe agent loop independent of workspace and REPL lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from queue import Empty, Queue
from threading import Lock, Thread
from citra.logging import Logger

from ..cli.rendering import (
    ToolCallRenderState,
    render_assistant_text,
    working_animation,
)
from ..cli.input import terminal_ui_state
from ..context import ExecutionContext
from ..tools.enable_tools import EnableTools
from ..tools.session_memory import RequirementTool, TodoTool
from ..tools.tool import Tool
from ..tools.tool_registry import ToolRegistry
from ..utils.chat_completions_api import (
    ModelCall,
    ModelRequestInterrupted,
    ModelResponse,
    call_api,
)
from ..utils.model_tokenizer import tokenize
from .chat_message import ToolCall
from .response import execute_tool_call
from .session import AgentSession

_logger = Logger("agent_runner.py")


ApiCall = Callable[[ModelCall], ModelResponse]
_CANCELLED_BY_HARD_STOP = "cancelled: hard stop requested before execution"


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
        self._stop_lock = Lock()
        self._stop_generation = 0

        _logger.debug(
            "AgentRunner initialized",
            workflow=getattr(context, "workflow", None).__class__.__name__,
        )

    def run_turn(self) -> None:
        """Execute the run turn operation."""
        self.context.ensure_active()
        stop_generation = self._current_stop_generation()

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
        tool_render_state = ToolCallRenderState()

        while True:
            if self._should_stop(stop_generation):
                _logger.info("Agent stop requested; stopping model loop")
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

            tools = tool_registry.index_by_model_name(tools_by_id.values())

            _logger.trace(
                "Resolved tools for model request",
                count=len(tools),
            )

            model_config = self.context.config.model()
            model_id = model_config.id
            max_input_tokens = model_config.max_input_tokens
            memory_services = self.session.memory.values()

            request_messages = tuple(
                self.session.get_last_messages_up_to_tokenLength(
                    model_id=model_id,
                    length=max_input_tokens,
                )
            )
            input_tokens = _token_count(model_id, request_messages)
            model_call = ModelCall(
                context=self.context,
                messages=request_messages,
                tools=tools,
                system_prompt=prompt,
                reasoning_effort=model_config.reasoning_effort,
                model_config=model_config,
                retry_interrupt=lambda: (
                    self.session.steering.has_pending()
                    or self._should_stop(stop_generation)
                ),
                memory_services=memory_services,
            )

            _logger.debug(
                "Calling model",
                model=model_id,
                tools=len(tools),
            )

            try:
                if self.render_output:
                    with working_animation():
                        response = self._call_model(model_call, stop_generation)
                else:
                    response = self._call_model(model_call, stop_generation)

            except ModelRequestInterrupted:
                if self._should_stop(stop_generation):
                    _logger.info("Model request interrupted by agent stop")
                    return
                _logger.info("Model request interrupted by steering")
                continue

            if response is None or self._should_stop(stop_generation):
                _logger.info("Discarding model response after agent stop")
                return

            assistant = response.assistant

            if self.render_output:
                terminal_ui_state.record_tokens(
                    input_tokens=response.usage.input_tokens or input_tokens,
                    output_tokens=(
                        response.usage.output_tokens
                        if response.usage.output_tokens is not None
                        else _token_count(model_id, assistant)
                    ),
                )

            text = assistant.content

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

            tool_calls = assistant.tool_calls

            _logger.debug(
                "Assistant response processed",
                tool_calls=len(tool_calls),
            )

            self.session.add_assistant_message(assistant)

            if not tool_calls:
                if self.session.steering.has_pending():
                    _logger.debug("Steering pending after final response; continuing")
                    continue

                todo_tool = tools_by_id.get(TodoTool.TOOL_ID)
                requirement_tool = tools_by_id.get(RequirementTool.TOOL_ID)

                if (
                    isinstance(requirement_tool, RequirementTool)
                    and requirement_tool.has_unsatisfied_requirements()
                    and not self._is_serial_role_turn()
                ):
                    _logger.info("Continuing due to unsatisfied requirements")

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
                    _logger.info("Continuing due to outstanding TODOs")

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

            completed_tool_calls: list[tuple[ToolCall, Tool | None, str]] = []

            for index, tool_call in enumerate(tool_calls):
                if self._should_stop(stop_generation):
                    _logger.info("Agent stop requested during tool batch")
                    for pending_call in tool_calls[index:]:
                        result = _CANCELLED_BY_HARD_STOP
                        self.session.add_tool_result(pending_call.id, result)
                        self._emit(
                            AgentRunEvent(
                                kind="tool-result",
                                role="tool",
                                content=result,
                                tool=pending_call.name,
                            )
                        )
                        if self.render_output:
                            completed_tool_calls.append(
                                (
                                    pending_call,
                                    tools.get(pending_call.name),
                                    result,
                                )
                            )
                    if self.render_output:
                        tool_render_state.render_batch(completed_tool_calls)
                    return

                call_id = tool_call.id

                tool_name = tool_call.name

                _logger.debug(
                    "Executing tool call",
                    tool=tool_name,
                )

                self._emit(
                    AgentRunEvent(
                        kind="tool-call",
                        role="assistant",
                        content=tool_call.arguments,
                        tool=tool_name,
                    )
                )

                result = execute_tool_call(
                    tools,
                    tool_call,
                    session=self.session,
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
                    completed_tool_calls.append(
                        (tool_call, tools.get(tool_name), result)
                    )

                self.session.add_tool_result(
                    call_id,
                    result,
                )

            if self.render_output:
                tool_render_state.render_batch(completed_tool_calls)

    def _emit(self, event: AgentRunEvent) -> None:
        """Handle emit."""
        sink = self.event_sink

        if sink is not None:
            sink(event)

    def request_stop(self) -> None:
        """Interrupt the active run without closing lifecycle services."""
        with self._stop_lock:
            self._stop_generation += 1

    def _current_stop_generation(self) -> int:
        """Return the generation captured by a newly starting run."""
        with self._stop_lock:
            return self._stop_generation

    def _should_stop(self, generation: int) -> bool:
        """Return whether this run was stopped or its workspace is closing."""
        with self._stop_lock:
            stopped = generation != self._stop_generation
        return stopped or self.context.workspace.is_closing

    def _call_model(
        self,
        model_call: ModelCall,
        stop_generation: int,
    ) -> ModelResponse | None:
        """Wait interruptibly for a synchronous provider call."""
        completed: Queue[ModelResponse | BaseException] = Queue(maxsize=1)

        def invoke() -> None:
            try:
                completed.put(self.api_call(model_call))
            except BaseException as error:  # noqa: BLE001 - re-raised by caller
                completed.put(error)

        Thread(target=invoke, name="citra-model-call", daemon=True).start()
        while not self._should_stop(stop_generation):
            try:
                result = completed.get(timeout=0.05)
            except Empty:
                continue
            if isinstance(result, BaseException):
                raise result
            return result
        return None

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


def _token_count(model_id: str, value: object) -> int:
    """Return a best-effort count without allowing UI accounting to fail a turn."""
    try:
        serialized = json.dumps(value, ensure_ascii=False, default=str)
        return tokenize(model_id=model_id, text=serialized)
    except (RuntimeError, TypeError, ValueError, OSError):
        return 0


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
