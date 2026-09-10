from __future__ import annotations

import contextlib
import time
from threading import Event, Thread
from types import SimpleNamespace
from unittest import mock

from citra.agent import AgentSession, ToolResultMessage, UserMessage
from citra.agent.interactions import UserInteractionBroker
from citra.agent.runner import AgentRunEvent, AgentRunner
from citra.cli.repl import run_turn_with_steering
from citra.tools.default_registry import ToolSet
from citra.utils.chat_completions_api import (
    ModelCall,
    ModelResponse,
    parse_model_response,
)


def test_user_interaction_broker_round_trip() -> None:
    broker = UserInteractionBroker()
    result = []
    thread = Thread(
        target=lambda: result.append(
            broker.ask("Choose", ("A", "B"), timeout=2)
        )
    )
    thread.start()
    request = None
    while request is None:
        request = broker.take()
    assert request.question == "Choose"
    assert broker.respond(request.id, "2")
    thread.join(timeout=2)
    assert result == ["2"]


def test_user_typing_extends_broker_handoff_deadline() -> None:
    broker = UserInteractionBroker()
    result: list[str | None] = []
    thread = Thread(
        target=lambda: result.append(
            broker.ask("Explain", (), timeout=0.08)
        )
    )
    thread.start()
    request = None
    while request is None:
        request = broker.take()

    for _ in range(3):
        time.sleep(0.04)
        assert broker.record_activity(request.id)

    assert broker.respond(request.id, "still here")
    thread.join(timeout=1)
    assert result == ["still here"]


def _runner_context() -> SimpleNamespace:
    model = SimpleNamespace(
        id="test-model",
        max_input_tokens=100_000,
        reasoning_effort=None,
    )
    workflow = SimpleNamespace(
        tool_set=ToolSet(core_tools=(), deferred_tools=()),
        get_task_steering=lambda *_arguments: None,
        get_system_prompt=lambda *_arguments: "",
        get_user_message_prefix=lambda *_arguments: None,
        is_serial=False,
    )
    return SimpleNamespace(
        ensure_active=lambda: None,
        workspace=SimpleNamespace(disabled_tool_ids=(), is_closing=False),
        workflow=workflow,
        config=SimpleNamespace(model=lambda: model),
    )


def test_mid_turn_steering_waits_until_next_model_call(monkeypatch) -> None:
    session = AgentSession()
    session.add_user_message("start")
    entered = Event()
    release = Event()
    seen_messages = []
    executed: list[str] = []

    def fake_api(model_call: ModelCall) -> ModelResponse:
        seen_messages.append(model_call.messages)
        if len(seen_messages) == 1:
            entered.set()
            assert release.wait(2)
            return parse_model_response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "function": {
                                            "name": "first",
                                            "arguments": "{}",
                                        },
                                    },
                                    {
                                        "id": "call-2",
                                        "function": {
                                            "name": "second",
                                            "arguments": "{}",
                                        },
                                    },
                                ],
                            }
                        }
                    ]
                }
            )
        return parse_model_response(
            {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        )

    def execute(_tools, tool_call, **_keywords) -> str:
        executed.append(tool_call.id)
        return f"completed {tool_call.id}"

    monkeypatch.setattr("citra.agent.runner.execute_tool_call", execute)
    runner = AgentRunner(
        _runner_context(),
        session,
        api_call=fake_api,
        render_output=False,
    )
    thread = Thread(target=runner.run_turn)
    thread.start()
    assert entered.wait(2)
    session.queue_steering("Do not write that file; stop.")
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(seen_messages) == 2
    assert executed == ["call-1", "call-2"]
    assert isinstance(seen_messages[1][-1], UserMessage)
    assert "Do not write" in seen_messages[1][-1].content
    assert [
        message.content
        for message in seen_messages[1]
        if isinstance(message, ToolResultMessage)
    ] == ["completed call-1", "completed call-2"]


def test_steering_received_during_final_response_continues_the_turn() -> None:
    session = AgentSession()
    session.add_user_message("start")
    entered = Event()
    release = Event()
    seen_messages = []

    def fake_api(model_call: ModelCall) -> ModelResponse:
        seen_messages.append(model_call.messages)
        if len(seen_messages) == 1:
            entered.set()
            assert release.wait(2)
            return parse_model_response(
                {
                    "choices": [
                        {"message": {"role": "assistant", "content": "stale answer"}}
                    ]
                }
            )
        return parse_model_response(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": "corrected answer"}}
                ]
            }
        )

    runner = AgentRunner(
        _runner_context(),
        session,
        api_call=fake_api,
        render_output=False,
    )
    thread = Thread(target=runner.run_turn)
    thread.start()
    assert entered.wait(2)
    session.queue_steering("Use the other implementation.")
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(seen_messages) == 2
    assert isinstance(seen_messages[1][-1], UserMessage)
    assert "other implementation" in seen_messages[1][-1].content


def test_hard_stop_interrupts_an_active_model_wait() -> None:
    session = AgentSession()
    session.add_user_message("start")
    entered = Event()
    release = Event()

    def blocked_api(_model_call: ModelCall) -> ModelResponse:
        entered.set()
        release.wait(5)
        return parse_model_response(
            {"choices": [{"message": {"role": "assistant", "content": "late"}}]}
        )

    runner = AgentRunner(
        _runner_context(),
        session,
        api_call=blocked_api,
        render_output=False,
    )
    thread = Thread(target=runner.run_turn)
    thread.start()
    assert entered.wait(1)

    started = time.monotonic()
    runner.request_stop()
    thread.join(timeout=1)
    elapsed = time.monotonic() - started
    release.set()

    assert not thread.is_alive()
    assert elapsed < 0.5
    assert session.get_messages() == [UserMessage("start")]


def test_second_interrupt_stops_agent_without_closing_repl_turn() -> None:
    release = Event()
    soft_stops: list[bool] = []
    hard_stops: list[bool] = []

    class InterruptTwice:
        calls = 0

        def prompt_until(self, *_arguments, **_keywords) -> None:
            self.calls += 1
            raise KeyboardInterrupt

    application = SimpleNamespace(
        run_agent_turn=lambda: release.wait(5),
        interactions=SimpleNamespace(take=lambda: None, has_pending=lambda: False),
        request_soft_stop=lambda: soft_stops.append(True),
        request_hard_shutdown=lambda: hard_stops.append(True),
    )
    started = time.monotonic()
    try:
        run_turn_with_steering(application, input_service=InterruptTwice())
    finally:
        release.set()

    assert time.monotonic() - started < 0.5
    assert soft_stops == [True]
    assert hard_stops == [True]


def test_closed_input_does_not_spin_while_agent_finishes() -> None:
    class ClosedInput:
        calls = 0

        def prompt_until(self, *_, **__):
            self.calls += 1
            raise EOFError

    class Application:
        interactions = UserInteractionBroker()
        session = AgentSession()

        @staticmethod
        def run_agent_turn() -> None:
            time.sleep(0.05)

    input_service = ClosedInput()
    with mock.patch(
        "citra.cli.repl.patch_stdout",
        return_value=contextlib.nullcontext(),
    ):
        run_turn_with_steering(Application(), input_service=input_service)
    assert input_service.calls == 1


def test_agent_command_is_dispatched_from_foreground_steering_prompt() -> None:
    class Input:
        calls = 0

        def prompt_until(self, predicate, *_, **__):
            self.calls += 1
            if self.calls == 1:
                return "/agent show worker-1"
            while not predicate():
                time.sleep(0.01)
            return None

    commands: list[str] = []

    class Application:
        interactions = UserInteractionBroker()
        session = AgentSession()

        @staticmethod
        def run_agent_turn() -> None:
            time.sleep(0.05)

        @staticmethod
        def handle_command(command: str) -> bool:
            commands.append(command)
            return True

    input_service = Input()
    with mock.patch(
        "citra.cli.repl.patch_stdout",
        return_value=contextlib.nullcontext(),
    ):
        run_turn_with_steering(Application(), input_service=input_service)

    assert commands == ["/agent show worker-1"]
    assert Application.session.steering.has_pending() is False


def test_runner_observer_captures_tool_activity_without_rendering() -> None:
    session = AgentSession(memory_enabled=False)
    session.add_user_message("inspect")
    calls = 0
    events: list[AgentRunEvent] = []

    def fake_api(_model_call: ModelCall) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return parse_model_response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "missing_tool",
                                            "arguments": '{"path":"x"}',
                                        },
                                    },
                                ],
                            }
                        }
                    ]
                }
            )
        return parse_model_response(
            {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        )

    model = SimpleNamespace(
        id="test-model",
        max_input_tokens=100_000,
        reasoning_effort=None,
    )
    workflow = SimpleNamespace(
        tool_set=ToolSet(core_tools=(), deferred_tools=()),
        get_task_steering=lambda *_: None,
        get_system_prompt=lambda *_: "",
        get_user_message_prefix=lambda *_: None,
        is_serial=False,
    )
    context = SimpleNamespace(
        ensure_active=lambda: None,
        workspace=SimpleNamespace(
            disabled_tool_ids=(),
            is_closing=False,
        ),
        workflow=workflow,
        config=SimpleNamespace(
            model=lambda: model,
            memory=SimpleNamespace(enabled=False),
        ),
    )

    with mock.patch("citra.agent.runner.render_assistant_text") as rendered:
        AgentRunner(
            context,
            session,
            api_call=fake_api,
            event_sink=events.append,
            render_output=False,
        ).run_turn()

    assert [event.kind for event in events] == [
        "tool-call",
        "tool-result",
        "assistant",
    ]
    assert events[0].tool == "missing_tool"
    assert "unknown tool" in events[1].content
    rendered.assert_not_called()
