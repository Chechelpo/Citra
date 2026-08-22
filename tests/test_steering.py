from __future__ import annotations

import contextlib
from types import SimpleNamespace
from threading import Event, Thread
import time
from unittest import mock

from citra.agent import AgentSession
from citra.agent.interactions import UserInteractionBroker
from citra.agent.runner import AgentRunner
from citra.cli.repl import run_turn_with_steering


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


def test_mid_turn_steering_cancels_unstarted_tool_calls() -> None:
    session = AgentSession()
    session.add_user_message("start")
    entered = Event()
    release = Event()
    seen_messages = []
    call_count = 0

    def fake_api(*, context, messages, tools):
        nonlocal call_count
        call_count += 1
        seen_messages.append(messages)
        if call_count == 1:
            entered.set()
            assert release.wait(2)
            return {
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
                                        "name": "write",
                                        "arguments": '{"path":"should-not-exist","content":"x"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "stopped"}}]}

    context = SimpleNamespace(
        config=SimpleNamespace(
            message_context=SimpleNamespace(uncompressed_messages=20),
            model=SimpleNamespace(reasoning_effort=None),
        )
    )
    runner = AgentRunner(context, session, api_call=fake_api)
    thread = Thread(target=runner.run_turn)
    thread.start()
    assert entered.wait(2)
    session.queue_steering("Do not write that file; stop.")
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert call_count == 2
    assert any(
        message.get("role") == "user"
        and "Do not write" in str(message.get("content"))
        for message in seen_messages[1]
    )
    assert "cancelled: user steering" in session.get_messages()[2]["content"]


def test_steering_received_during_final_response_continues_the_turn() -> None:
    session = AgentSession()
    session.add_user_message("start")
    entered = Event()
    release = Event()
    seen_messages = []

    def fake_api(*, context, messages, tools):
        seen_messages.append(messages)
        if len(seen_messages) == 1:
            entered.set()
            assert release.wait(2)
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": "stale answer"}}
                ]
            }
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "corrected answer"}}
            ]
        }

    context = SimpleNamespace(
        config=SimpleNamespace(
            message_context=SimpleNamespace(uncompressed_messages=20),
            model=SimpleNamespace(reasoning_effort=None),
        )
    )
    runner = AgentRunner(context, session, api_call=fake_api)
    thread = Thread(target=runner.run_turn)
    thread.start()
    assert entered.wait(2)
    session.queue_steering("Use the other implementation.")
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(seen_messages) == 2
    assert any(
        message.get("role") == "user"
        and "other implementation" in str(message.get("content"))
        for message in seen_messages[1]
    )


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
