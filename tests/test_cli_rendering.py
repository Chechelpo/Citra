"""Deterministic visual tests for the centralized Rich CLI."""

import json
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from citra.agent.chat_message import ToolCall, UserMessage
from citra.agent.runner import AgentRunner
from citra.agent.session import AgentSession
from citra.cli import rendering
from citra.cli.rendering import SessionHeader, format_elapsed
from citra.commands.model import ModelCommand
from citra.cli.theme import CITRA_THEME
from citra.sandbox import SandboxMode
from citra.tools.default_registry import ToolSet
from citra.tools.tool import InvalidToolArguments
from citra.utils.chat_completions_api import (
    ModelCall,
    ModelResponse,
    parse_model_response,
)


def test_format_elapsed_uses_seconds_for_short_work() -> None:
    assert format_elapsed(12.4) == "12s"


def test_format_elapsed_uses_minutes_for_long_work() -> None:
    assert format_elapsed(125.2) == "2m 5s"


def test_format_elapsed_never_reports_negative_time() -> None:
    assert format_elapsed(-1) == "0s"


def _recording_console(width: int = 88) -> Console:
    return Console(
        record=True,
        width=width,
        color_system=None,
        force_terminal=False,
        theme=CITRA_THEME,
    )


def test_header_exposes_session_context(monkeypatch) -> None:
    output = _recording_console()
    monkeypatch.setattr(rendering, "console", output)
    rendering.print_header(
        SessionHeader(
            workflow="default",
            connection="https://api.example.test : gpt-professional",
            sandbox_mode=SandboxMode.FULL_SANDBOX,
            workspace=Path("/work/project"),
        ),
    )
    rendered = output.export_text()

    assert "CITRA" in rendered
    assert "gpt-professional" in rendered
    assert "agentic coding workspace" in rendered
    assert "workflow:" in rendered
    assert "default" in rendered
    assert "https://api.example.test" in rendered
    assert "FULL_SANDBOX" in rendered
    assert "─" * 40 in rendered


def test_structured_panels_are_capped_on_ultrawide_terminals(monkeypatch) -> None:
    output = _recording_console(width=240)
    monkeypatch.setattr(rendering, "console", output)
    rendering.print_header(
        SessionHeader(
            workflow="default",
            connection="https://api.example.test : gpt-professional",
            sandbox_mode=SandboxMode.FULL_SANDBOX,
            workspace=Path("/work/project"),
        ),
    )

    assert max(map(len, output.export_text().splitlines())) == 120


def test_edit_tool_call_displays_an_actual_diff(monkeypatch) -> None:
    output = _recording_console()
    monkeypatch.setattr(rendering, "console", output)
    tool = SimpleNamespace(
        id="edit",
        parse_arguments=json.loads,
        format_call_log=lambda _arguments: "--- a/app.py\n+++ b/app.py\n@@\n-old\n+new",
    )
    call = ToolCall("call-1", "edit", json.dumps({"path": "app.py"}))

    rendering.render_tool_call_start(call, tool)
    rendered = output.export_text()

    assert "• Edit" in rendered
    assert "-old" in rendered
    assert "+new" in rendered


def test_read_tool_call_is_a_compact_file_list(monkeypatch) -> None:
    output = _recording_console()
    monkeypatch.setattr(rendering, "console", output)
    tool = SimpleNamespace(
        id="read",
        parse_arguments=json.loads,
        format_call_log=lambda _arguments: "src/app.py\ntests/test_app.py",
    )
    call = ToolCall("call-2", "read", "{}")

    rendering.render_tool_call_start(call, tool)
    rendered = output.export_text()

    assert "• Read" in rendered
    assert "src/app.py" in rendered
    assert "tests/test_app.py" in rendered
    assert "-old" not in rendered


def test_question_uses_a_clear_choice_panel(monkeypatch) -> None:
    output = _recording_console(width=72)
    monkeypatch.setattr(rendering, "console", output)

    rendering.render_question("Which release channel?", ["Stable", "Preview"])
    rendered = output.export_text()

    assert "Which release channel?" in rendered
    assert "1. Stable" in rendered
    assert "2. Preview" in rendered
    assert rendered.count("╭") == 1


def test_multiline_shell_call_is_not_mislabeled_as_files(monkeypatch) -> None:
    output = _recording_console()
    monkeypatch.setattr(rendering, "console", output)
    tool = SimpleNamespace(
        id="bash",
        parse_arguments=json.loads,
        format_call_log=lambda _arguments: "$ printf 'one\\ntwo'\\ncwd=.",
    )
    call = ToolCall("call-shell", "bash", json.dumps({"cmd": "printf"}))

    rendering.render_tool_call_start(call, tool)
    rendered = output.export_text()

    assert "Bash" in rendered
    assert "printf" in rendered
    assert "paths" not in rendered
    assert "files" not in rendered


def test_broken_tool_formatters_do_not_break_rendering(monkeypatch) -> None:
    output = _recording_console()
    monkeypatch.setattr(rendering, "console", output)

    def fail(_value):
        raise RuntimeError("formatter bug")

    tool = SimpleNamespace(
        id="custom",
        parse_arguments=json.loads,
        format_call_log=fail,
    )
    call = ToolCall(
        "call-custom",
        "custom",
        json.dumps({"path": "src/app.py", "line": 4}),
    )

    rendering.render_tool_call_start(call, tool)
    rendering.render_tool_call_result("first line\nsecond line", tool)
    rendered = output.export_text()

    assert '"path": "src/app.py"' in rendered
    assert "first line  … +1 lines" in rendered


def test_invalid_json_is_identified_instead_of_shown_as_arguments(monkeypatch) -> None:
    output = _recording_console()
    monkeypatch.setattr(rendering, "console", output)
    def reject(_raw: str) -> dict[str, object]:
        raise InvalidToolArguments("Invalid JSON arguments for tool 'read'")

    tool = SimpleNamespace(id="read", parse_arguments=reject)
    call = ToolCall("call-bad", "read", '{"path":')

    assert rendering.render_tool_call_start(call, tool) is None
    assert "Invalid JSON arguments" in output.export_text()


def test_multiline_result_keeps_continuation_lines_indented(monkeypatch) -> None:
    output = _recording_console()
    monkeypatch.setattr(rendering, "console", output)
    tool = SimpleNamespace(id="edit")

    rendering.render_tool_call_result(
        "ok\nLSP diagnostics after edit:\nwarning: slow",
        tool,
        arguments={"path": "src/app.py"},
    )

    rendered = output.export_text()
    assert "  └ ok (+0, -0)" in rendered
    assert "    LSP diagnostics after edit:" in rendered
    assert "    warning: slow" in rendered


def test_successful_edits_render_as_one_numbered_diff_tree(monkeypatch) -> None:
    output = _recording_console(width=100)
    monkeypatch.setattr(rendering, "console", output)
    state = rendering.ToolCallRenderState()
    previews = {
        "src/a.py": (
            "--- a/src/a.py\n+++ b/src/a.py\n"
            "@@ -3,3 +3,3 @@\n context\n-old\n+new\n tail"
        ),
        "src/b.py": (
            "--- a/src/b.py\n+++ b/src/b.py\n"
            "@@ -8,2 +8,3 @@\n value\n+extra\n end"
        ),
    }

    def make_tool(path: str):
        return SimpleNamespace(
            id="edit",
            parse_arguments=json.loads,
            format_call_log=lambda _arguments: previews[path],
        )

    calls = []
    for index, path in enumerate(previews, 1):
        tool = make_tool(path)
        call = ToolCall(f"edit-{index}", "edit", json.dumps({"path": path}))
        state.prepare_call(call, tool)
        calls.append((call, tool, "ok"))

    state.render_batch(calls)

    rendered = output.export_text()
    assert "• Edited 2 files (+2 -1)" in rendered
    assert "└ src/a.py (+1 -1)" in rendered
    assert "   4 -old" in rendered
    assert "   4 +new" in rendered
    assert "└ src/b.py (+1 -0)" in rendered
    assert "ok (+1, -1)" in rendered


def test_write_result_uses_call_path_without_legacy_log_formatter(
    monkeypatch,
) -> None:
    output = _recording_console()
    monkeypatch.setattr(rendering, "console", output)
    tool = SimpleNamespace(id="write")

    rendering.render_tool_call_result(
        "ok",
        tool,
        arguments={"file_path": "src/new.py"},
    )

    assert "Wrote src/new.py" in output.export_text()


def test_command_result_preserves_errors(monkeypatch) -> None:
    output = _recording_console()
    monkeypatch.setattr(rendering, "console", output)

    rendering.render_tool_call_result(
        "permission-denied: network access was not granted",
        SimpleNamespace(id="bash"),
    )

    assert "permission-denied" in output.export_text()


def test_model_request_diagnostics_use_semantic_styles(monkeypatch) -> None:
    output = _recording_console()
    monkeypatch.setattr(rendering, "console", output)

    rendering.render_model_debug("Starting model request")
    rendering.render_model_retry(
        "was rate limited",
        delay=1.25,
        attempt=2,
        max_attempts=4,
    )
    rendering.render_model_warning("Provider returned an empty response.")
    rendering.render_model_error("Model API returned HTTP 401: unauthorized")

    rendered = output.export_text()
    assert "· Starting model request" in rendered
    assert "Retrying in 1.2s (attempt 2/4)" in rendered
    assert "! Provider returned an empty response." in rendered
    assert "× Model API returned HTTP 401" in rendered


def test_command_usage_renders_forms_and_arguments_as_a_tree(monkeypatch) -> None:
    output = _recording_console(width=110)
    monkeypatch.setattr(rendering, "console", output)

    rendering.render_command_usage((ModelCommand.usage,))

    rendered = output.export_text()
    assert "Usage" in rendered
    assert "/model — Inspect and modify named model profiles." in rendered
    assert "├── /model show [profile]" in rendered
    assert "│   └── [profile] — Profile name." in rendered
    assert "/model set [--profile <profile>] <setting> <value>" in rendered


def test_adjacent_tool_calls_share_semantic_group_until_category_changes(
    monkeypatch,
) -> None:
    output = _recording_console()
    monkeypatch.setattr(rendering, "console", output)
    state = rendering.ToolCallRenderState()
    tools = {
        name: SimpleNamespace(
            id=name,
            parse_arguments=json.loads,
            format_call_log=lambda arguments: str(arguments.get("path", "run")),
        )
        for name in ("grep", "find", "bash", "glob")
    }

    calls = [
        (
            ToolCall(
                f"call-{index}",
                name,
                json.dumps({"path": f"src/{name}.py"}),
            ),
            tools[name],
            "ok",
        )
        for index, name in enumerate(("grep", "find", "bash", "glob"), 1)
    ]
    state.render_batch(calls)

    rendered = output.export_text()
    assert rendered.count("• Explored") == 2
    assert rendered.count("• Ran commands") == 1
    assert "  └ Grep" in rendered
    assert "  └ Find" in rendered
    assert "  └ Bash" in rendered
    assert "  └ Glob" in rendered


def test_runner_renders_tool_batch_only_after_every_call_completes(
    monkeypatch,
) -> None:
    session = AgentSession(memory_enabled=False)
    session.add_user_message("inspect")
    responses = iter(
        (
            parse_model_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "function": {
                                            "name": "grep",
                                            "arguments": "{}",
                                        },
                                    },
                                    {
                                        "id": "call-2",
                                        "function": {
                                            "name": "find",
                                            "arguments": "{}",
                                        },
                                    },
                                ],
                            }
                        }
                    ]
                }
            ),
            _text_response("done"),
        )
    )
    activity: list[str] = []

    def execute(*_arguments, **_keywords) -> str:
        activity.append("execute")
        assert "render" not in activity
        return "ok"

    def render_batch(self, calls) -> None:
        del self
        materialized = list(calls)
        activity.append("render")
        assert [call.id for call, _tool, _result in materialized] == [
            "call-1",
            "call-2",
        ]

    monkeypatch.setattr("citra.agent.runner.execute_tool_call", execute)
    monkeypatch.setattr(rendering.ToolCallRenderState, "render_batch", render_batch)
    model = SimpleNamespace(
        id="test-model",
        max_input_tokens=100_000,
        reasoning_effort=None,
    )
    workflow = SimpleNamespace(
        tool_set=ToolSet(core_tools=(), deferred_tools=()),
        get_task_steering=lambda *_arguments: None,
        get_system_prompt=lambda *_arguments: "",
        is_serial=False,
    )
    context = SimpleNamespace(
        ensure_active=lambda: None,
        workspace=SimpleNamespace(disabled_tool_ids=(), is_closing=False),
        workflow=workflow,
        config=SimpleNamespace(model=lambda: model),
    )

    AgentRunner(context, session, api_call=lambda _call: next(responses)).run_turn()

    assert activity == ["execute", "execute", "render"]


def test_two_runner_turns_group_calls_across_model_cycles(monkeypatch) -> None:
    output = _recording_console()
    monkeypatch.setattr(rendering, "console", output)
    session = AgentSession(memory_enabled=False)
    responses = iter(
        (
            _tool_response("grep", "call-1"),
            _tool_response("find", "call-2"),
            _tool_response("bash", "call-3"),
            _text_response("First turn complete."),
            _tool_response("glob", "call-4"),
            _text_response("Second turn complete."),
        )
    )
    requests: list[ModelCall] = []

    def fake_api(model_call: ModelCall) -> ModelResponse:
        requests.append(model_call)
        return next(responses)

    model = SimpleNamespace(
        id="llama",
        max_input_tokens=100_000,
        reasoning_effort=None,
    )
    workflow = SimpleNamespace(
        tool_set=ToolSet(core_tools=(), deferred_tools=()),
        get_task_steering=lambda *_arguments: None,
        get_system_prompt=lambda *_arguments: "",
        is_serial=False,
    )
    context = SimpleNamespace(
        ensure_active=lambda: None,
        workspace=SimpleNamespace(disabled_tool_ids=(), is_closing=False),
        workflow=workflow,
        config=SimpleNamespace(model=lambda: model),
    )
    runner = AgentRunner(context, session, api_call=fake_api)

    session.add_user_message("First simulated turn")
    runner.run_turn()
    session.add_user_message("Second simulated turn")
    runner.run_turn()

    rendered = output.export_text()
    assert rendered.count("• Explored") == 2
    assert rendered.count("• Ran commands") == 1
    assert rendered.index("• Explored") < rendered.index("• Ran commands")
    assert "First turn complete." in rendered
    assert "Second turn complete." in rendered
    assert requests
    assert all(isinstance(request, ModelCall) for request in requests)
    assert requests[0].messages[-1] == UserMessage("First simulated turn")


def _tool_response(name: str, call_id: str) -> ModelResponse:
    """Build one simulated model response containing a tool call."""
    return parse_model_response({
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    })


def _text_response(content: str) -> ModelResponse:
    """Build one simulated model response containing final assistant prose."""
    return parse_model_response(
        {"choices": [{"message": {"role": "assistant", "content": content}}]}
    )
