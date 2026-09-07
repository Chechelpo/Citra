"""Deterministic visual tests for the centralized Rich CLI."""

import json
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from citra.cli import rendering
from citra.cli.rendering import format_elapsed
from citra.cli.theme import CITRA_THEME


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
    config = SimpleNamespace(model=lambda: SimpleNamespace(id="gpt-professional"))

    rendering.print_header(config, Path("/work/project"))
    rendered = output.export_text()

    assert "CITRA" in rendered
    assert "gpt-professional" in rendered
    assert "agentic coding workspace" in rendered
    assert "─" * 40 in rendered


def test_structured_panels_are_capped_on_ultrawide_terminals(monkeypatch) -> None:
    output = _recording_console(width=240)
    monkeypatch.setattr(rendering, "console", output)
    config = SimpleNamespace(model=lambda: SimpleNamespace(id="gpt-professional"))

    rendering.print_header(config, Path("/work/project"))

    assert max(map(len, output.export_text().splitlines())) == 120


def test_edit_tool_call_displays_an_actual_diff(monkeypatch) -> None:
    output = _recording_console()
    monkeypatch.setattr(rendering, "console", output)
    tool = SimpleNamespace(
        id="edit",
        format_call_log=lambda _arguments: "--- a/app.py\n+++ b/app.py\n@@\n-old\n+new",
    )
    call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "edit", "arguments": json.dumps({"path": "app.py"})},
    }

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
        format_call_log=lambda _arguments: "src/app.py\ntests/test_app.py",
    )
    call = {
        "id": "call-2",
        "type": "function",
        "function": {"name": "read", "arguments": "{}"},
    }

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
