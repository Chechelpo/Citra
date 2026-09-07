"""Tests for terminal activity rendering helpers."""

from citra.cli.rendering import format_elapsed


def test_format_elapsed_uses_seconds_for_short_work() -> None:
    assert format_elapsed(12.4) == "12s"


def test_format_elapsed_uses_minutes_for_long_work() -> None:
    assert format_elapsed(125.2) == "2m 5s"


def test_format_elapsed_never_reports_negative_time() -> None:
    assert format_elapsed(-1) == "0s"
