"""Shared diagnostic formatting and built-in syntax fallbacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


_SEVERITIES = {1: "error", 2: "warning", 3: "information", 4: "hint"}


def format_diagnostics(
    diagnostics: list[Any],
    *,
    path: Path,
    display_path: Callable[[Path], str] | None = None,
    limit: int = 250,
) -> str | None:
    shown_path = display_path(path) if display_path is not None else str(path)
    lines: list[str] = []
    for diagnostic in diagnostics[:limit]:
        if not isinstance(diagnostic, dict):
            continue
        range_value = diagnostic.get("range")
        if not isinstance(range_value, dict):
            range_value = {}
        start = range_value.get("start")
        if not isinstance(start, dict):
            start = {}
        line = start.get("line", 0)
        character = start.get("character", 0)
        if not isinstance(line, int):
            line = 0
        if not isinstance(character, int):
            character = 0
        severity = _SEVERITIES.get(diagnostic.get("severity"), "diagnostic")
        source_raw = diagnostic.get("source")
        source = f" [{source_raw}]" if isinstance(source_raw, str) and source_raw else ""
        message = diagnostic.get("message", "")
        if not isinstance(message, str):
            message = str(message)
        lines.append(
            f"{shown_path}:{line + 1}:{character + 1}: {severity}{source}: {message}"
        )
    return "\n".join(lines) if lines else None


def json_fallback_diagnostics(text: str) -> list[dict[str, Any]]:
    try:
        json.loads(text)
    except json.JSONDecodeError as error:
        return [
            {
                "range": {
                    "start": {"line": max(0, error.lineno - 1), "character": max(0, error.colno - 1)},
                    "end": {"line": max(0, error.lineno - 1), "character": max(0, error.colno)},
                },
                "severity": 1,
                "source": "json",
                "message": error.msg,
            }
        ]
    return []
