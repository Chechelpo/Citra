"""Shared terminal palette and console for Citra's interactive UI."""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

BACKGROUND = "#181818"
SURFACE = "#222222"
MUTED = "#8a8a8a"
ACCENT = "#7aa2f7"

CITRA_THEME = Theme(
    {
        "citra.brand": "bold #e0af68",
        "citra.accent": f"bold {ACCENT}",
        "citra.muted": MUTED,
        "citra.success": "bold #9ece6a",
        "citra.warning": "bold #e0af68",
        "citra.error": "bold #f7768e",
        "citra.tool": "bold #7dcfff",
        "citra.path": "underline #bb9af7",
        "citra.border": "#444444",
        "citra.surface": f"on {SURFACE}",
    }
)

console = Console(theme=CITRA_THEME, highlight=False, soft_wrap=False)
