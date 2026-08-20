# src/citra/utils/terminal.py

"""
Shared ANSI styling and terminal helpers.

Kept dependency-free so that both the main agent loop and individual
commands can render consistent output.
"""

from __future__ import annotations

import os


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

BLUE = "\033[34m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"


def separator() -> str:
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80

    return f"{DIM}{'─' * min(width, 80)}{RESET}"
