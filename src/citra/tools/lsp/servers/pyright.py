"""Pyright language-server definition."""

from .base import ServerDefinition


PYRIGHT = ServerDefinition(
    id="pyright",
    executable="pyright-langserver",
    arguments=("--stdio",),
    install_hint="Install Pyright (for example: pipx install pyright).",
)
