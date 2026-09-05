"""Pyrefly language-server definition."""

from ..interpreters import resolve_python
from ..language import Language
from .base import InstallCandidate, ServerDefinition

PYREFLY = ServerDefinition(
    id="pyrefly",
    executable="pyrefly",
    languages=(Language.PYTHON,),
    arguments=("lsp",),
    requires=("python",),
    interpreter_resolver=resolve_python,
    install_candidates=(
        InstallCandidate("pip", ("pyrefly",), ("pip", "install", "pyrefly")),
        InstallCandidate("uv", ("pyrefly",), ("uv", "tool", "install", "pyrefly")),
    ),
    install_hint="Install Pyrefly (for example: uv tool install pyrefly).",
)
