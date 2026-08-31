"""Pyright language-server definition."""

from ..interpreters import resolve_python
from ..language import Language
from .base import InstallCandidate, ServerDefinition

PYRIGHT = ServerDefinition(
    id="pyright",
    executable="pyright-langserver",
    languages=(Language.PYTHON,),
    arguments=("--stdio",),
    requires=("node",),
    settings={
        "python": {"analysis": {"diagnosticMode": "openFilesOnly"}},
        "pyright": {},
    },
    interpreter_resolver=resolve_python,
    install_candidates=(
        InstallCandidate(
            "pacman", ("pyright",), ("sudo", "pacman", "-S", "--needed", "pyright")
        ),
        InstallCandidate("npm", ("pyright",), ("npm", "install", "-g", "pyright")),
    ),
    install_hint="Install Pyright (for example: npm install -g pyright).",
)
