"""JavaScript/TypeScript language-server definition."""

from ..language import Language
from .base import InstallCandidate, ServerDefinition


TYPESCRIPT_LANGUAGE_SERVER = ServerDefinition(
    id="typescript",
    executable="typescript-language-server",
    languages=(Language.JAVASCRIPT, Language.TYPESCRIPT),
    arguments=("--stdio",),
    requires=("node",),
    install_candidates=(
        InstallCandidate(
            "pacman",
            ("typescript-language-server", "typescript"),
            ("sudo", "pacman", "-S", "--needed", "typescript-language-server", "typescript"),
        ),
        InstallCandidate(
            "npm",
            ("typescript-language-server", "typescript"),
            ("npm", "install", "-g", "typescript-language-server", "typescript"),
        ),
    ),
    install_hint="Install typescript-language-server and TypeScript.",
)
