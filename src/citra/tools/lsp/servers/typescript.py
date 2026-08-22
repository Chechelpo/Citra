"""JavaScript/TypeScript language-server definition."""

from .base import ServerDefinition


TYPESCRIPT_LANGUAGE_SERVER = ServerDefinition(
    id="typescript-language-server",
    executable="typescript-language-server",
    arguments=("--stdio",),
    install_hint=(
        "Install both typescript-language-server and typescript "
        "(for example with your system npm/pnpm toolchain)."
    ),
)

