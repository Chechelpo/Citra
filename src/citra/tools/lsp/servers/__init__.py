"""Built-in language-server definitions."""

from .base import ServerDefinition
from .pyright import PYRIGHT
from .typescript import TYPESCRIPT_LANGUAGE_SERVER


SERVERS: dict[str, ServerDefinition] = {
    PYRIGHT.id: PYRIGHT,
    TYPESCRIPT_LANGUAGE_SERVER.id: TYPESCRIPT_LANGUAGE_SERVER,
}

__all__ = [
    "PYRIGHT",
    "SERVERS",
    "ServerDefinition",
    "TYPESCRIPT_LANGUAGE_SERVER",
]

