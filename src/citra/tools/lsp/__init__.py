"""Sandboxed, lifecycle-scoped language-server subsystem."""

from .client import LspClient
from .config import LspConfig, ServerConfig
from .errors import LspError
from .language import Language, detect_language
from .manager import LspManager
from .positions import SourcePosition, SourceRange

__all__ = [
    "Language",
    "LspClient",
    "LspConfig",
    "LspError",
    "LspManager",
    "ServerConfig",
    "SourcePosition",
    "SourceRange",
    "detect_language",
]

