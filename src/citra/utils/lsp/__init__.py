"""Sandboxed, lifecycle-scoped language-server subsystem."""

from .client import LspClient, configuration_for_section
from .config import LspConfig, ServerConfig
from .errors import LspError, LspUnavailable
from .language import Language, detect_language
from .manager import ClientKey, LspClientHandle, LspManager
from .positions import SourcePosition, SourceRange
from .servers import InstallCandidate, ServerDefinition

__all__ = [
    "ClientKey",
    "Language",
    "LspClient",
    "LspClientHandle",
    "LspConfig",
    "InstallCandidate",
    "LspError",
    "LspManager",
    "LspUnavailable",
    "ServerConfig",
    "ServerDefinition",
    "SourcePosition",
    "SourceRange",
    "configuration_for_section",
    "detect_language",
]

