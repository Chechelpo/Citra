"""Typed configuration primitives for the LSP subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class LspConfig:
    """Represent LspConfig."""
    enabled: bool = True
    startup_timeout: float = 30.0
    request_timeout: float = 15.0
    diagnostics_timeout: float = 10.0
    cold_diagnostics_timeout: float = 45.0
    json_fallback: bool = True


@dataclass(frozen=True)
class ServerConfig:
    """Represent ServerConfig."""
    command: tuple[str, ...]
    environment: Mapping[str, str] = field(default_factory=dict)
    extensions: tuple[str, ...] = ()
    initialization_options: Mapping[str, Any] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)
    cold_diagnostics_timeout: float | None = None


@dataclass(frozen=True)
class RecursiveDiagnosticsConfig:
    """Represent RecursiveDiagnosticsConfig."""
    max_files: int = 2000
    max_diagnostics: int = 250
    include_source: bool = False
    excludes: tuple[str, ...] = ()
