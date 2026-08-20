"""Typed configuration primitives for the LSP subsystem.

The package accepts configuration through plain frozen dataclasses.
The host application may construct these from TOML, JSON, CLI options,
or any other source — the ``lsp/`` package does not care where they
originate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class LspConfig:
    """Top-level configuration for the LSP subsystem."""

    enabled: bool = True
    startup_timeout: float = 30.0
    request_timeout: float = 15.0
    diagnostics_timeout: float = 10.0


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for a single language-server invocation.

    Attributes:
        command: The server command as a tuple of arguments (argv).
        environment: Environment variables to pass to the server process.
        extensions: File extensions (with leading dot) this server handles.
        initialization_options: ``initializationOptions`` sent in ``initialize``.
        settings: Workspace ``settings`` sent via ``workspace/didChangeConfiguration``.
    """

    command: tuple[str, ...]
    environment: Mapping[str, str] = field(default_factory=dict)
    extensions: tuple[str, ...] = ()
    initialization_options: Mapping[str, Any] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecursiveDiagnosticsConfig:
    """Configuration for recursive (whole-project) diagnostics.

    Controls bounds and filtering when discovering and analysing many
    source files in one pass.
    """

    max_files: int = 2000
    max_diagnostics: int = 250
    include_source: bool = False
    excludes: tuple[str, ...] = ()
