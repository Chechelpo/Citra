"""Declarative language-server and installer definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..language import Language


@dataclass(frozen=True)
class InstallCandidate:
    manager: str
    packages: tuple[str, ...]
    command: tuple[str, ...]


CommandFactory = Callable[[str, Path, Path], tuple[str, ...]]


@dataclass(frozen=True)
class ServerDefinition:
    id: str
    executable: str
    languages: tuple[Language, ...]
    arguments: tuple[str, ...] = ()
    settings: Mapping[str, Any] = field(default_factory=dict)
    initialization_options: Mapping[str, Any] = field(default_factory=dict)
    cold_diagnostics_timeout: float | None = None
    install_candidates: tuple[InstallCandidate, ...] = ()
    requires: tuple[str, ...] = ()
    command_factory: CommandFactory | None = None
    install_hint: str = "Install the language server with a supported package manager."
