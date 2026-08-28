from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citra.context.turn_workspace import WorkspaceContext


@dataclass(frozen=True)
class RuntimeDiscoveryResult:
    """Filesystem requirements discovered before sandbox creation."""

    readonly_binds: tuple[Path, ...] = ()


class RuntimeDiscovery(ABC):
    """Base class for one host-runtime discovery utility."""

    @classmethod
    @abstractmethod
    def discover(
        cls,
        workspace: WorkspaceContext,
    ) -> RuntimeDiscoveryResult:
        """Return host paths that must be exposed read-only."""
