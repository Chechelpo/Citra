"""Compatibility imports for the lifecycle workspace implementation.

The authoritative implementation lives in :mod:`citra.context.turn_workspace`.
Keeping this module prevents older embedders from silently importing the
obsolete unsandboxed duplicate that previously lived here.
"""

from .turn_workspace import AvailablePathAlias, WorkspaceContext


__all__ = [
    "AvailablePathAlias",
    "WorkspaceContext",
]
