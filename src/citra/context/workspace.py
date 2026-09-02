"""Compatibility imports for the copied-project lifecycle context."""

from .session_context import AvailablePathAlias, WorkspaceContext

AgentRuntime = WorkspaceContext


__all__ = [
    "AvailablePathAlias",
    "AgentRuntime",
    "WorkspaceContext",
]
