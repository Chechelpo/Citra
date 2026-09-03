"""Compatibility imports for the copied-project lifecycle context."""

from .workspace_context import AvailablePathAlias, WorkspaceContext

AgentRuntime = WorkspaceContext


__all__ = [
    "AvailablePathAlias",
    "AgentRuntime",
    "WorkspaceContext",
]
