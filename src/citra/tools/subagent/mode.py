"""Compatibility aliases for the former subagent mode module."""

from .workflow import (
    SubagentMode,
    SubagentToolset,
    SubagentWorkflow,
    build_subagent_mode,
    build_subagent_workflow,
    default_subagent_toolset,
)

__all__ = [
    "SubagentMode",
    "SubagentToolset",
    "SubagentWorkflow",
    "build_subagent_mode",
    "build_subagent_workflow",
    "default_subagent_toolset",
]
