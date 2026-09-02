from __future__ import annotations

from typing import TYPE_CHECKING

from citra.config import (
    BashConfig,
    BrowserConfig,
    CitraConfig,
    LintContextConfig,
    LintRuleConfig,
    LspContextConfig,
    MemoryConfig,
    ModelConfig,
    NotificationConfig,
    RetryConfig,
    SandboxPolicy,
    SubprocessConfig,
    WebSearchConfig,
)
from .runtime import (
    CopyPolicy,
    RuntimeAsset,
    RuntimeProcessSupervisor,
    RuntimeProvisionError,
    RuntimeProvisioner,
    ToolDefinition,
)
from .session_context import RuntimeClosingError, RuntimeState, WorkspaceContext
from citra.config import ModelConfigStore

AgentRuntime = WorkspaceContext

if TYPE_CHECKING:
    from .agent_context import ExecutionContext


def __getattr__(name: str):
    if name == "ExecutionContext":
        from .agent_context import ExecutionContext

        return ExecutionContext
    raise AttributeError(name)

__all__ = [
    "ModelConfigStore",
    "BashConfig",
    "BrowserConfig",
    "ExecutionContext",
    "CitraConfig",
    "LintContextConfig",
    "LintRuleConfig",
    "MemoryConfig",
    "ModelConfig",
    "LspContextConfig",
    "NotificationConfig",
    "RetryConfig",
    "SandboxPolicy",
    "SubprocessConfig",
    "WebSearchConfig",
    "WorkspaceContext",
    "AgentRuntime",
    "RuntimeClosingError",
    "RuntimeState",
    "CopyPolicy",
    "RuntimeAsset",
    "RuntimeProvisionError",
    "RuntimeProcessSupervisor",
    "RuntimeProvisioner",
    "ToolDefinition",
]
