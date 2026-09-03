import os
from pathlib import Path

from ._analysis import LintContextConfig, LintRuleConfig, LspContextConfig
from ._citra_config import CitraConfig, MemoryConfig, NotificationConfig
from ._model_config import ModelConfig, ModelConfigStore, RetryConfig
from ._sandbox_policy import SandboxPolicy
from ._tool_config import (
    BashConfig,
    BrowserConfig,
    SubprocessConfig,
    ToolConfigs,
    WebSearchConfig,
)


def _config_path() -> Path:
    """Handle config path."""
    raw = os.environ.get("CITRA_CONFIG_PATH")
    if not raw:
        raise RuntimeError("CITRA_CONFIG_PATH is not defined.")
    return Path(raw).expanduser().resolve()


def get_config() -> CitraConfig:
    """Return get config."""
    return CitraConfig.load(_config_path())


__all__ = (
    "BashConfig",
    "BrowserConfig",
    "CitraConfig",
    "ModelConfig",
    "ModelConfigStore",
    "MemoryConfig",
    "NotificationConfig",
    "LintContextConfig",
    "LintRuleConfig",
    "LspContextConfig",
    "RetryConfig",
    "SandboxPolicy",
    "SubprocessConfig",
    "ToolConfigs",
    "WebSearchConfig",
    "get_config",
)
