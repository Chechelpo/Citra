import os
from pathlib import Path

from ._citra_config import CitraConfig
from ._model_config import ModelConfigStore
from ._sandbox_policy import SandboxPolicy
from ._tool_config import ToolConfigs


def _config_path() -> Path:
    raw = os.environ.get('CITRA_CONFIG_PATH')
    if not raw:
        raise RuntimeError('CITRA_CONFIG_PATH is not defined.')
    return Path(raw).resolve()


def get_config() -> CitraConfig:
    return CitraConfig.load(_config_path())


__all__ = (
    'CitraConfig',
    'SandboxPolicy',
    'ToolConfigs',
    ''
    'ModelConfigStore',
    'get_config',
)
