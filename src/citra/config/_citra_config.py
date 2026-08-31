from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from citra.config._model_config import ModelConfigStore
from citra.config._sandbox_policy import SandboxPolicy
from citra.config._tool_config import ToolConfigs
from citra.config._analysis import LspContextConfig, LintContextConfig

@dataclass(frozen=True)
class CitraConfig:
    """
    Fully resolved Citra configuration.

    Domain-specific parsing is delegated to dedicated configuration modules.
    This class only owns aggregation and lifecycle-level configuration.
    """

    model_config_store: ModelConfigStore
    tools: ToolConfigs
    sandbox_policy: SandboxPolicy

    lsp: LspContextConfig = LspContextConfig()
    lint: LintContextConfig = LintContextConfig()

    default_model_profile: str | None = None

    @staticmethod
    def load(config_folder_path:Path) -> CitraConfig:
        return CitraConfig(
            model_config_store=ModelConfigStore.load(config_folder_path),
            tools=ToolConfigs.load(config_folder_path),
            sandbox_policy= SandboxPolicy.load(config_folder_path)
        )
    