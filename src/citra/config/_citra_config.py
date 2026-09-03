from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import tomllib

from citra.config._analysis import (
    LintContextConfig,
    LspContextConfig,
    load_lint_config,
    load_lsp_config,
)
from citra.config._constants import LINTING_CONFIG_FILE, TOOLS_CONFIG_FILE
from citra.config._model_config import ModelConfig, ModelConfigStore
from citra.config._sandbox_policy import SandboxPolicy
from citra.config._tool_config import (
    BashConfig,
    BrowserConfig,
    SubprocessConfig,
    ToolConfigs,
    WebSearchConfig,
)


@dataclass(frozen=True)
class MemoryConfig:
    """Represent MemoryConfig."""
    enabled: bool = True


@dataclass(frozen=True)
class NotificationConfig:
    """Represent NotificationConfig."""
    prompt_bell: bool = False

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
    memory: MemoryConfig = MemoryConfig()
    notifications: NotificationConfig = NotificationConfig()
    default_model_profile: str | None = None

    @classmethod
    def load(
        cls,
        config_folder_path: str | Path | None = None,
    ) -> CitraConfig:
        """Handle load."""
        if config_folder_path is None:
            raw = os.environ.get("CITRA_CONFIG_PATH")
            if not raw:
                raise RuntimeError("CITRA_CONFIG_PATH is not defined.")
            config_folder_path = raw

        config_dir = Path(config_folder_path).expanduser().resolve()
        if not config_dir.is_dir():
            raise NotADirectoryError(
                f"Citra config directory does not exist: {config_dir}"
            )

        tools_raw = _load_toml(config_dir / TOOLS_CONFIG_FILE)
        lint_path = config_dir / LINTING_CONFIG_FILE
        lint = (
            load_lint_config(_load_toml(lint_path))
            if lint_path.is_file()
            else LintContextConfig()
        )

        return CitraConfig(
            model_config_store=ModelConfigStore.load(config_dir),
            tools=ToolConfigs.create(tools_raw),
            sandbox_policy=SandboxPolicy.load(config_dir),
            lsp=load_lsp_config(tools_raw),
            lint=lint,
        )

    @classmethod
    def create(cls, config_folder_path: str | Path) -> CitraConfig:
        """Construct the aggregate from the canonical config directory."""
        return cls.load(config_folder_path)

    def model(self, name: str | None = None) -> ModelConfig:
        """Handle model."""
        return self.model_config_store.get(name or self.default_model_profile)

    def models(self) -> tuple[str, ...]:
        """Handle models."""
        return self.model_config_store.names()

    def with_default_model_profile(self, name: str | None) -> CitraConfig:
        """Handle with default model profile."""
        if name is not None:
            self.model_config_store.get(name)
        return replace(self, default_model_profile=name)

    @property
    def web_search(self) -> WebSearchConfig:
        """Handle web search."""
        return self.tools.web_search

    @property
    def bash(self) -> BashConfig:
        """Handle bash."""
        return self.tools.bash

    @property
    def subprocess(self) -> SubprocessConfig:
        """Handle subprocess."""
        return self.tools.subprocess

    @property
    def browser(self) -> BrowserConfig:
        """Handle browser."""
        return self.tools.browser


def _load_toml(path: Path) -> dict[str, object]:
    """Handle load toml."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as file:
        raw = tomllib.load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must contain a TOML table.")
    return raw
