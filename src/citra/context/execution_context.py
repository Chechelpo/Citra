from dataclasses import dataclass, field
from pathlib import Path
import os
import platform
import shutil

from citra.context.workspace import WorkspaceContext
from citra.utils.sandbox import WorkspaceSandbox

from .config_loader import CitraConfig


@dataclass(frozen=True)
class ExecutionContext:
    workspace: WorkspaceContext

    __os: str = field(
        init=False,
    )
    __config: CitraConfig = field(
        init=False,
    )
    __sandbox: WorkspaceSandbox = field(
        init=False,
    )

    def __post_init__(
        self,
    ) -> None:
        os_name = platform.system().lower()

        if os_name == "darwin":
            os_name = "macos"

        config_path_raw = os.environ.get(
            "CITRA_CONFIG_PATH"
        )

        if config_path_raw is None:
            raise RuntimeError(
                "CITRA_CONFIG_PATH is not defined. "
                "Citra should be started through start.sh."
            )

        config = CitraConfig.load(
            Path(config_path_raw)
        )

        sandbox = WorkspaceSandbox(
            self.workspace
        )

        object.__setattr__(
            self,
            "_ExecutionContext__os",
            os_name,
        )

        object.__setattr__(
            self,
            "_ExecutionContext__config",
            config,
        )

        object.__setattr__(
            self,
            "_ExecutionContext__sandbox",
            sandbox,
        )

    @property
    def os(
        self,
    ) -> str:
        return self.__os

    @property
    def config(
        self,
    ) -> CitraConfig:
        return self.__config

    @property
    def sandbox(
        self,
    ) -> WorkspaceSandbox:
        return self.__sandbox

    @property
    def model_config(
        self,
    ):
        return self.__config.model

    @property
    def web_search_config(
        self,
    ):
        return self.__config.web_search

    def has_command(
        self,
        cmd: str,
    ) -> bool:
        return shutil.which(
            cmd
        ) is not None