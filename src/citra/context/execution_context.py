from .libraries import Libraries
from dataclasses import dataclass, field
from pathlib import Path
import os
import platform
import shutil

from citra.context.turn_workspace import WorkspaceContext
from citra.utils.sandbox import WorkspaceSandbox
from citra.utils.sandboxed_filesystem import SandboxedFilesystem
from citra.utils.browser_manager import BrowserManager
from citra.utils.managed_subprocess import ManagedSubprocesses

from .config_loader import CitraConfig


@dataclass(frozen=True)
class ExecutionContext:
    workspace: WorkspaceContext
    libraries: Libraries = field(
        default_factory=Libraries
    )
    lsp_manager: object | None = None
    user_interactions: object | None = None
    provided_config: CitraConfig | None = field(
        default=None,
        repr=False,
    )
    __os: str = field(
        init=False,
    )
    __config: CitraConfig = field(
        init=False,
    )
    __sandbox: WorkspaceSandbox = field(
        init=False,
    )
    __filesystem: SandboxedFilesystem = field(
        init=False,
    )
    __subprocesses: ManagedSubprocesses = field(init=False)
    __browser: BrowserManager = field(init=False)

    def __post_init__(
        self,
    ) -> None:
        os_name = platform.system().lower()

        if os_name == "darwin":
            os_name = "macos"

        config = self.provided_config

        if config is None:
            config_path_raw = os.environ.get(
                "CITRA_CONFIG_PATH"
            )

            if config_path_raw is None:
                raise RuntimeError(
                    "CITRA_CONFIG_PATH is not defined. "
                    "Citra should be started through start.sh."
                )

            config = CitraConfig.load()

        sandbox = WorkspaceSandbox(
            self.workspace,
            config=config.sandbox,
        )
        filesystem = SandboxedFilesystem(
            sandbox
        )
        subprocesses = ManagedSubprocesses(sandbox)
        browser = BrowserManager(
            sandbox,
            self.workspace.workspace,
            request_timeout=config.browser.request_timeout,
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

        object.__setattr__(
            self,
            "_ExecutionContext__filesystem",
            filesystem,
        )
        object.__setattr__(
            self,
            "_ExecutionContext__subprocesses",
            subprocesses,
        )
        object.__setattr__(self, "_ExecutionContext__browser", browser)

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
    def filesystem(
        self,
    ) -> SandboxedFilesystem:
        return self.__filesystem

    @property
    def subprocesses(self) -> ManagedSubprocesses:
        return self.__subprocesses

    @property
    def browser(self) -> BrowserManager:
        return self.__browser

    def close(self) -> None:
        self.__browser.close()
        self.__subprocesses.close()

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
