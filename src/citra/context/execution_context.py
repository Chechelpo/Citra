import logging
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
from citra.utils.repo_map import RepoMap
from citra.utils.model_tokenizer import tokenize
from citra.tools.skills.skill_registry import SkillRegistry
from citra.tools.linting import LintRunner

from .config_loader import CitraConfig


DEFAULT_CONTEXT_TOKEN_LIMIT = 2_000


@dataclass(frozen=True)
class ExecutionContext:
    workspace: WorkspaceContext
    skills: SkillRegistry
    logger = logging.getLogger(__name__)

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
    __repo_map: RepoMap = field(init=False)
    __lint_runner: LintRunner = field(init=False)

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
        repo_map = RepoMap(self.workspace)
        lint_runner = LintRunner(
            self.workspace,
            sandbox,
            config.lint,
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
        object.__setattr__(
            self,
            "_ExecutionContext__repo_map",
            repo_map,
        )
        object.__setattr__(
            self,
            "_ExecutionContext__lint_runner",
            lint_runner,
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

    @property
    def repo_map(self) -> RepoMap:
        return self.__repo_map

    def close(self) -> None:
        manager = self.lsp_manager
        close_lsp = getattr(manager, "close", None)
        if callable(close_lsp):
            close_lsp()
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

    def truncate_output(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        """Cap model-facing output using the active model tokenizer."""
        if max_tokens is None:
            max_tokens = DEFAULT_CONTEXT_TOKEN_LIMIT

        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        model_id = self.__config.model().id
        total_tokens = tokenize(model_id, text)

        if total_tokens <= max_tokens:
            return text

        notice = f"\n... <context truncated after {max_tokens} tokens>"
        notice_tokens = tokenize(model_id, notice)
        content_budget = max_tokens - notice_tokens

        if content_budget <= 0:
            return notice

        cut = max(1, int(len(text) * content_budget / total_tokens))
        candidate = text[:cut]

        newline = candidate.rfind("\n")
        if newline != -1:
            candidate = candidate[:newline]

        candidate_tokens = tokenize(model_id, candidate)

        while candidate and candidate_tokens > content_budget:
            cut = max(
                1,
                int(
                    len(candidate)
                    * content_budget
                    / candidate_tokens
                    * 0.98
                ),
            )
            candidate = candidate[:cut]

            newline = candidate.rfind("\n")
            if newline != -1:
                candidate = candidate[:newline]

            candidate_tokens = tokenize(
                model_id,
                candidate,
            )

        return candidate + notice

    def diagnostics_for_path(
        self,
        path_raw: str,
    ) -> str | None:
        """Return advisory LSP diagnostics for a workspace path when available."""
        manager = self.lsp_manager

        if manager is None:
            return None

        collect = getattr(
            manager,
            "diagnostics_for_path",
            None,
        )
        if not callable(collect):
            return None

        try:
            result = collect(
                path_raw,
                filesystem=self.filesystem,
            )
            return result if isinstance(result, str) else None
        except Exception:
            self.logger.exception(
                "Could not collect LSP diagnostics for %s",
                path_raw,
            )
            return None

    def lint_for_path(
        self,
        path_raw: str,
    ) -> str | None:
        """Return configured lint failures for a modified project path."""
        try:
            return self.__lint_runner.lint_for_path(
                path_raw
            )
        except Exception:
            self.logger.exception(
                "Could not run configured lint checks for %s",
                path_raw,
            )
            return None
