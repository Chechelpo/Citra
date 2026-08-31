from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from citra.context.session_context import WorkspaceContext
from citra.tools.linting import LintRunner
from citra.tools.skills.skill_registry import SkillRegistry
from citra.utils.browser_manager import BrowserManager
from citra.utils.managed_subprocess import ManagedSubprocesses
from citra.utils.model_tokenizer import tokenize
from citra.utils.repo_map import RepoMap
from citra.sandbox import WorkspaceSandbox
from citra.sandbox import SandboxedFilesystem

from citra.config.config_loader import CitraConfig

if TYPE_CHECKING:
    from citra.modes import Mode
    from citra.utils.lsp import LspManager
    from citra.workflows import Workflow, WorkflowRun, WorkflowRuntime

DEFAULT_CONTEXT_TOKEN_LIMIT = 2_000


@dataclass(frozen=True)
class AgentContext:
    """
    Data class holding the execution context for a single agent (subagent or main).
    """
    workspace: WorkspaceContext
    skills: SkillRegistry
    logger = logging.getLogger(__name__)

    lsp_manager: LspManager | None = None
    user_interactions: object | None = None
    subagents: object | None = None
    workflow_runtime: WorkflowRuntime | None = None
    workflow_run: WorkflowRun | None = None
    provided_config: CitraConfig | None = field(
        default=None,
        repr=False,
    )
    provided_mode: Mode | None = field(
        default=None,
        repr=False,
    )
    provided_workflow: Workflow | None = field(
        default=None,
        repr=False,
    )
    provided_sandbox: WorkspaceSandbox | None = field(
        default=None,
        repr=False,
    )
    __os: str = field(
        init=False,
    )
    __config: CitraConfig = field(
        init=False,
    )
    __mode: Mode = field(
        init=False,
    )
    __workflow: Workflow = field(
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

        workflow_runtime = self.workflow_runtime
        workflow = self.provided_workflow
        if workflow_runtime is not None:
            if (
                workflow is not None
                and workflow_runtime.workflow is not workflow
            ):
                raise ValueError(
                    "ExecutionContext workflow and WorkflowRuntime differ"
                )
            workflow = workflow_runtime.workflow

        mode = self.provided_mode
        if workflow is not None and mode is None:
            mode = workflow.initial_mode
        if mode is None:
            from citra.modes import ModeRegistry

            mode = ModeRegistry(
                config_path=os.environ.get("CITRA_CONFIG_PATH"),
            ).active_mode

        if workflow is None:
            from citra.workflows import simple_workflow

            workflow = simple_workflow(mode)

        if workflow_runtime is None:
            from citra.workflows import WorkflowRuntime

            workflow_runtime = WorkflowRuntime(
                workflow=workflow,
                workspace=self.workspace,
                operator_sandbox_config=config.sandbox,
                sandbox=self.provided_sandbox,
            )
            object.__setattr__(self, "workflow_runtime", workflow_runtime)
        elif (
            self.provided_sandbox is not None
            and self.provided_sandbox is not workflow_runtime.sandbox
        ):
            raise ValueError(
                "ExecutionContext sandbox is not owned by its WorkflowRuntime"
            )
        sandbox = workflow_runtime.sandbox
        self.workspace.provisioning.health_check_tools(
            sandbox,
            cwd=self.workspace.workspace,
        )
        self.workspace.write_runtime_manifest()
        filesystem = SandboxedFilesystem(
            sandbox
        )
        subprocesses = ManagedSubprocesses(sandbox)
        browser = BrowserManager(
            sandbox,
            self.workspace.workspace,
            request_timeout=config.browser.request_timeout,
            browsers_path=(
                self.workspace.provisioning.asset_path("playwright-browsers")
                or self.workspace.cache / "playwright"
            ),
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
            "_ExecutionContext__mode",
            mode,
        )

        object.__setattr__(
            self,
            "_ExecutionContext__workflow",
            workflow,
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
    def mode(self) -> Mode:
        return self.__mode

    @property
    def workflow(self) -> Workflow:
        return self.__workflow

    def activate_mode(
        self,
        mode: Mode,
        *,
        skills: SkillRegistry,
        workflow_run: WorkflowRun | None,
    ) -> None:
        """Bind one isolated mode turn to the persistent workflow runtime."""
        mode.validate()
        object.__setattr__(self, "_ExecutionContext__mode", mode)
        object.__setattr__(self, "skills", skills)
        object.__setattr__(self, "workflow_run", workflow_run)

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

    def close(self, *, force: bool = False) -> None:
        self.workspace.begin_closing()
        if force:
            # Hard shutdown has one aggregate process bound. Individual
            # service close calls below then become bookkeeping operations.
            self.workspace.processes.terminate_all(force=True)
        manager = self.lsp_manager
        close_lsp = getattr(manager, "close", None)
        if callable(close_lsp):
            try:
                close_lsp(force=force)
            except TypeError:
                close_lsp()
        subagents = self.subagents
        close_subagents = getattr(subagents, "close", None)
        if callable(close_subagents):
            try:
                close_subagents()
            except Exception:
                self.logger.exception(
                    "Failed to close subagent supervisor."
                )
        self.__browser.close(force=force)
        self.__subprocesses.close(force=force)

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
        return self.workspace.resolve_command(cmd) is not None

    def resolve_command(self, cmd: str) -> str | None:
        path = self.workspace.resolve_command(cmd)
        return str(path) if path is not None else None

    def ensure_active(self) -> None:
        self.workspace.ensure_active()

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
        try:
            result = manager.diagnostics_for_path(
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
