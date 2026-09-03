from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from citra.context.workspace_context import WorkspaceContext
from citra.tools.linting import LintRunner
from citra.tools.skills.skill_registry import SkillRegistry
from citra.utils.browser_manager import BrowserManager
from citra.utils.managed_subprocess import ManagedSubprocesses
from citra.utils.model_tokenizer import tokenize
from citra.utils.repo_map import RepoMap
from citra.sandbox.sandbox import WorkspaceSandbox
from citra.sandbox.sandboxed_filesystem import SandboxedFilesystem

from citra.config import CitraConfig, ModelConfig, WebSearchConfig

if TYPE_CHECKING:
    from citra.utils.lsp import LspManager
    from citra.workflows import SingleModeWorkflow, WorkflowRuntime

DEFAULT_CONTEXT_TOKEN_LIMIT = 2_000


@dataclass(frozen=True)
class ExecutionContext:
    """
    Data class holding the execution context for a single agent (subagent or main).
    """
    workspace: WorkspaceContext
    skills: SkillRegistry
    config: CitraConfig
    workflow_runtime: WorkflowRuntime
    logger = logging.getLogger(__name__)

    lsp_manager: LspManager | None = None
    user_interactions: object | None = None
    subagents: object | None = None
    __os: str = field(
        init=False,
    )
    __workflow: SingleModeWorkflow = field(
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
        from citra.workflows import WorkflowRuntime

        if not isinstance(self.workspace, WorkspaceContext):
            raise TypeError("workspace must be a WorkspaceContext")
        if not isinstance(self.skills, SkillRegistry):
            raise TypeError("skills must be a SkillRegistry")
        if not isinstance(self.config, CitraConfig):
            raise TypeError("config must be a CitraConfig")
        if not isinstance(self.workflow_runtime, WorkflowRuntime):
            raise TypeError("workflow_runtime must be a WorkflowRuntime")

        os_name = platform.system().lower()

        if os_name == "darwin":
            os_name = "macos"

        workflow_runtime = self.workflow_runtime
        workflow = workflow_runtime.active_workflow
        sandbox = workflow_runtime.sandbox
        self.workspace.provisioning.health_check_tools(
            sandbox,
            cwd=self.workspace.workspace,
        )
        self.workspace.write_runtime_manifest()
        filesystem: SandboxedFilesystem = SandboxedFilesystem(sandbox)
        subprocesses = ManagedSubprocesses(sandbox)
        browser = BrowserManager(
            sandbox,
            self.workspace.workspace,
            request_timeout=self.config.browser.request_timeout,
            browsers_path=(
                self.workspace.provisioning.asset_path("playwright-browsers")
                or self.workspace.cache / "playwright"
            ),
        )
        repo_map = RepoMap(self.workspace)
        lint_runner = LintRunner(
            self.workspace,
            sandbox,
            self.config.lint,
        )

        object.__setattr__(
            self,
            "_ExecutionContext__os",
            os_name,
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
    def workflow(self) -> SingleModeWorkflow:
        return self.__workflow

    def activate_workflow(
        self,
        workflow: SingleModeWorkflow,
        *,
        skills: SkillRegistry,
    ) -> None:
        """Bind one serial phase to the persistent workflow runtime."""
        workflow.validate()
        root = self.workflow_runtime.workflow
        if not root.is_serial:
            raise RuntimeError("Only serial workflows can activate a new phase")
        active = self.workflow_runtime.require_active_run().current_step.workflow
        if workflow is not active:
            raise ValueError("Activated workflow is not the runtime's active phase")
        object.__setattr__(self, "_ExecutionContext__workflow", workflow)
        object.__setattr__(self, "skills", skills)

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
        if manager is not None:
            manager.close(force=force)
        subagents = self.subagents
        if subagents is not None:
            from citra.tools.subagent.supervisor import SubagentSupervisor
            if isinstance(subagents, SubagentSupervisor):
                try:
                    subagents.close()
                except Exception:
                    self.logger.exception(
                        "Failed to close subagent supervisor."
                    )
        self.__browser.close(force=force)
        self.__subprocesses.close(force=force)

    def model_config(self) -> ModelConfig:
        return self.config.model()

    @property
    def web_search_config(
        self,
    ) -> WebSearchConfig:
        return self.config.tools.web_search

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

        model_id = self.config.model().id
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
