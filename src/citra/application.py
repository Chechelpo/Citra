"""Citra process lifecycle and long-lived service ownership."""

from __future__ import annotations

import atexit
import os
from pathlib import Path

from .agent import AgentSession, UserInteractionBroker
from .agent.runner import AgentRunner, ApiCall
from .commands import COMMAND_REGISTRY
from .context import CitraConfig, ExecutionContext, WorkspaceContext
from .tools.default_registry import TOOL_REGISTRY
from .tools.lsp import LspConfig, LspManager
from .utils.chat_completions_api import call_api
from .utils.sandbox import WorkspaceSandbox
from .utils.terminal import RESET, YELLOW
from .tools.skills.skill_registry import SkillRegistry


class CitraApplication:
    """Own everything that must survive individual agent turns."""

    def __init__(
        self,
        *,
        config: CitraConfig,
        source_workspace: Path,
        api_call: ApiCall = call_api,
    ) -> None:
        self.config = config
        self.source_workspace = source_workspace.resolve()
        self.session = AgentSession()
        self.workspace = WorkspaceContext.create(
            config=config.workspace_context,
            workspace=self.source_workspace,
        )
        self.skills = SkillRegistry(
            agent_session=self.session,
            skills_root=None
        )
        try:
            self.interactions = UserInteractionBroker()
            self.lsp_manager = LspManager(
                self.workspace,
                WorkspaceSandbox(
                    self.workspace,
                    config=config.sandbox,
                ),
                config=LspConfig(
                    enabled=config.lsp.enabled,
                    startup_timeout=config.lsp.startup_timeout,
                    request_timeout=config.lsp.request_timeout,
                    diagnostics_timeout=config.lsp.diagnostics_timeout,
                ),
            )
            self.context = ExecutionContext(
                self.workspace,
                skills=self.skills,
                lsp_manager=self.lsp_manager,
                user_interactions=self.interactions,
                provided_config=config,
            )
            self.runner = AgentRunner(
                self.context,
                self.session,
                api_call=api_call,
            )
        except Exception:
            self.workspace.cleanup()
            raise
        self._closed = False
        atexit.register(self.close)

    @classmethod
    def create(
        cls,
        *,
        config: CitraConfig | None = None,
        source_workspace: str | Path | None = None,
        api_call: ApiCall = call_api,
    ) -> CitraApplication:
        config = config or CitraConfig.load()
        source = Path(
            source_workspace
            or config.workspace_context.permanent_workspace
            or os.getcwd()
        ).expanduser().resolve()
        return cls(config=config, source_workspace=source, api_call=api_call)

    def run_agent_turn(self) -> None:
        self.runner.run_turn()

    def handle_command(self, user_input: str) -> bool:
        body = user_input[1:]
        parts = body.split(None, 1)
        command_id = parts[0] if parts else ""
        args = parts[1].strip() if len(parts) > 1 else ""
        if command_id == "exit":
            command_id = "q"
        command = COMMAND_REGISTRY.instantiate(command_id, self.context)
        if command is None:
            print(
                f"{YELLOW}⏺ Unknown command: /{command_id}. "
                f"Type /help for available commands.{RESET}"
            )
            return True
        result = command.run(args)
        if result.output:
            print(f"\n{result.output}")
        if result.clear_messages:
            self.session.clear_history(clear_memory=True)
        return not result.exit

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        self.interactions.close()

        try:
            self.lsp_manager.close()
        finally:
            try:
                self.context.close()
            finally:
                TOOL_REGISTRY.release_session(self.session)
                self.workspace.cleanup()

    def __enter__(self) -> CitraApplication:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
