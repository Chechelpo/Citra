"""Citra process lifecycle and long-lived service ownership."""

from __future__ import annotations

import atexit
import os
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock

from .agent import AgentSession, UserInteractionBroker
from .agent.runner import AgentRunner, ApiCall
from .commands import COMMAND_REGISTRY
from .context import CitraConfig, ExecutionContext, WorkspaceContext
from .modes import Mode, ModeRegistry
from  citra.utils.lsp import LspConfig, LspManager
from .tools.skills.skill_registry import SkillRegistry
from .utils.chat_completions_api import call_api
from  citra.sandbox import WorkspaceSandbox
from .utils.terminal import RESET, YELLOW


class CitraApplication:
    """Own everything that must survive individual agent turns."""

    def __init__(
        self,
        *,
        config: CitraConfig,
        source_workspace: Path,
        api_call: ApiCall = call_api,
        mode: Mode | None = None,
        mode_registry: ModeRegistry | None = None,
    ) -> None:
        self.config = config
        self.source_workspace = source_workspace.resolve()
        self.mode_registry = mode_registry or ModeRegistry(
            config_path=os.environ.get("CITRA_CONFIG_PATH"),
        )
        self.mode = mode or self.mode_registry.active_mode
        self.mode.validate()
        self.session = AgentSession(
            memory_enabled=config.memory.enabled,
        )
        self._close_lock = Lock()
        self._closing = False
        self._closed = False
        self._hard_shutdown = Event()
        workspace_config = replace(
            config.workspace_context,
            direct_source=self.mode.sandbox_config.mode.uses_direct_source,
        )
        self.workspace = WorkspaceContext.create(
            config=workspace_config,
            workspace=self.source_workspace,
            runtime_config=config.runtime,
            browser_path=config.browser.browsers_path,
        )
        try:
            self.skills = SkillRegistry(
                agent_session=self.session,
                memory_enabled=config.memory.enabled,
                skills_root=(
                    Path(
                        os.environ.get(
                            "CITRA_INSTALL_ROOT",
                            str(Path(__file__).resolve().parents[2]),
                        )
                    )
                    / "skills"
                ),
            )
            self.interactions = UserInteractionBroker()
            sandbox = WorkspaceSandbox(
                self.workspace,
                config=config.sandbox,
                mode_config=self.mode.sandbox_config,
            )
            self.lsp_manager = LspManager(
                self.workspace,
                sandbox,
                config=LspConfig(
                    enabled=config.lsp.enabled,
                    startup_timeout=config.lsp.startup_timeout,
                    request_timeout=config.lsp.request_timeout,
                    diagnostics_timeout=config.lsp.diagnostics_timeout,
                    cold_diagnostics_timeout=config.lsp.cold_diagnostics_timeout,
                    json_fallback=config.lsp.json_fallback,
                ),
            )
            self.context = ExecutionContext(
                self.workspace,
                skills=self.skills,
                lsp_manager=self.lsp_manager,
                user_interactions=self.interactions,
                provided_config=config,
                provided_mode=self.mode,
                provided_sandbox=sandbox,
            )
            self.runner = AgentRunner(
                self.context,
                self.session,
                api_call=api_call,
            )
        except Exception:
            self.workspace.cleanup()
            raise
        atexit.register(self.close)

    @classmethod
    def create(
        cls,
        *,
        config: CitraConfig | None = None,
        source_workspace: str | Path | None = None,
        api_call: ApiCall = call_api,
        mode: Mode | None = None,
        mode_registry: ModeRegistry | None = None,
    ) -> CitraApplication:
        config = config or CitraConfig.load()
        source = Path(
            source_workspace
            or config.workspace_context.permanent_workspace
            or os.getcwd()
        ).expanduser().resolve()
        return cls(
            config=config,
            source_workspace=source,
            api_call=api_call,
            mode=mode,
            mode_registry=mode_registry,
        )

    def run_agent_turn(self) -> None:
        self.workspace.ensure_active()
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

    @property
    def is_closing(self) -> bool:
        with self._close_lock:
            return self._closing or self._closed

    @property
    def hard_shutdown_requested(self) -> bool:
        return self._hard_shutdown.is_set()

    def request_hard_shutdown(self) -> None:
        """Close lifecycle services without waiting for the agent thread."""
        self._hard_shutdown.set()
        self.close(force=True)

    def close(self, *, force: bool = False) -> None:
        with self._close_lock:
            if self._closed or self._closing:
                return
            self._closing = True

        self.workspace.begin_closing()
        errors: list[BaseException] = []
        try:
            try:
                atexit.unregister(self.close)
            except Exception:
                pass
            try:
                self.interactions.close()
            except BaseException as error:
                errors.append(error)
            try:
                self.context.close(force=force)
            except BaseException as error:
                errors.append(error)
        finally:
            with self._close_lock:
                self._closing = False
                self._closed = not self.workspace.root.exists()

        if errors:
            detail = "; ".join(str(error) for error in errors)
            raise RuntimeError(
                f"Citra shutdown was incomplete; runtime={self.workspace.root}: {detail}"
            ) from errors[0]

    def __enter__(self) -> CitraApplication:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
