"""Citra process lifecycle and long-lived service ownership."""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from threading import Event, Lock

from .agent import AgentSession, UserInteractionBroker
from .agent.runner import AgentRunner, ApiCall
from .commands import COMMAND_REGISTRY
from .context import CitraConfig, ExecutionContext, WorkspaceContext
from .workflows import (
    SingleModeWorkflow,
    Workflow,
    WorkflowRegistry,
    WorkflowRun,
    WorkflowRuntime,
)
from citra.utils.lsp import LspConfig, LspManager
from .tools.session_memory import (
    CheckpointTool,
    RequirementTool,
    TodoTool,
    WorkingStateTool,
)
from .tools.skills.skill_registry import SkillRegistry
from .tools.subagent import SubagentSupervisor
from .utils.chat_completions_api import call_api
from .utils.terminal import RESET, YELLOW


class CitraApplication:
    """Own everything that must survive individual agent turns."""

    def __init__(
        self,
        *,
        config: CitraConfig,
        source_workspace: Path,
        api_call: ApiCall = call_api,
        workflow: Workflow | None = None,
        workflow_registry: WorkflowRegistry | None = None,
    ) -> None:
        """Initialize the instance."""
        if not isinstance(config, CitraConfig):
            raise TypeError("config must be a CitraConfig")
        if not isinstance(source_workspace, Path):
            raise TypeError("source_workspace must be a Path")
        if not callable(api_call):
            raise TypeError("api_call must be callable")
        if workflow is not None and not isinstance(workflow, Workflow):
            raise TypeError("workflow must be a Workflow")
        if (
            workflow_registry is not None
            and not isinstance(workflow_registry, WorkflowRegistry)
        ):
            raise TypeError("workflow_registry must be a WorkflowRegistry")
        self.config = config
        self._api_call = api_call
        self.source_workspace = source_workspace.resolve()
        self.workflow_registry = workflow_registry or WorkflowRegistry(
            config_path=os.environ.get("CITRA_CONFIG_PATH"),
        )
        self.workflow = workflow or self.workflow_registry.active_workflow
        self.workflow.validate()
        initial_workflow = self.workflow.initial_workflow
        initial_workflow.validate()
        self.sandbox_config = self.workflow.sandbox_config
        self.session = AgentSession(
            memory_enabled=config.memory.enabled,
        )
        self._close_lock = Lock()
        self._closing = False
        self._closed = False
        self._hard_shutdown = Event()
        self.workspace = WorkspaceContext.create(
            workspace=self.source_workspace,
            temporary_workspace=config.sandbox_policy.workspace_parent,
            browser_path=config.browser.browsers_path,
            sandbox_mode=self.sandbox_config.mode,
        )
        try:
            sandbox_policy = config.sandbox_policy.clone()
            sandbox_policy.apply_workflow_config(self.sandbox_config)
            sandbox_policy.add_readonly_bind(
                self.workspace.runtime,
                Path("/runtime"),
            )
            sandbox_policy.add_runtime_mounts(
                self.workspace.runtime_readonly_binds
            )
            for root in self.workspace.writable_roots:
                if root != self.workspace.workspace:
                    sandbox_policy.add_writable_bind(root)
            self.workflow_runtime = WorkflowRuntime.provision(
                workflow=self.workflow,
                workspace=self.workspace,
                policy=sandbox_policy,
            )
            self.skills = SkillRegistry(
                agent_session=self.session,
                memory_enabled=config.memory.enabled,
                workflow=initial_workflow,
                skills_root=self._skills_root(),
            )
            self.interactions = UserInteractionBroker()
            sandbox = self.workflow_runtime.sandbox
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
            self.subagent_supervisor = SubagentSupervisor(
                parent_workspace=self.workspace,
                parent_root=self.workspace.root,
                api_call=api_call,
            )
            self.context = ExecutionContext(
                self.workspace,
                skills=self.skills,
                config=config,
                workflow_runtime=self.workflow_runtime,
                lsp_manager=self.lsp_manager,
                user_interactions=self.interactions,
                subagents=self.subagent_supervisor,
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
        workflow: Workflow | None = None,
        workflow_registry: WorkflowRegistry | None = None,
    ) -> CitraApplication:
        """Handle create."""
        config = config or CitraConfig.load()
        source = Path(
            source_workspace
            or os.getcwd()
        ).expanduser().resolve()
        return cls(
            config=config,
            source_workspace=source,
            api_call=api_call,
            workflow=workflow,
            workflow_registry=workflow_registry,
        )

    @property
    def active_workflow(self) -> SingleModeWorkflow:
        """Workflow currently executing, including a serial workflow phase."""
        return self.context.workflow

    @property
    def workflow_run(self) -> WorkflowRun | None:
        """Handle workflow run."""
        return self.workflow_runtime.active_run

    def prepare_user_turn(self, content: str) -> None:
        """Seed either the persistent agent or a new isolated serial run."""
        content = content.strip()
        if not content:
            raise ValueError("User task cannot be empty")
        if not self.workflow.is_serial:
            self.session.add_user_message(content)
            return

        self.workflow_runtime.start_run(content)
        try:
            self._activate_serial_step()
        except Exception:
            self.workflow_runtime.cancel_run()
            raise

    def run_agent_turn(self, user_input: str | None = None) -> None:
        """Execute the run agent turn operation."""
        self.workspace.ensure_active()
        if user_input is not None:
            self.prepare_user_turn(user_input)
        if not self.workflow.is_serial:
            self.runner.run_turn()
            return

        active_run = self.workflow_runtime.active_run
        if active_run is None or active_run.is_terminal:
            inferred = self._latest_user_message()
            if inferred is None:
                raise RuntimeError(
                    "Serial workflow requires prepare_user_turn(task) before execution"
                )
            self.prepare_user_turn(inferred)

        run = self.workflow_runtime.active_run
        assert run is not None
        while not run.is_terminal:
            step = run.begin_step()
            checkpoint_revision = self._checkpoint_revision()
            print(f"\n{YELLOW}⏺ Workflow phase: {step.step_id}{RESET}")
            self.runner.run_turn()
            if run.is_terminal:
                break
            handoff_error = self._submit_serial_handoff(
                run,
                checkpoint_revision=checkpoint_revision,
            )
            if handoff_error is not None:
                checkpoint_name = (
                    CheckpointTool.resolve_definition_for_context(
                        self.context
                    ).function.name
                )
                self.session.add_user_message(
                    "The workflow controller rejected phase completion: "
                    f"{handoff_error}\n\n"
                    "Reconcile the durable memory tools, set an appropriate "
                    f"transition with `{checkpoint_name}`, then return a "
                    "self-contained final assistant message for the next role."
                )
                self.runner.run_turn()
            if run.is_terminal:
                break
            if handoff_error is not None:
                handoff_error = self._submit_serial_handoff(
                    run,
                    checkpoint_revision=checkpoint_revision,
                )
            if handoff_error is not None:
                run.cancel()
                raise RuntimeError(
                    f"Workflow phase {step.step_id!r} could not hand off: "
                    f"{handoff_error}"
                )
            run.advance()
            if not run.is_terminal:
                try:
                    self._activate_serial_step()
                except Exception:
                    run.cancel()
                    raise

    def _activate_serial_step(self) -> None:
        """Handle activate serial step."""
        run = self.workflow_runtime.active_run
        if run is None or run.is_terminal:
            raise RuntimeError("Cannot activate a terminal workflow run")
        workflow = run.current_step.workflow
        # Conversation history and reasoning are role-local. Structured memory
        # is task-scoped and deliberately shared by the workflow run.
        session = AgentSession(
            memory=run.memory,
            memory_enabled=True,
        )
        session.add_user_message(run.phase_input())
        skills = SkillRegistry(
            agent_session=session,
            memory_enabled=True,
            workflow=workflow,
            skills_root=self._skills_root(),
        )
        self.session = session
        self.skills = skills
        self.context.activate_workflow(
            workflow,
            skills=skills,
        )
        self.runner = AgentRunner(
            self.context,
            session,
            api_call=self._api_call,
        )

    def _checkpoint_revision(self) -> int:
        """Handle checkpoint revision."""
        checkpoint = self.session.memory.get(CheckpointTool.TOOL_ID)
        if not isinstance(checkpoint, CheckpointTool):
            return 0
        return checkpoint.revision

    def _submit_serial_handoff(
        self,
        run: WorkflowRun,
        *,
        checkpoint_revision: int,
    ) -> str | None:
        """Handle submit serial handoff."""
        checkpoint_tool = self.session.memory.get(CheckpointTool.TOOL_ID)
        if not isinstance(checkpoint_tool, CheckpointTool):
            return "the checkpoint memory tool was not used"
        if checkpoint_tool.revision <= checkpoint_revision:
            return "the phase did not update its checkpoint"

        checkpoint = checkpoint_tool.current_checkpoint
        if checkpoint is None:
            return "the phase cleared its checkpoint instead of setting one"
        next_step = (checkpoint.next_step or "").strip()
        if not next_step:
            return "the checkpoint does not declare next_step"

        step = run.current_step
        if next_step not in step.allowed_next:
            allowed = ", ".join(step.allowed_next)
            return (
                f"step {step.step_id!r} cannot transition to "
                f"{next_step!r}; allowed: {allowed}"
            )
        if next_step == "complete":
            completion_error = self._memory_completion_error()
            if completion_error is not None:
                return completion_error

        message = self._latest_assistant_handoff()
        if message is None:
            return "the phase did not return a final assistant message"

        try:
            run.submit_handoff(summary=message, next_step=next_step)
        except (RuntimeError, ValueError) as error:
            return str(error)
        return None

    def _memory_completion_error(self) -> str | None:
        """Handle memory completion error."""
        requirement = self.session.memory.get(RequirementTool.TOOL_ID)
        if (
            isinstance(requirement, RequirementTool)
            and requirement.has_unsatisfied_requirements()
        ):
            return "workflow completion requires all valid requirements satisfied"
        todo = self.session.memory.get(TodoTool.TOOL_ID)
        if isinstance(todo, TodoTool) and todo.has_outstanding_todos():
            return "workflow completion requires all valid TODOs to be complete"
        working = self.session.memory.get(WorkingStateTool.TOOL_ID)
        if isinstance(working, WorkingStateTool) and working.get_extracts():
            return (
                "workflow completion requires working states to be resolved "
                "or discarded"
            )
        return None

    def _latest_assistant_handoff(self) -> str | None:
        """Handle latest assistant handoff."""
        for message in reversed(self.session.get_messages()):
            if message.get("role") != "assistant":
                continue
            if message.get("tool_calls"):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return None

    def _skills_root(self) -> Path:
        """Handle skills root."""
        return (
            Path(
                os.environ.get(
                    "CITRA_INSTALL_ROOT",
                    str(Path(__file__).resolve().parents[2]),
                )
            )
            / "skills"
        )

    def _latest_user_message(self) -> str | None:
        """Handle latest user message."""
        for message in reversed(self.session.get_messages()):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    normalized = content.strip()
                    if normalized.startswith("# Workflow task"):
                        continue
                    if normalized.startswith(
                        "The workflow controller rejected phase completion:"
                    ):
                        continue
                    return normalized
        return None

    def handle_command(self, user_input: str) -> bool:
        """Handle handle command."""
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
        """Return whether is closing."""
        with self._close_lock:
            return self._closing or self._closed

    @property
    def hard_shutdown_requested(self) -> bool:
        """Handle hard shutdown requested."""
        return self._hard_shutdown.is_set()

    def request_hard_shutdown(self) -> None:
        """Close lifecycle services without waiting for the agent thread."""
        self._hard_shutdown.set()
        self.close(force=True)

    def request_soft_stop(self) -> None:
        """Stop a serial macro after the active model turn reaches safety."""
        self.workflow_runtime.cancel_run()
        self.session.queue_steering(
            "Stop the current work safely and return control to the user."
        )

    def close(self, *, force: bool = False) -> None:
        """Handle close."""
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
            try:
                self.workspace.cleanup(
                    force=force,
                    preserve_workspace=True,
                )
            except BaseException as error:
                errors.append(error)
        finally:
            with self._close_lock:
                self._closing = False
                self._closed = self.workspace.lifecycle_state.value == "closed"

        if errors:
            detail = "; ".join(str(error) for error in errors)
            raise RuntimeError(
                "Citra shutdown was incomplete; "
                f"runtime={self.workspace.root}: {detail}"
            ) from errors[0]

    def __enter__(self) -> CitraApplication:
        """Enter the managed lifecycle."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit the managed lifecycle."""
        self.close()
