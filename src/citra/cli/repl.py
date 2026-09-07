"""Interactive REPL, including foreground steering of background turns."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from threading import Event, Thread
from typing import Any

from prompt_toolkit.patch_stdout import patch_stdout

from ..agent.interactions import UserPromptRequest
from ..agent.runner import ApiCall
from ..application import CitraApplication
from ..utils.chat_completions_api import call_api
from ..utils.process_logging import process_log
from ..utils.terminal import terminal_bell
from ..workflows import Workflow, WorkflowRegistry
from .input import terminal_input, terminal_ui_state
from .rendering import (
    CliSessionLayout,
    SessionHeader,
    SessionFooter,
    console,
    print_header,
    render_notice,
    render_question,
    render_workflow_picker,
)

logger = logging.getLogger(__name__)


def _compact_path(path: str | Path) -> str:
    """Render home-relative paths with Codex-style ``~/`` notation."""
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(Path.home().resolve())
    except ValueError:
        return str(resolved)
    return "~" if not relative.parts else f"~/{relative.as_posix()}"


def _session_footer(application: CitraApplication) -> SessionFooter:
    """Build typed environment details for the persistent prompt footer."""
    model = application.config.model()
    model_selection = (
        model.name
        if model.name == model.id
        else f"{model.name} ({model.id})"
    )
    source = _compact_path(application.workspace.source_workspace)
    return SessionFooter(
        model_selection=model_selection,
        source_directory=source,
        workspace_directory=_compact_path(application.workspace.workspace),
        process_name=application.workspace.runtime_id,
    )


def _session_header(application: CitraApplication) -> SessionHeader:
    """Build immutable workflow and sandbox information for the session header."""
    model = application.config.model()
    return SessionHeader(
        workflow=application.workflow.name,
        connection=f"{model.host} : {model.id}",
        sandbox_mode=application.sandbox_config.mode,
        workspace=application.workspace.workspace,
    )


def _session_layout(application: CitraApplication) -> CliSessionLayout:
    """Build the static typed layout used by the current prompt and header."""
    return CliSessionLayout(
        header=_session_header(application),
        footer=_session_footer(application),
    )


def _initial_prompt_text() -> str:
    """Return the shared composer's first-message backdrop text."""
    return "/help or type your first message"


def _steering_prompt_text() -> str:
    """Return the shared composer's concise steering backdrop text."""
    return "enter steering"


class HardShutdownRequested(RuntimeError):
    """Legacy compatibility exception; hard stops no longer close the app."""


def select_startup_workflow(
    registry: WorkflowRegistry,
    *,
    input_service: Any = terminal_input,
) -> Workflow:
    """Select the sandbox-owning workflow before provisioning starts."""
    render_workflow_picker(
        [
            (
                workflow.name,
                workflow.description or "",
                workflow is registry.default_workflow,
            )
            for workflow in registry.workflows
        ]
    )

    while True:
        selection = input_service.prompt("workflow › ").strip()
        try:
            return registry.select(selection)
        except (KeyError, ValueError) as error:
            render_notice(str(error), level="error")


def is_command(user_input: str) -> bool:
    """Return whether is command."""
    return user_input.startswith("/")


def _answer_model_prompt(
    application: CitraApplication,
    request: UserPromptRequest,
    *,
    input_service: Any = terminal_input,
) -> None:
    """Handle answer model prompt."""
    if application.config.notifications.prompt_bell:
        terminal_bell()
    console.print()
    render_question(request.question, request.options)
    application.interactions.record_activity(request.id)
    answer = input_service.prompt_with_idle_timeout(
        timeout=request.timeout,
        message="› ",
        on_activity=lambda: application.interactions.record_activity(request.id),
    )
    if answer is None:
        render_notice(
            f"No response within {request.timeout:g}s; continuing without user input.",
            level="warning",
        )
    application.interactions.respond(request.id, answer)


def run_turn_with_steering(
    application: CitraApplication,
    *,
    input_service: Any = terminal_input,
    status_footer: str = "",
) -> None:
    """Run one agent turn while the terminal remains a steering channel."""
    done = Event()
    errors: list[BaseException] = []

    def worker() -> None:
        """Handle worker."""
        try:
            application.run_agent_turn()
        except BaseException as error:  # noqa: BLE001 - re-raised on foreground thread
            errors.append(error)
        finally:
            done.set()

    thread = Thread(target=worker, name="citra-agent", daemon=True)
    thread.start()

    input_closed = False
    soft_stop_requested = False

    def handle_interrupt() -> bool:
        """Handle handle interrupt."""
        nonlocal soft_stop_requested
        if not soft_stop_requested:
            soft_stop_requested = True
            application.request_soft_stop()
            render_notice(
                "Stop queued. Press Ctrl+C again to stop agent calls now.",
                level="warning",
            )
            return False

        application.request_hard_shutdown()
        render_notice("Agent calls stopped.", level="warning")
        return True

    with patch_stdout(raw=True):
        while not done.is_set():
            request = application.interactions.take()
            if request is not None:
                if input_closed:
                    application.interactions.respond(request.id, None)
                    continue

                try:
                    _answer_model_prompt(
                        application,
                        request,
                        input_service=input_service,
                    )
                except KeyboardInterrupt:
                    application.interactions.respond(request.id, None)
                    if handle_interrupt():
                        return
                except EOFError:
                    application.interactions.respond(request.id, None)
                    application.session.queue_steering(
                        "Finish the current safe boundary and stop; "
                        "the input stream closed."
                    )
                    input_closed = True
                continue

            if input_closed:
                done.wait(0.1)
                continue

            try:
                steering = input_service.prompt_until(
                    lambda: done.is_set() or application.interactions.has_pending(),
                    message=_steering_prompt_text(),
                    boxed=True,
                    footer=status_footer,
                )
            except KeyboardInterrupt:
                if handle_interrupt():
                    return
                continue
            except EOFError:
                application.session.queue_steering(
                    "Finish the current safe boundary and stop; the input stream closed."
                )
                input_closed = True
                continue
            if steering is not None and is_command(steering):
                command_parts = steering[1:].split(None, 1)
                command_id = command_parts[0] if command_parts else ""
                if command_id in {"agent", "memory", "workflow"}:
                    application.handle_command(steering)
                else:
                    render_notice(
                        "Only /agent, /memory, and /workflow are available during a turn.",
                        level="warning",
                    )
                continue
            if steering is not None and application.session.queue_steering(steering):
                render_notice("Steering queued.", level="success")

    thread.join()
    if errors:
        raise errors[0]


def main(
    *,
    api_call: ApiCall = call_api,
    input_service: Any = terminal_input,
    interactive_workflow_selection: bool | None = None,
) -> None:
    """Handle main."""
    workflow_registry = WorkflowRegistry(
        config_path=os.environ.get("CITRA_CONFIG_PATH"),
    )
    should_prompt = (
        sys.stdin.isatty()
        if interactive_workflow_selection is None
        else interactive_workflow_selection
    )
    workflow = (
        select_startup_workflow(
            workflow_registry,
            input_service=input_service,
        )
        if should_prompt
        else workflow_registry.select()
    )
    application = CitraApplication.create(
        api_call=api_call,
        workflow=workflow,
        workflow_registry=workflow_registry,
    )
    with process_log(application.workspace.logs):
        _run_application(application, input_service=input_service)


def _run_application(
    application: CitraApplication,
    *,
    input_service: Any,
) -> None:
    """Run and shut down an application under its process log handler."""
    try:
        terminal_ui_state.reset()
        initial_layout = _session_layout(application)
        print_header(initial_layout.header)
        while True:
            try:
                layout = _session_layout(application)
                user_input = input_service.prompt(
                    _initial_prompt_text(),
                    boxed=True,
                    footer=layout.footer.render(),
                ).strip()
                if not user_input:
                    continue
                if is_command(user_input):
                    if not application.handle_command(user_input):
                        break
                    continue
                application.prepare_user_turn(user_input)
                if sys.stdin.isatty():
                    run_turn_with_steering(
                        application,
                        input_service=input_service,
                        status_footer=layout.footer.render(),
                    )
                else:
                    # Piped/headless invocations have no concurrent input
                    # channel, but retain the same lifecycle and agent runner.
                    application.run_agent_turn()
                console.print()
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as error:
                logger.exception("Agent turn failed")
                render_notice(f"Error: {error}", level="error")
    finally:
        project = application.workspace.workspace
        application.close()
        if project.is_dir():
            render_notice(
                f"Project checkout preserved at {project}. Review and commit it when ready.",
                level="success",
            )
