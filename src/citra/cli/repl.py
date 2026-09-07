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
from .input import terminal_input
from .rendering import (
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


class HardShutdownRequested(RuntimeError):
    """The second interrupt requested bounded application shutdown."""


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

    def handle_interrupt() -> None:
        """Handle handle interrupt."""
        nonlocal soft_stop_requested
        if not soft_stop_requested:
            soft_stop_requested = True
            application.request_soft_stop()
            render_notice("Stop queued. Press Ctrl+C again to exit.", level="warning")
            return

        render_notice("Hard shutdown requested.", level="warning")
        try:
            application.request_hard_shutdown()
        except Exception as error:
            raise HardShutdownRequested(str(error)) from error
        raise HardShutdownRequested()

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
                    handle_interrupt()
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
                    message="↪ steer  ",
                )
            except KeyboardInterrupt:
                handle_interrupt()
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
        print_header(application.config, application.workspace.workspace)
        while True:
            try:
                user_input = input_service.prompt(
                    "› ",
                    boxed=True,
                    footer=(
                        f"{application.config.model().id} default · "
                        f"{_compact_path(application.workspace.source_workspace)}"
                    ),
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
                    )
                else:
                    # Piped/headless invocations have no concurrent input
                    # channel, but retain the same lifecycle and agent runner.
                    application.run_agent_turn()
                console.print()
            except (KeyboardInterrupt, EOFError):
                break
            except HardShutdownRequested as error:
                if str(error):
                    logger.error(
                        "Hard shutdown failed: %s",
                        error,
                    )
                    render_notice(f"Hard shutdown error: {error}", level="error")
                break
            except Exception as error:
                logger.exception("Agent turn failed")
                render_notice(f"Error: {error}", level="error")
    finally:
        project = application.workspace.workspace
        application.close(force=application.hard_shutdown_requested)
        if project.is_dir():
            render_notice(
                f"Project checkout preserved at {project}. Review and commit it when ready.",
                level="success",
            )
