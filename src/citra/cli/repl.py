"""Interactive REPL, including foreground steering of background turns."""

from __future__ import annotations

import os
import sys
import time
from threading import Event, Thread
from typing import Any

from prompt_toolkit.patch_stdout import patch_stdout

from ..agent.interactions import UserPromptRequest
from ..agent.runner import ApiCall
from ..application import CitraApplication
from ..modes import Mode, ModeRegistry
from ..utils.chat_completions_api import call_api
from ..utils.terminal import (
    BLUE,
    BOLD,
    CYAN,
    DIM,
    GREEN,
    RED,
    RESET,
    YELLOW,
    separator,
    terminal_bell,
)
from ..utils.terminal_input import terminal_input
from .rendering import print_header


class HardShutdownRequested(RuntimeError):
    """The second interrupt requested bounded application shutdown."""


def select_startup_mode(
    registry: ModeRegistry,
    *,
    input_service: Any = terminal_input,
) -> Mode:
    """Select a mode before any runtime or sandbox service is constructed."""
    print(f"{BOLD}Select a Citra mode:{RESET}")
    for index, mode in enumerate(registry.modes, 1):
        description = f" — {mode.description}" if mode.description else ""
        default = (
            f" {GREEN}(default){RESET}"
            if mode is registry.default_mode
            else ""
        )
        print(f"  {DIM}{index}.{RESET} {mode.name}{description}{default}")

    while True:
        selection = input_service.prompt(
            f"{BOLD}{BLUE}mode❯{RESET} "
        ).strip()
        try:
            return registry.select(selection)
        except (KeyError, ValueError) as error:
            print(f"{RED}⏺ {error}{RESET}")


def is_command(user_input: str) -> bool:
    return user_input.startswith("/")


def _answer_model_prompt(
    application: CitraApplication,
    request: UserPromptRequest,
    *,
    input_service: Any = terminal_input,
) -> None:
    if application.config.notifications.prompt_bell:
        terminal_bell()
    print()
    print(f"{CYAN}⏺{RESET} {BOLD}{request.question}{RESET}")
    if request.options:
        for index, option in enumerate(request.options, 1):
            print(f"  {DIM}{index}.{RESET} {option}")
        print(f"\n{DIM}Type a number or a free-form answer.{RESET}")
    else:
        print(f"{DIM}(open-ended question){RESET}")
    remaining = max(0.01, request.timeout - (time.monotonic() - request.created_at))
    answer = input_service.prompt_with_idle_timeout(
        timeout=remaining,
        message=f"{BOLD}{BLUE}❯{RESET} ",
    )
    if answer is None:
        print(
            f"{YELLOW}⏺ (no response within {request.timeout:g}s — "
            f"proceeding as user-unavailable){RESET}"
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
        try:
            application.run_agent_turn()
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    thread = Thread(target=worker, name="citra-agent", daemon=True)
    thread.start()

    input_closed = False
    soft_stop_requested = False

    def handle_interrupt() -> None:
        nonlocal soft_stop_requested
        if not soft_stop_requested:
            soft_stop_requested = True
            application.session.queue_steering(
                "Stop the current work safely and return control to the user."
            )
            print(f"{YELLOW}⏺ Stop instruction queued. Press Ctrl+C again to exit.{RESET}")
            return

        print(f"{YELLOW}⏺ Hard shutdown requested.{RESET}")
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
                    message=f"{BOLD}{BLUE}↪ steer{RESET} {DIM}(Enter to send){RESET} ",
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
            if steering is not None and application.session.queue_steering(steering):
                print(f"{GREEN}⏺ Steering queued.{RESET}")

    thread.join()
    if errors:
        raise errors[0]


def main(
    *,
    api_call: ApiCall = call_api,
    input_service: Any = terminal_input,
    interactive_mode_selection: bool | None = None,
) -> None:
    mode_registry = ModeRegistry(
        config_path=os.environ.get("CITRA_CONFIG_PATH"),
    )
    should_prompt = (
        sys.stdin.isatty()
        if interactive_mode_selection is None
        else interactive_mode_selection
    )
    mode = (
        select_startup_mode(
            mode_registry,
            input_service=input_service,
        )
        if should_prompt
        else mode_registry.select()
    )
    application = CitraApplication.create(
        api_call=api_call,
        mode=mode,
        mode_registry=mode_registry,
    )
    try:
        print_header(application.config, application.source_workspace)
        while True:
            try:
                print(separator())
                user_input = input_service.prompt(f"{BOLD}{BLUE}❯{RESET} ").strip()
                print(separator())
                if not user_input:
                    continue
                if is_command(user_input):
                    if not application.handle_command(user_input):
                        break
                    continue
                application.session.add_user_message(user_input)
                if sys.stdin.isatty():
                    run_turn_with_steering(
                        application,
                        input_service=input_service,
                    )
                else:
                    # Piped/headless invocations have no concurrent input
                    # channel, but retain the same lifecycle and agent runner.
                    application.run_agent_turn()
                print()
            except (KeyboardInterrupt, EOFError):
                break
            except HardShutdownRequested as error:
                if str(error):
                    print(f"{RED}⏺ Hard shutdown error: {error}{RESET}")
                break
            except Exception as error:
                print(f"{RED}⏺ Error: {error}{RESET}")
    finally:
        application.close(force=application.hard_shutdown_requested)
