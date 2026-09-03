from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

from .skill import Skill

if TYPE_CHECKING:
    from citra.context import ExecutionContext


class WebAppDebugging(Skill):
    """
    Teach the agent how to debug a web application using Citra's
    subprocess and browser tools.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__(
            "web-app-debugging",
            "Describes how to run, inspect, reproduce, diagnose, and verify "
            "web applications using Citra's subprocess and browser tools.",
            Path(),
        )

    @override
    def get_md(
        self,
        context: ExecutionContext,
    ) -> str:
        if not context.config.memory.enabled:
            return _PROMPT.replace(_MEMORY_INSTRUCTION, "")
        return _PROMPT


_MEMORY_INSTRUCTION = """\
Record the important conditions of the reproduction in session memory when the
task is non-trivial and those conditions will matter during later verification.

"""

_PROMPT: str = """
# Debugging web applications

Use this workflow when investigating or verifying behavior in a web
application.

Prefer reproducing actual runtime behavior over reasoning from source code
alone.

The main tools for this workflow are:

* `subprocess` for lifecycle-scoped development servers and other persistent
  processes;
* `browser` for interacting with the running application in headless Chromium;
* source inspection, LSP, text search, and diagnostics for locating the implementation
  responsible for observed behavior.

## General workflow

For a non-trivial web application bug, generally work through this cycle:

1. Understand how the application is normally started.
2. Start the relevant development server with `subprocess`.
3. Inspect server startup output and resolve startup failures before opening
   the browser.
4. Open the application with `browser`.
5. Reproduce the reported behavior through normal user interactions.
6. Inspect browser state, console output, page errors, and server output.
7. Locate the responsible implementation using semantic and textual code tools.
8. Make the smallest coherent fix.
9. Re-run focused diagnostics or tests.
10. Exercise the same browser workflow again and verify that the original
    failure no longer occurs.
11. Check for newly introduced browser console errors, page errors, or server
    failures.
12. Stop processes that are no longer needed.

Do not stop after identifying a likely source-code cause when the application
can reasonably be run and the behavior can be verified directly.

## Starting the application

Use `subprocess` for development servers rather than launching a long-running
server with a one-shot Bash command.

Before starting a server:

* inspect the project's package scripts, documentation, configuration, or
  existing development conventions;
* prefer the project's normal development command;
* use the appropriate project directory as the subprocess working directory;
* reuse existing dependencies and lockfiles instead of installing packages
  unnecessarily.

When starting a server, `sleep_after` together with `poll_after=true` is useful
for obtaining initial startup output in the same operation.

After starting it, inspect the returned output before assuming the server is
ready.

A process that was started successfully may continue running across later
agent turns. Keep track of its process ID.

Use `subprocess` `poll` to inspect subsequent output while debugging.

Do not repeatedly start duplicate development servers when an existing process
is still suitable. Use `subprocess` `list` when the current process state is
uncertain.

If a server needs network access, request it only when genuinely required and
provide the concrete reason.

## Distinguish startup failures from application failures

Before debugging the browser:

* confirm that the development server remains alive;
* look for compilation errors, missing modules, configuration failures, port
  conflicts, stack traces, and other startup diagnostics;
* confirm the expected listening address or port from actual server output
  when possible.

If the application cannot start, debug that failure first.

Do not interpret a browser connection failure as a frontend bug until the
server itself has been checked.

## Opening the application

Use `browser` to test the actual web application.

Browser navigation requires an absolute HTTP or HTTPS URL and a concrete
network reason.

After opening the application, obtain a `snapshot` before interacting with the
page.

Snapshots provide stable element references. Prefer those references for
clicking, filling, selecting, checking, hovering, and other interactions.

Do not guess selectors when a snapshot already provides a stable reference.

Take another snapshot after navigation or substantial UI changes when earlier
element references may no longer describe the current page state.

## Reproduce before changing code

When debugging an observed application bug, reproduce it before editing when
reasonably possible.

Follow the same user-visible sequence that causes the problem:

* navigate to the relevant page;
* enter representative inputs;
* click the relevant controls;
* wait for asynchronous UI changes when required;
* observe the resulting state.

Record the important conditions of the reproduction in session memory when the
task is non-trivial and those conditions will matter during later verification.

Do not modify code merely because a source path looks suspicious. Establish
the runtime symptom first when the environment permits it.

## Inspect browser evidence

Use browser evidence aggressively when diagnosing frontend behavior.

### Snapshot

Use `snapshot` to inspect:

* visible page structure;
* accessible roles and labels;
* current controls;
* stable interaction references;
* whether expected UI elements appeared or disappeared.

Use snapshots as the normal basis for interaction.

### Console

Use `console` when JavaScript runtime warnings, logging, failed requests, or
framework diagnostics may explain the behavior.

Inspect console output after reproducing the issue and again after applying the
fix.

Do not assume that a visually correct page is healthy if new console errors
are present.

### Page errors

Use `errors` to inspect uncaught page exceptions and other browser-level
runtime failures.

When a user-visible failure coincides with an exception, use the exception
message and stack information to guide source inspection.

### Screenshots

Use `screenshot` when visual state matters or when the rendered result is hard
to understand from the accessibility snapshot alone.

Screenshots are supporting evidence. Prefer snapshots for locating and
interacting with elements because their references are more stable.

Store debugging screenshots under `@tmp` unless they are intentionally part of
the project.

## Inspect server evidence while reproducing

Frontend behavior may originate in backend code, server rendering, API
handlers, proxies, bundlers, or development middleware.

After reproducing a failure, poll the development server when server-side
output could be relevant.

Correlate browser observations with:

* backend exceptions;
* request logs;
* framework warnings;
* build or hot-reload errors;
* serialization failures;
* failed API handlers.

Do not debug the frontend in isolation when server output provides evidence
that the failure is server-side.

## Locate the responsible code

After obtaining runtime evidence, use code intelligence to trace it back to the
implementation.

Prefer LSP when symbol identity matters:

* definitions;
* references;
* implementations;
* inferred types;
* diagnostics.

Prefer focused text search for:

* literal error messages;
* route strings;
* CSS classes;
* configuration;
* log text;
* HTTP paths;
* user-visible text.

Use browser evidence to narrow code investigation instead of manually scanning
large portions of the project.

## Iterative debugging

Web debugging is usually iterative.

A productive loop is:

1. reproduce;
2. capture browser and server evidence;
3. form a specific hypothesis;
4. inspect the relevant source;
5. make a focused change;
6. run diagnostics;
7. let the development server rebuild or restart as appropriate;
8. reload or repeat the browser interaction;
9. compare the new behavior with the original failure.

Do not accumulate several speculative fixes before checking whether the first
one changed the observed behavior.

If the development environment supports hot reload, reuse the running server
unless evidence shows that it needs to be restarted.

## Browser reloads and state

Use `reload` when testing a source change that should affect the current page.

Be aware that application state may persist through:

* cookies;
* local storage;
* session storage;
* backend sessions;
* in-memory development server state.

If the behavior depends on clean state, establish that explicitly rather than
assuming reload produces a fresh session.

Use normal application interactions to establish state whenever possible.

## Unsafe browser actions

Prefer normal browser interactions and snapshots.

Browser actions such as arbitrary JavaScript evaluation or file upload may be
disabled by configuration and may require additional authorization.

Do not depend on unsafe browser actions when the same behavior can be tested
through normal user-visible interactions.

If an unsafe action is genuinely necessary, use it narrowly and provide the
required concrete reason.

## Network requests and permissions

Browser navigation and networked subprocesses can require authorization.

Request network access only when needed for the debugging workflow.

A local development application may still require browser navigation
authorization because the browser treats opening an HTTP origin as network
access.

Do not repeatedly request access to unrelated origins.

## Verification after a fix

A web application fix is not verified merely because the edited source looks
correct.

When practical, verify all of the following:

1. relevant source diagnostics are clean;
2. the development server builds or reloads successfully;
3. the original reproduction sequence now behaves correctly;
4. expected UI state is visible in a fresh browser snapshot;
5. browser page errors do not show a new failure;
6. browser console output does not show a newly introduced error;
7. relevant server output does not contain a new exception or build failure.

If the task concerns visual behavior, use a screenshot when it materially helps
confirm the result.

If verification cannot be performed, state exactly which part could not be
verified.

## Process cleanup

Persistent subprocesses belong to the Citra lifecycle and may intentionally
remain alive while debugging continues.

Stop servers that are no longer needed, especially when:

* the debugging task is complete;
* a replacement server must use the same port;
* the process is repeatedly failing;
* keeping it alive serves no later work.

Do not kill a useful running development server merely because one individual
debugging step is complete.

## Debugging discipline

Keep evidence and hypotheses separate.

Do not invent:

* browser console output;
* server logs;
* rendered state;
* network failures;
* stack traces;
* successful interactions.

Observe them using the available tools.

Prefer actual runtime reproduction and verification over speculative source-only
debugging whenever the application can reasonably be executed.
"""
