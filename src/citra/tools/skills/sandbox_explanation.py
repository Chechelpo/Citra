"""Model-facing description of the process-lifetime Agent Runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

from citra.sandbox.sandbox import SandboxMode

from .skill import Skill

if TYPE_CHECKING:
    from citra.context import ExecutionContext


@dataclass(frozen=True)
class SandboxPromptEnvironment:
    workspace: str
    source: str
    home: str
    tmp: str
    cache: str
    env: str
    runtime: str
    mode: SandboxMode
    extra_readonly_binds: tuple[str, ...]


_PROFILES: dict[SandboxMode, str] = {
    SandboxMode.FULL_ACCESS: "full-access",
    SandboxMode.ONLY_SOURCE: "only-source",
    SandboxMode.PARTIAL_SANDBOX: "partial-sandbox",
    SandboxMode.FULL_SANDBOX: "full-sandbox",
}


class SandboxEnvironment(Skill):
    """Explain Citra's filesystem, process, and authority boundaries."""

    def __init__(self) -> None:
        super().__init__(
            "sandbox-environment",
            "Explains Citra's Agent Runtime, filesystem, and configured "
            "read-only host binds, calibrated to your active Sandbox.",
            Path(),
        )

    @override
    def get_md(self, context: ExecutionContext) -> str:
        environment = _collect_environment(context)
        return _PROMPT.format(
            mode_name=environment.mode.name,
            mode_profile=_PROFILES[environment.mode],
            project_views=_project_views(environment),
            dependency_policy=_dependency_policy(environment.mode),
            host_visibility=_host_visibility(environment.mode),
            process_network=_process_network(environment.mode),
            operating_discipline=_operating_discipline(environment.mode),
            home=environment.home,
            tmp=environment.tmp,
            cache=environment.cache,
            env=environment.env,
            runtime=environment.runtime,
            extra_readonly_binds=_format_paths(environment.extra_readonly_binds),
        )


def _collect_environment(context: ExecutionContext) -> SandboxPromptEnvironment:
    workspace = context.workspace
    sandbox = context.sandbox.environment_info()
    return SandboxPromptEnvironment(
        workspace=str(workspace.workspace),
        source=str(workspace.source_workspace),
        home=str(workspace.home),
        tmp=str(workspace.tmp),
        cache=str(workspace.cache),
        env=str(workspace.env),
        runtime=str(workspace.runtime),
        mode=sandbox.mode,
        extra_readonly_binds=tuple(str(path) for path in sandbox.extra_readonly_binds),
    )


def _format_paths(paths: tuple[str, ...]) -> str:
    if not paths:
        return "- None configured."
    return "\n".join(f"- `{path}`" for path in paths)


def _project_views(environment: SandboxPromptEnvironment) -> str:
    match environment.mode:
        case SandboxMode.FULL_ACCESS:
            return f"""\
The authoritative project is the host filesystem itself. The Agent Runtime
has not isolated any boundary for this mode; ordinary tools and sandboxed
commands run with the operator's full authority on the host.

`@workspace` and `@source` both point at:

`{environment.workspace}`

Project writes take effect on the host immediately. The Commit and
materialization tools are unavailable in this mode because no staging
boundary exists to enforce."""

        case SandboxMode.ONLY_SOURCE:
            return f"""\
The authoritative project is exposed directly at:

`{environment.workspace}` (`@workspace` and `@source`)

Relative paths resolve there. It is writable, and every edit affects the
source immediately. The Commit and materialization tools are unavailable in
this mode."""

        case SandboxMode.PARTIAL_SANDBOX | SandboxMode.FULL_SANDBOX:
            return f"""\
The complete writable project snapshot is already available at:

`{environment.workspace}` (`@workspace`)

Relative paths resolve there. Use it for edits, builds, tests, and project
artifacts. No materialization step is required.

The authoritative original project is:

`{environment.source}` (`@source`)

It is independently readable and immutable to ordinary tools and sandboxed
commands. Changes in `@workspace` affect `@source` only when deliberately
staged and applied through Citra's Commit workflow. Commit verifies the source
against its startup/advanced baseline before writing and never creates a Git
commit or changes the source repository's index/history."""


def _dependency_policy(mode: SandboxMode) -> str:
    match mode:
        case SandboxMode.FULL_ACCESS:
            return (
                "A package-manager command that intentionally edits a project "
                "manifest changes the host filesystem immediately."
            )
        case SandboxMode.ONLY_SOURCE:
            return (
                "A package-manager command that intentionally edits a project "
                "manifest changes the authoritative source immediately."
            )
        case SandboxMode.PARTIAL_SANDBOX | SandboxMode.FULL_SANDBOX:
            return (
                "A package-manager command that intentionally edits a manifest "
                "does so in `@workspace`, and that edit becomes durable only "
                "through Commit."
            )


def _host_visibility(mode: SandboxMode) -> str:
    match mode:
        case SandboxMode.FULL_ACCESS:
            return (
                "The Agent Runtime does not construct a sandbox for this mode; "
                "every command runs with the operator's normal host authority. "
                "Citra's controller metadata, the Agent Runtime root, and any "
                "other host paths remain reachable but should still be treated "
                "as controller-private state."
            )
        case SandboxMode.ONLY_SOURCE:
            return (
                "The Agent Runtime does not construct a sandbox for this mode; "
                "every command runs in the operator's normal host environment, "
                "but Commit is unavailable so the only way to mutate the source "
                "is via ordinary filesystem writes (which take effect immediately)."
            )
        case SandboxMode.PARTIAL_SANDBOX:
            return (
                "Bubblewrap builds each process from the persistent Agent "
                "Runtime roots, the immutable runtime layer, the read-only "
                "`@source` mount, and the writable Citra lifecycle roots. The "
                "sandbox masks most of the host filesystem but does not enforce "
                "all full-sandbox guarantees (for example, masked host "
                "directories may remain partially visible). Visibility never "
                "implies write permission."
            )
        case SandboxMode.FULL_SANDBOX:
            return (
                "Bubblewrap builds each process from the persistent Agent "
                "Runtime roots, the immutable runtime layer, explicit read-only "
                "fallback assets, read-only `@source`, synthetic process/device "
                "views, and narrowly required controller code. The sandbox does "
                "not normally expose the complete host root."
            )


def _process_network(mode: SandboxMode) -> str:
    match mode:
        case SandboxMode.FULL_ACCESS | SandboxMode.ONLY_SOURCE:
            return (
                "Without a sandbox, every Bash/managed-subprocess invocation "
                "inherits the host's network policy. The same `network=true` "
                "reason requirement still applies to individual tool calls, but "
                "there is no bubblewrap namespace enforcing it at the process "
                "level."
            )
        case SandboxMode.PARTIAL_SANDBOX | SandboxMode.FULL_SANDBOX:
            return (
                "Every Bash/fixed-worker invocation is a fresh isolated process "
                "over the same persistent runtime filesystem. LSPs, browser "
                "workers, and managed subprocesses may remain alive for the "
                "application lifecycle and are terminated before the runtime "
                "root is removed. Network access is disabled by default; use "
                "only a tool's explicit network request/approval path when "
                "network access is actually required. Package managers receive "
                "no network bypass merely because they install into `@env`. "
                "Provider credentials, operator configuration, SSH/GPG agents, "
                "IPC sockets, and other control-plane secrets are not "
                "materialized into the Agent Runtime by default."
            )


def _operating_discipline(mode: SandboxMode) -> str:
    match mode:
        case SandboxMode.FULL_ACCESS:
            project_rules = (
                "* edit and test the authoritative project on the host directly;\n"
                "* project writes take effect on the host immediately and cannot "
                "be undone by the Agent Runtime;\n"
                "* the host filesystem, system services, and any other state "
                "are reachable by ordinary commands;"
            )
        case SandboxMode.ONLY_SOURCE:
            project_rules = (
                "* edit and test the authoritative project directly;\n"
                "* remember that project writes take effect immediately;"
            )
        case SandboxMode.PARTIAL_SANDBOX:
            project_rules = (
                "* edit and test the already populated `@workspace`;\n"
                "* inspect `@source` only when the authoritative original matters;\n"
                "* apply project changes only through Commit;\n"
                "* be aware that some host paths may still be visible; treat "
                "anything outside the Agent Runtime roots as untrusted;"
            )
        case SandboxMode.FULL_SANDBOX:
            project_rules = (
                "* edit and test the already populated `@workspace`;\n"
                "* inspect `@source` only when the authoritative original matters;\n"
                "* apply project changes only through Commit;"
            )
    return (
        f"{project_rules}\n"
        "* use `@tmp` for disposable work and `@env` for staged dependencies;\n"
        "* treat `@runtime` and fallback host assets as immutable;\n"
        "* do not infer authority from path visibility;\n"
        "* request network access only with a concrete need;\n"
        "* use the narrowest dedicated tool that provides the required capability."
    )


_PROMPT = """
# Agent Runtime sandbox

You operate inside Citra's process-lifetime disposable Agent Runtime. Treat
its filesystem, process, network, and source boundaries as invariants. The
active `SandboxMode` is `{mode_name}` (`{mode_profile}`); the rest of this
document is calibrated to that mode.

## Project views

{project_views}

## Process-lifetime filesystem

```text
@home       {home}       writable disposable home
@tmp        {tmp}        writable scratch space
@cache      {cache}      writable shared caches
@env        {env}        writable shared dependency environment
@runtime    {runtime}    immutable provisioned runtime layer
```

`@config` and `@data` are compatibility aliases beneath `@home`. HOME, XDG,
temporary, package-cache, compiler-cache, and ecosystem-specific paths are
normalized into these roots. State persists across Bash calls, linters, LSPs,
browser workers, and managed subprocesses for this Citra process, then is
deleted when Citra exits.

Install experimental dependencies into `@env`; installation alone does not
alter project manifests. {dependency_policy} Never install into or attempt to
modify `@runtime`.

Citra's controller metadata is not model-facing even if an absolute runtime
root becomes known.

## Filesystem and host visibility

{host_visibility}

Configured additional read-only binds are:

{extra_readonly_binds}

Runtime provisioning may expose other declared host assets read-only when
copying their complete semantic asset would exceed the copy budget. Such binds
are recorded by Citra and do not grant access to surrounding host state.
Visibility never implies write permission.

## Processes and network

{process_network}

## Operating discipline

{operating_discipline}
"""
