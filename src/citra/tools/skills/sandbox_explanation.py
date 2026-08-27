"""Model-facing description of the process-lifetime Agent Runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

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
    direct_source: bool
    extra_readonly_binds: tuple[str, ...]


class SandboxEnvironment(Skill):
    """Explain Citra's filesystem, process, and authority boundaries."""

    def __init__(self) -> None:
        super().__init__(
            "sandbox-environment",
            "Explains Citra's Agent Runtime, filesystem, and configured "
            "read-only host binds.",
            Path(),
        )

    @override
    def get_md(self, context: ExecutionContext) -> str:
        environment = _collect_environment(context)
        return _PROMPT.format(
            project_views=_project_views(environment),
            dependency_policy=_dependency_policy(environment.direct_source),
            source_mount_policy=(
                "the writable authoritative source"
                if environment.direct_source
                else "read-only `@source`"
            ),
            operating_discipline=_operating_discipline(
                environment.direct_source
            ),
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
        direct_source=bool(getattr(workspace, "direct_source", False)),
        extra_readonly_binds=tuple(str(path) for path in sandbox.extra_readonly_binds),
    )


def _format_paths(paths: tuple[str, ...]) -> str:
    if not paths:
        return "- None configured."
    return "\n".join(f"- `{path}`" for path in paths)


def _project_views(environment: SandboxPromptEnvironment) -> str:
    if environment.direct_source:
        return f"""\
The authoritative project is exposed directly at:

`{environment.workspace}` (`@workspace` and `@source`)

Relative paths resolve there. It is writable, and every edit affects the
source immediately. The Commit and materialization tools are unavailable in
this mode."""
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


def _dependency_policy(direct_source: bool) -> str:
    if direct_source:
        return (
            "A package-manager command that intentionally edits a project "
            "manifest changes the authoritative source immediately."
        )
    return (
        "A package-manager command that intentionally edits a manifest does "
        "so in `@workspace`, and that edit becomes durable only through Commit."
    )


def _operating_discipline(direct_source: bool) -> str:
    if direct_source:
        project_rules = (
            "* edit and test the authoritative project directly;\n"
            "* remember that project writes take effect immediately;"
        )
    else:
        project_rules = (
            "* edit and test the already populated `@workspace`;\n"
            "* inspect `@source` only when the authoritative original matters;\n"
            "* apply project changes only through Commit;"
        )
    return "\n".join(
        (
            project_rules,
            "* use `@tmp` for disposable work and `@env` for staged dependencies;",
            "* treat `@runtime` and fallback host assets as immutable;",
            "* do not infer authority from path visibility;",
            "* request network access only with a concrete need;",
            "* use the narrowest dedicated tool that provides the required capability.",
        )
    )


_PROMPT = """
# Agent Runtime sandbox

You operate inside Citra's process-lifetime disposable Agent Runtime. Treat
its filesystem, process, network, and source boundaries as invariants.

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

Bubblewrap builds each process from the persistent Agent Runtime roots, the
immutable runtime layer, explicit read-only fallback assets,
{source_mount_policy}, synthetic process/device views, and narrowly required
controller code. The sandbox does not normally expose the complete host root.

Configured additional read-only binds are:

{extra_readonly_binds}

Runtime provisioning may expose other declared host assets read-only when
copying their complete semantic asset would exceed the copy budget. Such binds
are recorded by Citra and do not grant access to surrounding host state.
Visibility never implies write permission.

## Processes and network

Every Bash/fixed worker invocation is a fresh isolated process over the same
persistent runtime filesystem. LSPs, browser workers, and managed subprocesses
may remain alive for the application lifecycle and are terminated before the
runtime root is removed.

Network access is disabled by default. Use only a tool's explicit network
request/approval path when network access is actually required. Package
managers receive no network bypass merely because they install into `@env`.

Provider credentials, operator configuration, SSH/GPG agents, IPC sockets,
and other control-plane secrets are not materialized into the Agent Runtime by
default.

## Operating discipline

{operating_discipline}
"""
