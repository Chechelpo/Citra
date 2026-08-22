from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import override

from .skill import Skill
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citra.context import ExecutionContext
    from citra.agent import AgentSession

@dataclass(frozen=True)
class SandboxPromptEnvironment:
    workspace: str
    source: str
    home: str
    tmp: str
    cache: str
    config: str
    data: str
    runtime: str
    extra_readonly_binds: tuple[str, ...]


class SandboxEnvironment(Skill):
    """
    Explain Citra's execution sandbox and current lifecycle filesystem.
    """

    def __init__(self) -> None:
        super().__init__(
            "sandbox-environment",
            "Explains Citra's execution sandbox, lifecycle filesystem, "
            "and configured read-only host binds.",
            Path(),
        )

    @override
    def get_md(
        self,
        context: ExecutionContext,
    ) -> str:
        environment = _collect_environment(
            context
        )

        return _PROMPT.format(
            workspace=environment.workspace,
            source=environment.source,
            home=environment.home,
            tmp=environment.tmp,
            cache=environment.cache,
            config=environment.config,
            data=environment.data,
            runtime=environment.runtime,
            extra_readonly_binds=_format_paths(
                environment.extra_readonly_binds
            ),
        )


def _collect_environment(
    context: ExecutionContext,
) -> SandboxPromptEnvironment:
    workspace = context.workspace
    sandbox = context.sandbox.environment_info()

    return SandboxPromptEnvironment(
        workspace=str(workspace.workspace),
        source=str(workspace.source_workspace),
        home=str(workspace.home),
        tmp=str(workspace.tmp),
        cache=str(workspace.cache),
        config=str(workspace.config),
        data=str(workspace.data),
        runtime=str(workspace.runtime),
        extra_readonly_binds=tuple(
            str(path)
            for path in sandbox.extra_readonly_binds
        ),
    )


def _format_paths(
    paths: tuple[str, ...],
) -> str:
    if not paths:
        return "- None configured."

    return "\n".join(
        f"- `{path}`"
        for path in paths
    )

_PROMPT = """
# Execution sandbox

You operate inside Citra's lifecycle-scoped execution sandbox.

Treat its filesystem, process, and network boundaries as execution invariants.
Do not attempt to bypass them. A failure caused by one of these boundaries
usually means the operation belongs somewhere else or requires a different
tool, not that the boundary should be circumvented.

## Project workspaces

Citra separates the user's permanent project from your writable working copy.

### Source workspace

The original project is:

`{source}`

It is authoritative and read-only to ordinary filesystem and command tools.

Inside sandboxed commands it is also available through `@source` from the
agent workspace.

Use the source workspace for inspection. Do not attempt to modify it directly.

Existing source files must be materialized into the agent workspace before
they are edited. Changes made in the agent workspace do not modify the source
until they are deliberately applied through Citra's Commit workflow.

### Agent workspace

Your writable project workspace is:

`{workspace}`

Relative project paths resolve here, and sandboxed commands start here unless
another working directory is explicitly selected.

Use this workspace for project files that are being modified, built, tested,
or prepared for application back to the source.

Do not use it as a general scratch directory.

## Lifecycle filesystem

Citra also provides disposable process-lifetime directories:

```text
@home       {home}
@tmp        {tmp}
@cache      {cache}
@config     {config}
@data       {data}
@runtime    {runtime}
````

These directories are writable and persist across agent turns for the current
Citra process. They are removed with the lifecycle.

Use:

* `@tmp` for experiments, generated scratch data, extracted archives, temporary
  builds, exploratory clones, and other work that should not become part of the
  project;
* `@home` for disposable user-home state;
* `@cache`, `@config`, `@data`, and `@runtime` for tool state that naturally
  belongs in those locations.

The sandbox redirects conventional mutable environment locations such as
`HOME`, `TMPDIR`, and XDG directories into this lifecycle filesystem.

Citra's internal lifecycle root and trusted control-plane state are not
model-facing filesystem locations. Do not attempt to address or modify them.

## Path aliases

Model-facing filesystem tools understand:

```text
@source
@workspace
@home
@tmp
@cache
@config
@data
@runtime
```

Relative paths resolve from `@workspace`.

`~/...` resolves beneath `@home`.

Use ordinary relative paths for normal materialized project files.

## Filesystem visibility

Citra's filesystem inspection tools and sandboxed commands operate inside the
same Bubblewrap filesystem namespace.

Read-oriented filesystem tools such as Read, Glob, Grep, and Tree may inspect
any path that is visible inside that sandbox namespace, including configured
read-only host binds.

Relative paths still resolve from `@workspace`, and Citra path aliases provide
convenient access to the lifecycle filesystem and source project.

Visibility does not imply write permission.

Write-oriented filesystem operations remain restricted to Citra-owned writable
lifecycle directories. Paths exposed only through read-only host binds can be
inspected but cannot be modified.

## Host filesystem visibility

The sandbox starts from a read-only host filesystem compatibility baseline.

Sensitive or stateful host areas are masked, including normal host home,
temporary, runtime, mount, and similar state directories. Citra may then
reopen specifically approved paths.

As a result:

* some ordinary host paths may be readable;
* masked host paths are unavailable unless explicitly reopened;
* configured read-only binds may be inspected by Read, Glob, Grep, Tree, Bash,
  and other sandboxed tools;
* read visibility never grants write permission;
* ordinary writes remain limited to Citra-owned writable lifecycle directories
  unless the sandbox explicitly grants another writable bind.

The original source project is mounted read-only after other mount setup so
earlier compatibility or writable mounts cannot accidentally make it writable.

## Additional read-only host binds

The current sandbox configuration explicitly exposes these additional host
paths read-only:

{extra_readonly_binds}

These paths may be inspected directly using their absolute paths with
read-oriented filesystem tools or sandboxed commands.

They remain strictly read-only. Do not attempt to edit, overwrite, delete, or
create files beneath them.

A specifically exposed bind does not expose the surrounding masked host tree.
Only paths actually visible in the sandbox namespace should be assumed
accessible.

Citra may also expose additional read-only paths automatically when required
for executable, runtime, PATH, certificate, or development-tool compatibility.
Those paths follow the same rule: they may be inspected when visible, but they
must be treated as read-only.

## Temporary files and host state

Prefer `@tmp` for disposable work.

Do not place scratch files, exploratory repositories, extracted archives,
generated intermediates, or experimental artifacts into the project workspace
unless they are intended to become part of the project.

Host `/tmp` and similar conventional locations are isolated from real host
temporary state. Programs using the supplied environment will normally use
Citra's lifecycle temporary directory.

## Process environment

Sandboxed commands receive a development-compatible environment with mutable
user state redirected into Citra-owned lifecycle directories.

Host IPC and control-plane environment such as SSH-agent, GPG-agent, D-Bus,
and Git control variables may be removed by the sandbox unless a tool
deliberately supplies an allowed value.

The process and device views are sandbox-owned. Do not assume access to host
processes, host IPC sockets, container daemons, SSH agents, or arbitrary host
devices merely because equivalent paths normally exist on the host.

## Network

Sandboxed command execution has no network access by default.

When a tool supports networked execution, request network access only when the
task actually requires it and provide the concrete reason required by that
tool's authorization flow.

Prefer dedicated network-capable tools such as Web Search, Git, Curl, or
Browser when they more narrowly match the operation.

Do not repeatedly retry a network-dependent command with networking disabled.

## Operating discipline

Treat sandbox restrictions as part of the environment, not as problems to
circumvent.

In particular:

* inspect source through `@source` when working with the user's project;
* materialize source files before editing them;
* make project changes in `@workspace`;
* use `@tmp` for disposable work;
* use absolute paths when inspecting visible read-only host binds;
* treat non-Citra host paths as read-only unless explicitly documented
  otherwise;
* do not attempt to expose masked host state;
* do not attempt to write into read-only binds;
* do not attempt to access Citra's trusted control-plane state;
* do not infer write permission from read visibility;
* do not request network access without a concrete need;
* use the narrowest dedicated tool that already provides the required
  capability.
"""