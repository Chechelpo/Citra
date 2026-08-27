# Agent Runtime

Citra owns one process-lifetime Agent Runtime for each invocation. It is a
disposable developer filesystem and process environment, not a VM or a
container image.

## Filesystem contract

The runtime root is named `citra-process-<pid>-<nonce>` and contains:

| Path | Authority |
|---|---|
| `workspace/` | Complete writable startup copy of the selected source (omitted in direct-source mode) |
| `runtime/` | Immutable copied tool assets |
| `env/` | Shared writable dependency environment |
| `cache/` | Shared writable ecosystem caches |
| `tmp/` | Shared writable temporary state |
| `home/agent/` | Private writable agent home and XDG state |
| `metadata/` | Controller-only ownership, staging, and manifest data |

The original source remains available read-only as `@source`. Editing the
workspace never changes it implicitly. The Commit workflow is the only bridge
that applies staged workspace changes back to the source, with startup-snapshot
conflict checks for content, modes, symlinks, additions, and deletions.

Set `[workspace].direct_source = true` to skip the project copy. In this mode,
`@workspace`, `@source`, and relative project paths resolve to the permanent
source; writes take effect immediately, and the Commit/materialize tools are
not exposed. The remaining process-lifetime runtime, sandbox, dependency,
cache, home, temporary, and cleanup services remain active.

Set `[memory].enabled = false` to remove durable conversation-memory tools,
their system-prompt guidance, and the task-recognition memory skill. This does
not disable ordinary conversation history.

## Provisioning and sandboxing

Tools declare commands, assets, copy policy, environment contributions, and
optional health checks. Asset size is measured before copying. The hard copy
budget excludes the project workspace; an over-budget `copy-or-bind` asset is
exposed through an explicit read-only bind, while an over-budget
`copy-required` asset fails provisioning.

Bubblewrap starts from an empty root by default. It mounts provisioned runtime
assets and `runtime/` read-only, mounts the active project plus
`env/cache/tmp/home` writable, and creates synthetic process/device and
temporary views. In isolated-copy mode it remounts both source views
read-only; in direct-source mode the source itself is the writable project
mount. A broad host-root compatibility bind is available only as an explicit
operator configuration.

## Environment

`WorkspaceContext.environment()` is the canonical environment builder for
Bash, managed subprocesses, LSP, lint, browser workers, and fixed filesystem
workers. Precedence is:

1. safe inherited values;
2. tool defaults;
3. aggressive ecosystem normalization, when enabled;
4. configured runtime overrides;
5. forced isolation mappings.

Forced values include `HOME`, all conventional temp and XDG writable paths,
and `CITRA_WORKSPACE`, `CITRA_SOURCE`, `CITRA_AGENT_ROOT`, `CITRA_RUNTIME`,
`CITRA_ENV`, `CITRA_CACHE`, and `CITRA_TMP`. Configuration rejects overrides
for these reserved keys. Other override values can use aliases such as
`@env`, `@cache`, or `@workspace`.

## Lifecycle and recovery

Shutdown first marks the runtime as closing, preventing new model/tool and
managed-process work. It then closes interaction, LSP, browser, and subprocess
owners with bounded termination before deleting the full runtime root.
Cleanup is idempotent.

During an active turn, the first `Ctrl+C` queues one soft stop instruction. A
second `Ctrl+C` requests hard application shutdown, does not join the daemon
API thread, performs bounded child cleanup, and removes the runtime root.

Startup scans only directories matching the runtime naming convention. A
candidate is deleted only when valid owner metadata matches its name/PID and
the owning process identity is no longer live. Unverified roots are preserved
and reported as warnings.

`metadata/runtime-manifest.json` records the versioned runtime identity,
provisioning budget/usage, copy-or-bind modes, tool health, environment policy,
storage usage, soft limits, and warnings without recording environment values
or provider credentials.
