# `citra.context`

This package owns the process-lifetime project runtime and the execution
context shared by tools.

## Project ownership

- `WorkspaceContext.create()` receives the controller-owned project path and
  copies the complete project, including VCS metadata, into the runtime project
  root before model execution begins.
- Model-facing paths are relative to that copied project. `.` is the project
  root. Do not add a second project alias, source bridge, source mount, or
  model-facing materialization/apply workflow.
- The controller source path is private and must not appear in prompts, tool
  schemas, ordinary filesystem aliases, or model process environments.
- Project changes stay uncommitted. The user owns Git commits. The model may
  restore exact tracked files only through the `workspace` tool.
- At copy time, the controller records content identities for applicable files
  and symlinks. A copied Git repository root honors its tracked/non-ignored
  inventory; a plain directory or repository subdirectory uses a filesystem
  inventory. The user-only `/apply` command compares the checkout to that
  baseline, rejects concurrent source conflicts, copies confirmed changes back
  transactionally, and optionally stages clean paths in the containing Git
  worktree. Applying does not require a repository.
- Runtime/home/cache/tmp/env roots remain lifecycle-owned. The document library
  is available only through its trusted semantic tools.
- Controller process diagnostics live in `<process-root>/logs/latest.log`.
  `CITRA_ROOT/logs/config.toml` is the only installation log asset loaded.
  Normal shutdown preserves the process log directory beside `workspace/`.

## Configuration

`CITRA_CONFIG_PATH` names a directory with these domains:

| File | Purpose |
|------|---------|
| `models.toml` | Named profiles and orchestrator/subagent selectors. |
| `tools.toml` | Bash, subprocess, browser, web-search, and LSP settings. |
| `sandbox.toml` | Operator sandbox policy. |
| `linting.toml` | Optional project lint fallback. |

`CitraConfig.load()` aggregates these files. `tools.toml`, `models.toml`, and
`sandbox.toml` are required; `linting.toml` is optional. Curl is not a Citra
tool or config domain; network command-line work uses Bash policy.

The selected workflow's unified sandbox contribution is merged into a cloned
`SandboxPolicy` before `WorkspaceSandbox` is constructed. Operator network
denial is monotonic, and bind lists are additive.

Lint placeholders are `{path}`, `{relative_path}`, and `{project}`. Project
auto-detection reads the nearest `pyproject.toml` inside the copied project.

## Runtime provisioning

- Runtime discovery is an ordered tuple of `RuntimeDiscovery` objects. Extend
  coverage by adding an object to `runtime_discovery.DISCOVERIES`.
- `FULL_SANDBOX` must copy every required discovered runtime asset beneath the
  lifecycle-owned runtime directory and mount those copies at compatibility
  paths. It must fail provisioning instead of falling back to a host bind.
- `PARTIAL_SANDBOX` must expose those same assets through read-only host binds
  and perform no runtime copies.
- Both modes publish discovered commands through `/runtime/bin`. Every
  model-facing process launcher, including Bash, filesystem workers, LSP,
  background processes, browser workers, Git, and Mermaid, must use
  `WorkspaceSandbox.run()` or `WorkspaceSandbox.popen()` so this resolver and
  its isolated `PATH` remain authoritative.

## Execution context

`ExecutionContext` owns references to the selected config, active executable
workflow, workflow runtime, sandbox, scoped filesystem client, LSP manager,
browser, subprocess manager, repository map, and lint runner. Serial phase
changes may rebind the active single-mode workflow and skills, but must reuse
process-lifetime services.

Keep controller diagnostics distinct from model-facing descriptions. Internal
manifests may record source/runtime paths; prompts and tool results may not.
