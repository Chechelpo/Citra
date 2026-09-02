# `citra.context`

This package owns the process-lifetime project runtime and the execution
context shared by tools.

## Project ownership

- `WorkspaceContext.create()` receives the controller-owned project path and
  copies the complete project, including VCS metadata, into the runtime project
  root before model execution begins.
- Model-facing paths are relative to that copied project. `.` is the project
  root. Do not add a second project alias, source bridge, source mount, or
  materialization/apply workflow.
- The controller source path is private and must not appear in prompts, tool
  schemas, ordinary filesystem aliases, or model process environments.
- Project changes stay uncommitted. The user owns Git commits. The model may
  restore exact tracked files only through the `workspace` tool.
- Runtime/home/cache/tmp/env roots remain lifecycle-owned. The document library
  is available only through its trusted semantic tools.

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

Mode/workflow sandbox contributions are merged into a cloned `SandboxPolicy`
before `WorkspaceSandbox` is constructed. Operator network denial is
monotonic, and bind lists are additive.

Lint placeholders are `{path}`, `{relative_path}`, and `{project}`. Project
auto-detection reads the nearest `pyproject.toml` inside the copied project.

## Execution context

`ExecutionContext` owns references to the selected config, mode, workflow,
sandbox, scoped filesystem client, LSP manager, browser, subprocess manager,
repository map, and lint runner. Workflow phase changes may rebind the active
mode, skills, and run, but must reuse process-lifetime services.

Keep controller diagnostics distinct from model-facing descriptions. Internal
manifests may record source/runtime paths; prompts and tool results may not.
