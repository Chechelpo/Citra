# AGENTS.md — `citra.workflows`

This package owns every selectable execution workflow, process-level
orchestration, and sandbox policy selection. There is no separate runtime mode
abstraction: an ordinary agent configuration is a `SingleModeWorkflow`.

## Invariants

- Select the workflow before provisioning the workspace or sandbox.
- `WorkflowRuntime` owns the concrete `WorkspaceSandbox`; application and
  execution contexts only retain references to that owned instance.
- Every root `Workflow` exposes one concrete `sandbox_config`. Freeze that
  policy at startup and maintain it for the entire run; serial phase workflows
  must use the same configuration.
- Never replace the `WorkspaceContext`, `WorkspaceSandbox`, runtime, LSP
  manager, process supervisor, or subagent supervisor at a serial phase edge.
- Every serial phase gets a new `AgentSession` and `SkillRegistry`.
- Every serial role session shares the task-scoped `ConversationMemory` owned
  by its `WorkflowRun`; it never shares conversation or tool-call history.
- Cross-role context is limited to the original task, shared filesystem,
  structured memory tools, and the immediately previous role's final assistant
  message delivered as the next role's user message.
- A serial transition must be explicitly allowed by the active
  `WorkflowStep`. Loops are valid but bounded by `max_executions`.
- The standard checkpoint memory tool is the routing boundary. Require a new
  checkpoint revision and validate `next_step`; never infer routing from the
  assistant message.
- The assistant message carries operational handoff detail. Durable facts,
  decisions, constraints, TODOs, working state, and checkpoint state survive
  primarily through their standard memory tools.

## Built-ins

- `chat`: one persistent conversational workflow.
- `task`: one persistent general-purpose task workflow.
- `serial_roles`: explore → plan → implement → test → review with validated
  backward and repeat transitions; overrides the sandbox with `FULL_SANDBOX`.
- `architect`: one architect workflow with real component subagents; uses the
  sandbox with `FULL_SANDBOX`.
