# AGENTS.md — Workflows

This package owns selectable workflows, phase routing, and workflow-level
sandbox policy.

## Runtime invariants

- Select a workflow before provisioning the workspace and sandbox.
- `WorkflowRuntime` owns the concrete sandbox. Serial phase changes never
  replace the workspace, sandbox, LSP manager, process supervisor, or subagent
  supervisor.
- All phases of one serial workflow use the same sandbox configuration.
- Each serial phase gets a fresh `AgentSession` and `SkillRegistry`, while the
  run's task-scoped `ConversationMemory` and filesystem are shared.
- A role receives the original task, all retained structured memory, and only
  the previous role's final assistant handoff—not prior conversation history or
  hidden reasoning.
- A new checkpoint revision with an allowed `next_step` is the authoritative
  route. Assistant prose never selects the transition.
- Loops are valid and bounded by `max_executions`.
- Lifecycle transitions and rejection paths use level-appropriate logs whose
  logger/source identifies the originating module.

## Built-ins and modular discovery

`BUILTIN_WORKFLOW_TYPES` is the extension point for runtime discovery. Add a
new `Workflow` subclass to that tuple; do not add another conditional branch to
registry construction.

- `chat`: persistent conversational workflow.
- `task`: persistent general-purpose task workflow.
- `serial_roles`: balanced five-role workflow with proportional planning.
- `serial_roles_assured`: the same roles and feedback routes with stricter
  traceability and evidence coverage.
- `architect`: separate architecture-focused workflow.

The two serial-role workflows intentionally remain general. Architecture work
belongs to `architect`; a serial planner records only choices that materially
affect the current implementation.

## Serial roles and memory ownership

| Role | Primary responsibility | Records it normally mutates |
|---|---|---|
| Explore | Ground goal, boundaries, current behavior, and success | R, A, scope, constraint, fact, I, W, checkpoint |
| Plan | Produce the smallest executable TODO set | TODO, decision, fact, I, W, checkpoint |
| Implement | Change files and record the actual delta | TODO completion, CH, fact, I, W, checkpoint |
| Test | Produce independent evidence and reproducible findings | V, I, W, checkpoint |
| Review | Adjudicate R/A and route unresolved findings | R/A satisfaction, V, I, W, checkpoint |

Every role can read every retained record. Tool capabilities constrain writes
to the listed lifecycle actions.

The handoff trace is `R -> A -> TODO -> CH -> V/I`. The balanced workflow uses
links when helpful; the assured workflow requires full coverage but does not
require an extra planning artifact.

## Feedback routing

Late evidence routes to the earliest role that can correct it:

- unclear need or boundary → explore;
- flawed approach or plan → plan;
- code defect or incomplete change → implement;
- missing, blocked, or flaky verification → test;
- complete current evidence → review;
- all runtime gates clear → complete.

Test and review do not silently repair defects. They record an issue with
evidence and route it. On a repeated phase, update existing records or reopen
them rather than duplicating the same finding.

## Development rules

- Document every added or changed class and function.
- Keep role prompts direct and proportional; a local task may have one TODO.
- Keep selectable workflow discovery tuple-driven.
- Prefer explicit imports under `TYPE_CHECKING` to reflection when avoiding
  circular imports.
- Split any Python module before it exceeds 1000 lines.
- Test allowed and rejected transitions, fresh-session/shared-memory behavior,
  memory visibility, role capabilities, feedback loops, and completion gates.
