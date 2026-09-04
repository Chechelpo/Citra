# AGENTS.md — Session memory

This package provides typed, process-local state that survives conversation
compaction and serial-role changes. The filesystem remains the source of truth
for project state; memory records carry the task meaning, evidence, and routing
that a fresh role cannot recover from files alone.

## Core invariants

- Every memory tool extends `MemoryTool` and stores immutable extract
  dataclasses behind a defensive `get_extracts()` copy.
- Stable IDs are local to a record type and use a visible prefix in rendered
  context: R, A, TODO, CH, V, I, and W where applicable.
- Cross-record links are validated through concrete tool types and
  `ConversationMemory`; do not use reflection or `getattr()`.
- A fresh serial-role `AgentSession` may share one `ConversationMemory`. The
  registry rebinds retained tools to the active session without clearing their
  extracts.
- All retained memory services are injected as context, including services the
  current role may read but cannot mutate. Action capabilities remain
  role-specific.
- Mutating calls are logged by the tool lifecycle with the implementation
  module as `origin`. Add debug/warning/error logs for internal lifecycle
  events that are not visible through `Tool.execute()`.
- Keep memory compact. Record a durable delta, not a transcript.

## Record responsibilities

| Tool | Purpose | Normal owner |
|---|---|---|
| `requirement` | User-visible need and final satisfaction | Explore creates; review adjudicates |
| `acceptance_criteria` | Observable proof linked to R IDs | Explore creates; review adjudicates |
| `scope` | Included and excluded boundaries | Explore |
| `constraint` | Mandatory limitation or invariant | Explore, with later corrections |
| `fact` | Verified repository or environment evidence | Any investigating role |
| `decision` | Consequential choice that later work must preserve | Plan |
| `todo` | Proportional executable plan linked to R/A IDs | Plan creates; implement checks |
| `change` | Actual implemented delta and exact paths, linked to TODO/R/A | Implement |
| `verification` | Reproducible pass/fail/blocked evidence linked to R/A/CH/I | Test; review may add or invalidate evidence |
| `issue` | Routed risk, defect, requirement gap, plan gap, or test gap | Every serial role |
| `working_state` | Temporary hypothesis or reasoning | Any role; never survives a checkpoint unresolved |
| `checkpoint` | Authoritative phase route and compact status | Every serial role |

The intended trace is `R -> A -> TODO -> CH -> V/I`. Links are optional in the
balanced workflow when they add no value, but must never dangle. The assured
workflow requires complete coverage without requiring separate mapping
documents.

## Lifecycle rules

Requirements and acceptance criteria are created directly because they come
from the task contract. TODOs may be created directly for established work or
promoted from working state when provenance matters. Facts, decisions, and
constraints retain their existing working-state promotion semantics.

Working state is provisional:

- promote durable consequences, then resolve it; or
- discard it when no durable consequence exists.

Never leave active W records at a role boundary.

Verification is append-oriented. Retests record a new V and supersede old
evidence. Invalidation keeps history while marking evidence stale; removal is
only for records created in error. Each CH has a revision; changing a CH makes
active V evidence linked to its older revision stale. Failed, blocked, or
stale-change evidence prevents workflow completion.

Issues use explicit lifecycle and routing:

- `add` or `update` creates/reopens the finding;
- `resolve` requires correction evidence;
- `reopen` invalidates a prior resolution;
- `route` identifies the earliest capable phase: explore, plan, implement, or
  test;
- unresolved blocking issues prevent completion.

Updating or reopening an R/A invalidates prior satisfaction evidence. Review
alone normally calls `satisfy`, and cites current verification or precise
inspection evidence.

## Completion gates

Serial completion is accepted only when:

- every valid requirement and acceptance criterion is satisfied;
- every valid TODO is checked;
- no active failed or blocked verification remains;
- no unresolved blocking issue remains; and
- no provisional working state remains.

The runtime enforces these conditions. Prompts explain them but are not the
authority.

## Adding a memory tool

1. Add a documented `MemoryTool` subclass and immutable extract.
2. Define strict JSON schema with `additional_properties=False` and validate
   action-specific semantics in Python.
3. Add non-sensitive `format_call_log()` metadata.
4. Export the types from `session_memory.__init__`.
5. Add the tool class to `SESSION_MEMORY_TOOL_TYPES` in
   `default_registry.py`. Runtime discovery must remain tuple-driven, so adding
   a base-compatible class extends the registry without editing discovery
   control flow.
6. Grant the minimum actions required to each serial role and ensure at least
   one phase owns the full lifecycle.
7. Add focused lifecycle, cross-reference, context-handoff, and completion-gate
   tests.

Do not add a record merely because a phase exists. A new type is justified
when information has distinct ownership or lifecycle semantics that cannot be
represented accurately by an existing tool.
