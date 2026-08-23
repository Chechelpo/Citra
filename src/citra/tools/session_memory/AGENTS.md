# AGENTS.md — Conversation Memory Tools

> Cross-turn conversation-memory tools used by the agent to retain structured
> state such as TODOs, facts, decisions, constraints, working state, and resume
> checkpoints.

## Overview

Memory tools are model-visible tools that persist structured state for the
lifetime of a Citra conversation.

They do **not** write their memory to disk automatically. Their state is
retained for the process lifetime, survives user/agent turn boundaries, and
remains injected when older chat messages are omitted from the request context.

All memory tools extend:

```python
MemoryTool
    └── SessionTool
        └── Tool
```

### A lifecycle of working state → durable memory

Durable memory (TODOs, facts, decisions, constraints) is **not** created
directly. The agent first creates a provisional **working state**, then
**promotes** it into one or more durable memories. A working state may be
promoted to several durable entries, or never promoted and discarded.

The memory types and their ownership are:

* **Working state** (`working_state`) — provisional hypotheses and reasoning.
  The seed for everything else. Not authoritative.
* **TODO** (`todo`) — required work, optionally hierarchical.
* **Fact** (`fact`) — verified information, optionally with citations.
* **Decision** (`decision`) — a choice later work must respect.
* **Constraint** (`constraint`) — an active rule or invariant.
* **Checkpoint** (`checkpoint`) — a compact resume point. The one memory type
  **not** promoted from working state; it is set directly.

## Tool lifetime and shared state

Memory tools depend on instance persistence.

Instantiate tools for each model call through the registry and active session:

```python
tools = TOOL_REGISTRY.instantiate(context, session)
```

Do **not** construct memory-tool classes directly on every model call.
Registry instantiation is safe: it reuses the instances owned by a session
and refreshes their execution context.

### Shared `ConversationMemoryState`

All memory tools for one `AgentSession` share a single
`ConversationMemoryState` instance (defined in `memory_tool.py`) that owns the
working-state collection and their promotion references. This lets e.g. a
`todo` promotion record which durable memory was created from which working
state.

```python
memory_state = conversation_memory_state(session)
```

`conversation_memory_state()` stores one state per session by weak reference
(`WeakKeyDictionary`), with a strong-reference fallback keyed by `id(session)`
for sessions that cannot be weak-referenced.

## `memory_tool.py` — `MemoryTool`

`MemoryTool` is the abstract generic base class for all conversation-memory
tools.

It extends `SessionTool` and defines:

```python
@property
@abstractmethod
def heading(self) -> str: ...

@abstractmethod
def get_extracts(self) -> list[TExtract]: ...

@abstractmethod
def format_extract(self, extract: TExtract) -> str: ...

@abstractmethod
def should_offer_documentation(self) -> bool: ...
```

- `heading` — the Markdown section heading used when exposing memory.
- `get_extracts()` — the extracts currently retained. Return a copy where
  practical so callers cannot mutate internal state directly.
- `format_extract()` — convert one extract into LLM-readable Markdown.
- `should_offer_documentation()` — return whether this memory type may contain
  information worth persisting into repository documentation at the end of the
  work. Returning `True` only indicates the runtime may ask; it never writes
  documentation automatically.

The base `format_for_llm()` assembles `heading` + extracted formatted entries
into a Markdown section, returning `""` when there are no extracts.

### Extracts and immutable dataclasses

Extracts are immutable frozen dataclasses, for example:

```python
@dataclass(frozen=True)
class DecisionExtract:
    id: int
    content: str
    working_state_id: int
```

Each tool owns its extract collection privately (e.g. `__extracts`) and never
exposes the mutable list directly — `get_extracts()` returns a copy. IDs are
local to that memory tool.

Durable extracts carry a `working_state_id` (except `CheckpointExtract`) linking
them back to the working state they were promoted from. Promoting a working
state registers a `PromotionRef(kind, memory_id)` on it via
`register_promotion()`; removing a durable memory calls
`unregister_promotion()`.

## Working state: `WorkingStateTool`

`WorkingStateTool` manages provisional reasoning. It is the prerequisite step
for every other durable memory.

### Actions

* `create` — create one working state (`content`) or several (`contents`).
* `update` — replace the content of one working state (`id` + `content`).
* `resolve` — mark working state(s) resolved. Requires each to have at least one
  durable promotion (use `id`/`ids`).
* `discard` — drop working state(s) with no durable promotions.

### Rules

- `create` allows `content` or `contents`, never both, and rejects `id`/`ids`.
- `update` accepts exactly one `id` and one `content`.
- A working state with promotions cannot be discarded; it must be resolved.
- A working state without promotions cannot be resolved; it must be discarded.

### Extract shape

```python
@dataclass(frozen=True)
class WorkingStateExtract:
    id: int
    content: str
    created_turn: int
    updated_turn: int
    promotions: tuple[PromotionRef, ...] = ()
```

`PromotionRef` is `(kind: str, memory_id: int)`.

### LLM formatting

```md
- [W1] Provisional hypothesis...
  - promoted: FACT [3], TODO [5]
```

`WorkingStateTool.should_offer_documentation()` is `False`.

## `TodoTool`

`TodoTool` manages **hierarchical** TODOs, each promoted from a working state.

### Actions

* `promote` — create one or more TODOs from working states
  (`working_state_id`/`working_state_ids`). Optional `content` override and
  `parent_id` (sub-step) / `index` (sibling position) for a **single**
  promotion.
* `check` — mark TODOs completed (via `id`/`ids`). A TODO cannot be checked
  while descendant TODOs remain incomplete. Checking a TODO with a completed
  ancestor reopens that ancestor.
* `remove` — delete TODOs that are stale/invalid (via `id`/`ids`), including
  their descendants. Not equivalent to completion.

`remove` is **not** completion: performed work is `check`ed (`[ ] → [x]`), and
`remove` is reserved for TODOs that should no longer exist.

### Extract shape

```python
@dataclass(frozen=True)
class TodoExtract:
    id: int
    content: str
    working_state_id: int
    completed: bool = False
    parent_id: int | None = None
```

Storage is pre-order so each subtree stays contiguous while rendering flatly,
with indentation derived from ancestry depth.

### Completion invariant

```python
def has_outstanding_todos(self) -> bool:
    return any(not todo.completed for todo in self.__extracts)
```

Outstanding TODOs should block normal agent completion. The runtime, not the
model prompt, must enforce this.

`TodoTool.should_offer_documentation()` is `False` (execution state, not
long-term maintainer knowledge).

## `FactTool`

`FactTool` retains verified facts promoted from working states.

### Actions

* `promote` — promote a single working state (`working_state_id`, optional
  `content`, optional `citations`) or a batch (`facts` array).
* `remove` — delete stale/incorrect facts (`id`/`ids`).

### Citations

A fact may carry citations:

```python
@dataclass(frozen=True)
class Citation:
    type: str              # "file" | "url"
    source: str
    line: int | None = None
    end_line: int | None = None
    reference: str | None = None
```

- `file` citations use `source` (workspace-relative path), optional `line`,
  and `end_line` (which requires `line` and must not precede it).
  `reference` is not valid for file citations.
- `url` citations use `source` (the URL) and optional `reference`; `line` and
  `end_line` are invalid.

### LLM formatting

```md
- [3] Tool.execute validates before invoking _execute.
  - origin: working state W1
  - source: src/citra/tools/tool.py:61-74
```

`FactTool.should_offer_documentation()` is `False`.

## `DecisionTool`

`DecisionTool` records choices made during the run.

### Actions

* `promote` — promote working state(s) to decisions (`working_state_id`/
  `working_state_ids`; optional single `content`).
* `remove` — remove stale/superseded decisions (`id`/`ids`).

### Extract shape

```python
@dataclass(frozen=True)
class DecisionExtract:
    id: int
    content: str
    working_state_id: int
```

`DecisionTool.should_offer_documentation()` returns `True` when decisions
exist — they are strong candidates for end-of-run documentation. Returning a
proposal indicator is advisory and never writes documentation automatically.

## `ConstraintTool`

`ConstraintTool` records rules or limitations that must remain true during the
run.

### Actions

* `promote` — promote working state(s) to constraints.
* `remove` — remove a stale/inapplicable constraint.

### Extract shape

```python
@dataclass(frozen=True)
class ConstraintExtract:
    id: int
    content: str
    working_state_id: int
```

Constraints must be reintroduced into model context on subsequent iterations so
the agent does not forget them. `ConstraintTool.should_offer_documentation()`
returns `True` when constraints exist.

## `CheckpointTool`

`CheckpointTool` keeps one authoritative resume point across agent turns. It is
the one memory type set directly (`set`/`clear`), not promoted from working
state, because it is derived handoff state rather than a durable belief.

### Actions

* `set` — set the checkpoint (`content` required, `next_step` optional).
* `clear` — remove the checkpoint (`content`/`next_step` invalid for clear).

### Extract shape

```python
@dataclass(frozen=True)
class CheckpointExtract:
    content: str
    next_step: str | None
    turn: int
```

`CheckpointTool.should_offer_documentation()` is `False`.

## Injecting memory into model context

Memory tools expose their retained state to the model between iterations:

```python
sections: list[str] = []

for tool in tools.values():
    if not isinstance(tool, MemoryTool):
        continue
    section = tool.format_for_llm()
    if section:
        sections.append(section)

memory_context = "\n\n".join(sections)
```

A resulting memory block might look like:

```md
# Conversation Memory

## Todos
- [ ] [ID 2] Rewrite session_memory/AGENTS.md (from W1)

## Facts
- [3] Citra uses a working-state promotion model.
  - origin: working state W1

## Decisions
- [1] Keep memory tools process-local.

## Constraints
- [1] Durable memory must be promoted from working state.
```

Do not require the model to spend tool calls merely to rediscover memory that
the runtime already owns.

## Runtime responsibilities

Memory tool semantics are split between the tool and the agent runtime.

The tool is responsible for:

* storing memory
* validating operations
* changing extract state
* rendering memory
* signaling whether documentation may be offered

The runtime is responsible for:

* preserving memory tool instances and shared state throughout the conversation
* injecting memory into model context
* enforcing completion invariants (e.g. outstanding TODOs)
* deciding whether and when to ask the user about persistence

In particular, `TodoTool` may expose `has_outstanding_todos()`, but the outer
loop must enforce it.

## General implementation conventions

* Memory tools extend `MemoryTool`, not `SessionTool` directly.
* Keep model-facing function names concise: `working_state`, `todo`, `fact`,
  `decision`, `constraint`, `checkpoint`.
* Use the JSON schema DSL from `citra.utils.json_schema`. Never hand-write raw
  tool-schema dictionaries.
* Set `additional_properties=False`.
* Validate action-specific requirements in Python when the schema DSL cannot
  express them.
* Raise exceptions for invalid IDs, malformed operations, and duplicate/empty
  content.
* Use immutable extract dataclasses.
* Keep extract mutation private to the owning tool.
* Return plain strings from `_execute()`.
* Durable memory is process-local and conversation-durable; it is promoted from
  working state, never created directly.
* `should_offer_documentation()` does not imply automatic persistence.
* Obtain memory tools through the registry and active session during the loop.
* Keep memory formatting compact; it is repeatedly inserted into model context.