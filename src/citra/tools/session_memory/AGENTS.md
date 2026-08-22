# AGENTS.md — Conversation Memory Tools

> Cross-turn conversation-memory tools used by the agent to retain structured
> state such as TODOs, facts, decisions, constraints, and resume checkpoints.

## Overview

Memory tools are model-visible tools that persist structured state for the
lifetime of a Citra conversation.

They do **not** write their memory to disk automatically. Their instances are
owned by `AgentSession.memory`, survive user/agent turn boundaries, and remain
injected when older chat messages are omitted from the request context.

All memory tools extend:

```python
MemoryTool
    └── SessionTool
        └── Tool
```

Each memory tool is responsible for:

1. Holding its own extracts in memory.
2. Exposing model-facing actions through the normal `Tool` contract.
3. Rendering its extracts into an LLM-readable Markdown section.
4. Optionally proposing that useful session knowledge be persisted into project documentation at the end of the run.

## Tool lifetime

Memory tools depend on instance persistence.

Instantiate tools for each model call through the registry and active session:

```python
tools = TOOL_REGISTRY.instantiate(context, session)

while agent_running:
    ...
```

Do **not** construct memory-tool classes directly on every model call:

```python
while agent_running:
    tools = {"todo": TodoTool(context, session)}  # wrong
```

Registry instantiation is safe: it reuses the instances owned by
`session.memory` and refreshes their execution context.

## `memory_tool.py` — `MemoryTool`

`MemoryTool` is the abstract base class for all conversation-memory tools.

It extends `SessionTool` and defines the shared memory interface.

A memory tool should provide:

```python
@property
def heading(self) -> str:
    ...
```

The Markdown section heading used when exposing memory to the model.

```python
def get_extracts(self) -> list[TExtract]:
    ...
```

Returns the extracts currently retained by the tool.

Return a copy where practical so callers cannot mutate internal state directly.

```python
def format_extract(
    self,
    extract: TExtract,
) -> str:
    ...
```

Converts one extract into LLM-readable Markdown.

```python
def format_for_llm(self) -> str:
    ...
```

Produces the complete Markdown section for the tool.

The base implementation should normally assemble `heading`, `get_extracts()`, and `format_extract()`.

Memory tools may also expose:

```python
def get_documentation_proposal(
    self,
) -> DocumentationProposal | None:
    ...
```

This indicates whether information discovered during the run may be worth persisting for future maintainers.

Returning a proposal does **not** write documentation. It only gives the end-of-run layer enough information to ask whether documentation should be created or updated.

## Extract conventions

Extracts should normally be immutable dataclasses.

Example:

```python
@dataclass(frozen=True)
class DecisionExtract:
    id: int
    content: str
```

Each tool owns its extract collection:

```python
self.__extracts: list[DecisionExtract] = []
self.__next_id = 1
```

IDs are local to that memory tool unless a global ID scheme is introduced explicitly.

Do not expose the internal mutable list directly.

Prefer:

```python
def get_extracts(self) -> list[DecisionExtract]:
    return list(self.__extracts)
```

## Model-facing action conventions

Memory tools expose actions through a single tool function.

Example:

```json
{
  "action": "add",
  "content": "Preserve the existing public API."
}
```

Action-specific requirements that cannot be expressed by the current JSON Schema DSL must be validated inside `_execute()` or its helper methods.

For example:

```python
content = arguments.get("content")

if not content:
    raise ValueError(
        "'content' is required for constraint action 'add'."
    )
```

Use exceptions for invalid operations. The normal agent loop will surface the error to the model.

## `TodoTool`

`TodoTool` tracks work that the agent is obligated to complete.

### Actions

* `add` — create a new outstanding TODO.
* `check` — mark a TODO completed while retaining it in memory.
* `remove` — delete a TODO that is stale, invalid, irrelevant, or based on a bad assertion.

`remove` is **not** equivalent to completion.

A TODO that was actually performed should be checked:

```text
[ ] → [x]
```

A TODO should only be removed when the TODO itself should no longer exist.

### Extract shape

```python
@dataclass(frozen=True)
class TodoExtract:
    id: int
    content: str
    completed: bool = False
```

### LLM formatting

Format TODOs as Markdown task items:

```md
## TODOs
- [x] [1] Inspect the existing tool registry
- [ ] [2] Register the new memory tools
```

Checked TODOs remain visible as execution history.

### Completion invariant

Outstanding TODOs should block normal agent completion.

The runtime, not the model prompt, must enforce this.

Example:

```python
def has_outstanding_todos(self) -> bool:
    return any(
        not todo.completed
        for todo in self.__extracts
    )
```

Before accepting a final agent response, inspect the TODO tool. If unresolved TODOs remain, continue the agent loop instead of terminating.

Do not rely only on instructions such as “complete all TODOs before finishing.”

### Documentation

TODOs generally should not produce documentation proposals.

They are execution state, not long-term maintainer knowledge.

## `FactTool`

`FactTool` retains assertions learned during the conversation.

Facts may include supporting citations.

### Actions

* `add` — retain a fact, optionally with citations.
* `remove` — delete a stale, incorrect, or invalid fact.

### Extract shape

A fact should contain:

```python
@dataclass(frozen=True)
class FactExtract:
    id: int
    content: str
    citations: tuple[Citation, ...] = ()
```

### Citations

Supported citation source types:

* `file`
* `url`

A citation may be represented as:

```python
@dataclass(frozen=True)
class Citation:
    type: str
    source: str
    line: int | None = None
    end_line: int | None = None
    reference: str | None = None
```

### File citations

File citations use:

* `source` — workspace-relative file path.
* `line` — optional starting line.
* `end_line` — optional ending line.

Example:

```json
{
  "type": "file",
  "source": "src/citra/tools/tool.py",
  "line": 61,
  "end_line": 74
}
```

Rules:

* `end_line` requires `line`.
* `line` must be at least 1.
* `end_line` must not precede `line`.
* `reference` is not valid for file citations.

### URL citations

URL citations use:

* `source` — the URL.
* `reference` — optional anchor, heading, section, fragment, or other useful reference.

Example:

```json
{
  "type": "url",
  "source": "https://example.com/docs/auth",
  "reference": "OAuth 2.0"
}
```

Rules:

* URL citations must not use `line` or `end_line`.

### LLM formatting

Example:

```md
## Facts
- [1] Tool.execute validates arguments before invoking _execute.
  - source: src/citra/tools/tool.py:61-74
- [2] The external API uses OAuth 2.0.
  - source: https://example.com/docs/auth (OAuth 2.0)
```

Facts are process-local conversation memory unless deliberately persisted
elsewhere.

## `DecisionTool`

`DecisionTool` records choices made during the run.

Use it for implementation, architectural, behavioral, or design decisions that the agent should remain consistent with.

### Actions

* `add` — record a decision.
* `remove` — remove a decision that is stale, invalid, or superseded.

### Extract shape

```python
@dataclass(frozen=True)
class DecisionExtract:
    id: int
    content: str
```

### LLM formatting

```md
## Decisions
- [1] Keep argument validation inside Tool.execute().
- [2] Memory tools retain state on their tool instances.
```

### Documentation

Decisions are strong candidates for end-of-run documentation.

`get_documentation_proposal()` should return a proposal when meaningful decisions exist.

Typical rationale:

> These decisions may explain architectural or implementation choices to future maintainers.

The end-of-run layer may use the proposal to ask whether the decisions should be added to `AGENTS.md`, an architecture document, or another appropriate repository document.

Do not write documentation automatically merely because a proposal exists.

## `ConstraintTool`

`ConstraintTool` records rules or limitations that must remain true during the run.

Examples include:

* compatibility requirements
* invariants
* public API restrictions
* architectural boundaries
* repository conventions
* implementation limitations

### Actions

* `add` — record a constraint.
* `remove` — remove one that is stale, incorrect, or no longer applicable.

### Extract shape

```python
@dataclass(frozen=True)
class ConstraintExtract:
    id: int
    content: str
```

### LLM formatting

```md
## Constraints
- [1] Do not change the public Tool.execute() lifecycle.
- [2] Interactive input must use citra.utils.terminal_input.
```

Constraints should be reintroduced into model context on subsequent iterations so the agent does not forget them.

### Documentation

Constraints are strong candidates for end-of-run documentation because they frequently matter to future maintainers.

`get_documentation_proposal()` should return a proposal when retained constraints exist.

Typical rationale:

> These constraints may affect future changes and should be considered for repository documentation.

## Documentation proposals

Prefer a structured proposal type instead of raw dictionaries.

Example:

```python
@dataclass(frozen=True)
class DocumentationProposal:
    title: str
    reason: str
    content: str
```

Then:

```python
def get_documentation_proposal(
    self,
) -> DocumentationProposal | None:
    ...
```

At the end of the run, collect proposals from all memory tools:

```python
proposals = []

for tool in tools.values():
    if not isinstance(tool, MemoryTool):
        continue

    proposal = tool.get_documentation_proposal()

    if proposal is not None:
        proposals.append(proposal)
```

The runtime may then ask the user whether the proposed knowledge should be written into repository documentation.

The proposal mechanism is advisory. It must not silently persist memory.

## Injecting memory into model context

Memory tools should expose their retained state to the model between iterations.

Example:

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

## TODOs
- [x] [1] Inspect the tool lifecycle
- [ ] [2] Add completion gating

## Facts
- [1] ToolRegistry stores tool classes rather than instances.
  - source: src/citra/tools/tool_registry.py:12-34

## Decisions
- [1] Memory state will live on persistent tool instances.

## Constraints
- [1] Tool.execute() remains the final validation and logging path.
```

Do not require the model to spend tool calls merely to rediscover memory that the runtime already owns.

## Runtime responsibilities

Memory tool semantics are split between the tool and the agent runtime.

The tool is responsible for:

* storing memory
* validating operations
* changing extract state
* rendering memory
* exposing documentation proposals

The runtime is responsible for:

* preserving memory tool instances throughout the conversation
* injecting memory into model context
* enforcing completion invariants
* collecting documentation proposals
* deciding when to ask the user about persistence

In particular, `TodoTool` may expose `has_outstanding_todos()`, but the outer loop must enforce it.

## General implementation conventions

* Memory tools extend `MemoryTool`, not `SessionTool` directly.
* Keep model-facing function names concise: `todo`, `fact`, `decision`, `constraint`.
* Use the JSON schema DSL from `citra.utils.json_schema`.
* Never hand-write raw tool-schema dictionaries.
* Set `additional_properties=False`.
* Validate action-specific requirements in Python when the schema DSL cannot express them.
* Raise exceptions for invalid IDs and malformed operations.
* Use immutable extract dataclasses.
* Keep extract mutation private to the owning tool.
* Return plain strings from `_execute()`.
* Memory is process-local and conversation-durable by default.
* Documentation proposals never imply automatic persistence.
* Obtain memory tools through the registry and active session during the loop.
* Keep memory formatting compact; it is repeatedly inserted into model context.
