# AGENTS.md — `citra.agent`

> Conversation, durable memory, steering, and model-loop state for a running
> Citra process.

---

## Purpose

The `citra.agent` package owns persistent **conversation state** that survives
both individual model API calls and user/agent turn boundaries.

It currently contains:

```text
agent/
├── __init__.py
├── conversation_memory.py
├── interactions.py
├── response.py
├── runner.py
├── session.py
└── steering.py
````

The package is responsible for:

* OpenAI-compatible conversation history;
* durable structured working memory;
* user steering messages;
* safe insertion of steering into conversation history;
* protocol-safe model/tool-call orchestration;
* foreground handoff for model-originated user questions.

It does **not** own:

```text
terminal rendering
terminal input
command registration
ExecutionContext
```

Those responsibilities belong elsewhere.

---

## `session.py`

### `AgentSession`

`AgentSession` represents one persistent Citra conversation.

It owns:

```python
message_groups: list[MessageGroup]
steering: SteeringInbox
memory: ConversationMemory
turn_number: int
```

`message_groups` contains the OpenAI-compatible conversation history without
ever splitting an assistant tool-call message from its tool results.

`steering` contains user instructions queued for later insertion.
`memory` owns the long-lived TODO/fact/decision/constraint/checkpoint tool
instances, so truncating old chat messages cannot erase working state.

### Message ownership

Only the agent execution path should directly mutate:

```python
session.message_groups
```

Other code should prefer the methods exposed by `AgentSession`.

Available operations:

| Method                                  | Purpose                                                             |
| --------------------------------------- | ------------------------------------------------------------------- |
| `add_user_message(content)`             | Append a normal `role="user"` message.                              |
| `add_assistant_message(message)`        | Append an assistant response.                                       |
| `add_tool_result(tool_call_id, result)` | Append a `role="tool"` result associated with a tool call.          |
| `queue_steering(content)`               | Add a user correction to the steering inbox.                        |
| `flush_steering()`                      | Convert all queued steering instructions into normal user messages. |
| `clear_history(clear_memory=True)`      | Clear history, steering, and normally durable memory.                |

---

## `ChatMessage`

`ChatMessage` is currently:

```python
ChatMessage = dict[str, Any]
```

It represents an OpenAI-compatible message object.

Typical message shapes include:

### User

```python
{
    "role": "user",
    "content": "Fix the failing test.",
}
```

### Assistant

```python
{
    "role": "assistant",
    "content": "...",
}
```

or:

```python
{
    "role": "assistant",
    "content": None,
    "tool_calls": [...],
}
```

### Tool

```python
{
    "role": "tool",
    "tool_call_id": "...",
    "content": "...",
}
```

Do not invent an incompatible internal conversation format unless the
model API layer is migrated at the same time.

---

## Safe steering boundaries

Steering instructions are intentionally kept outside `messages` until
they can be inserted safely.

`flush_steering()` converts queued steering instructions into normal user
messages only after all tool results in the active group are present:

```text
SteeringInbox
    ↓
flush_steering()
    ↓
role="user" messages
    ↓
conversation history
```

A critical invariant is:

> If an assistant message contains tool calls, all corresponding tool
> result messages must be appended before queued steering messages are
> flushed into conversation history.

For example, this ordering is valid:

```text
assistant(tool calls A, B)
tool(A result)
tool(B result)
user(steering correction)
assistant(...)
```

Do not insert steering between an assistant tool-call message and its
required tool results.

---

## `steering.py`

### `SteeringInbox`

`SteeringInbox` is a small thread-safe FIFO queue for user steering
instructions.

Internally it uses:

```python
deque[str]
Lock
```

FIFO ordering is intentional.

If the user submits:

```text
1. Don't change the database schema.
2. Reuse the existing service.
3. Run the Java tests afterward.
```

the model should receive them in that same order.

Do not replace it with LIFO/stack behavior.

---

## Steering API

### `push(message) -> bool`

Queues one steering message.

Expected behavior:

```text
non-empty message
    -> append
    -> return True

empty / whitespace-only message
    -> ignore
    -> return False
```

Messages should be stripped before storage.

---

### `drain() -> tuple[str, ...]`

Atomically:

1. copies all currently queued messages;
2. clears the queue;
3. returns the copied messages in FIFO order.

This is used by `AgentSession.flush_steering()`.

---

### `has_pending() -> bool`

Returns whether at least one steering message is waiting.

This should remain a cheap thread-safe check.

---

### `clear()`

Removes all pending steering messages.

Used when session state is reset.

---

### `__len__() -> int`

Returns the current number of queued steering messages.

Keep this operation thread-safe.

---

## Thread-safety rules

`SteeringInbox` may be accessed from different execution contexts, so
all queue inspection and mutation must remain protected by its lock.

Protect at minimum:

```text
push
drain
has_pending
clear
__len__
```

Do not expose the underlying `deque`.

`AgentSession.messages` itself is not currently a general-purpose
thread-safe container.

The intended ownership rule is:

```text
agent execution path
    -> mutates messages

other execution paths
    -> submit steering through SteeringInbox
```

Preserve this distinction.

---

## Conversation lifecycle

An `AgentSession` is intended to survive across multiple model calls.

Conceptually:

```text
AgentSession
│
├── user message
├── assistant response
├── tool results
├── assistant response
├── user message
├── ...
│
└── steering state
```

Do not recreate the session for every model API call or user turn.

Transient objects such as:

```text
Tool instances
individual API request objects
```

have a shorter lifetime than `AgentSession`.

---

## Clearing state

By default, `clear_history()` must clear all three:

```python
messages
steering
memory
```

This prevents queued steering from leaking into a newly cleared
conversation.

Passing `clear_memory=False` is reserved for callers deliberately replacing
only visible chat history. Do not clear application services or global
infrastructure from this method.

---

## Tool-call cancellation interaction

The agent execution layer may use:

```python
session.steering.has_pending()
```

to decide whether remaining unexecuted tool calls should be skipped.

When tool calls are skipped, the execution layer is responsible for
still appending valid synthetic tool results for the corresponding tool
call IDs.

That policy belongs to the execution loop.

`AgentSession` itself should remain responsible only for state and
message ordering primitives.

Do not move tool execution logic into `session.py`.

---

## Separation of responsibilities

Keep the package boundary approximately:

```text
citra.agent
    conversation state and memory
    steering and model-loop protocol

citra.tools
    model tools and execution

citra.context
    execution environment/configuration

citra.utils
    reusable helpers

application / cli
    process lifecycle and terminal orchestration
    API calls
    tool-call sequencing
```

`AgentSession` should not import:

```text
ToolRegistry
urllib
prompt_toolkit
terminal styling
model API helpers
language-server implementations
```

---

## `__init__.py`

Re-export the public agent-state API:

```python
from .session import AgentSession, ChatMessage
from .steering import SteeringInbox
```

Expected public surface:

```python
__all__ = [
    "AgentSession",
    "ChatMessage",
    "SteeringInbox",
]
```

Consumers should normally import:

```python
from .agent import AgentSession
```

rather than depending on internal implementation details.

---

## Notes for agents

* `AgentSession` is process-lifetime conversation state.
* `ExecutionContext` and the workspace are process-lifetime application
  services, but are not conversation memory.
* Ordinary tools remain transient. Memory-tool instances are owned by
  `AgentSession.memory` and rebound to the current context when instantiated.
* Steering uses FIFO ordering.
* Never directly insert steering into the middle of an unresolved
  assistant tool-call sequence.
* `flush_steering()` is only valid at a protocol-safe conversation
  boundary.
* Preserve OpenAI-compatible message ordering.
* Keep `SteeringInbox` thread-safe.
* Do not put model transport, terminal UI, or tool execution logic into
  this package.
* `clear_history()` clears both conversation messages and queued
  steering.
* Prefer adding small state-management primitives here rather than
  turning `AgentSession` into a general application controller.
