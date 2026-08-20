# AGENTS.md — `citra.utils`

> Shared helpers used across the codebase.

---

## Files

### `json_schema.py`

A small **DSL for building OpenAI-compatible JSON schemas** without
hand-writing dicts.

Key types (all frozen dataclasses):

| Type                  | Purpose                                          |
|-----------------------|--------------------------------------------------|
| `JsonSchema`          | Represents a JSON schema node (one of `JsonType`). |
| `JsonProperty`        | A named property inside an object schema (`name`, `schema`, `required`). |
| `FunctionDefinition`  | `name`, `description`, `parameters` (must be object), `strict`. |
| `ChatCompletionTool`  | Wraps a `FunctionDefinition` with `type="function"`. |

`JsonSchema` factory methods: `.string()`, `.integer()`, `.number()`,
`.boolean()`, `.array(items)`, `.object(properties)`.

Every schema has `.to_dict()` which produces the dict sent to the model
API. The `Tool` base class calls `Draft202012Validator.check_schema()`
on the dict at construction time.

**Use this DSL for all tool parameter definitions** — never write raw
dict literals.

---

### `workspace.py`

- **`resolve_path(context, path) → Path`** — resolves a user-provided
  path. Absolute paths are returned as-is (resolved). Relative paths
  are joined against `context.workspace`. `~` is expanded.
- **`display_path(context, path) → str`** — returns the path relative to
  the workspace if possible, otherwise the absolute path. Used for
  showing paths to the model/user.

Use these helpers for workspace-aware path handling instead of
reimplementing relative-path resolution in individual tools.

---

### `api.py`

- **`chat_completions_url(host) → str`** — normalizes an API host into
  the `/chat/completions` endpoint. Idempotent if the host already ends
  with `/chat/completions`.

Keep API URL normalization centralized here rather than duplicating it
inside the agent loop.

---

### `prompt.py`

Owns construction of Citra's **system prompt** and the dynamic
environment context supplied to the model on every API call.

The main public entry point is:

```python
build_system_prompt(context: ExecutionContext) -> str
