# AGENTS.md — Citra

> **Citra** is a minimal agentic coding assistant: an interactive REPL that
> exposes a set of tools to an OpenAI-compatible chat-completions model and
> runs the model's tool calls against the local workspace.

This file is the entry point for any agent (human or AI) working on the
project.  Each sub-package has its own `AGENTS.md` with more specific
information.  Read those when you need to work inside that package.

---

## Quick start

```bash
./start.sh          # launches the REPL (sets PYTHONPATH and config path)
pip install -e .    # editable install if you want `import citra` elsewhere
```

The project requires **Python ≥ 3.12** and the runtime dependencies
**`jsonschema`** and **`prompt-toolkit>=3.0`** (declared in
`pyproject.toml`; lockfile managed with `uv`).

---

## Repository layout

```
.
├── pyproject.toml          # build config, metadata, dependencies
├── start.sh                # entry-point script (sets env, runs `python -m citra.main`)
├── tests/                  # unittest suite (PromptUser, terminal_input, REPL)
└── src/
    └── citra/
        ├── main.py          # REPL + agentic loop + API calls
        ├── context/         # ExecutionContext, config loading          ← AGENTS.md
        ├── commands/        # slash commands (/q, /c, /help, /test)     ← AGENTS.md
        ├── tools/           # tool framework + concrete tools           ← AGENTS.md
        └── utils/           # shared helpers (JSON schema, terminal,
                             #   centralized terminal_input, …)          ← AGENTS.md
```

---

## Architecture overview

### Execution context

`citra.context.ExecutionContext` is a frozen dataclass created **once per
model API call**.  It captures the current OS, workspace path, and
configuration (model + web-search settings loaded from `config.toml`).
Tool and command instances are bound to a single context and discarded
after the call.

### Agentic loop (`main.py`)

1. User types a message (or a `/`-prefixed slash command) through the
   centralized `terminal_input` utility (no direct `input()` calls).
2. For non-command input the message is appended to the conversation and
   `run_agent_turn()` is called.
3. The loop:
   - creates a fresh `ExecutionContext` + fresh tool instances,
   - sends the conversation to the model's chat-completions endpoint,
   - executes any returned tool calls,
   - appends results to the conversation,
   - repeats until the model returns no tool calls.

### Tools (`citra.tools`)

Every tool subclasses `Tool` (in `tool.py`), declares a
`ChatCompletionTool` definition (name, description, JSON-schema
parameters), and implements `_execute(arguments) → Any`.  Arguments are
validated against the schema before execution.

Tools are **registered** in `tools/default_registry.py`, which builds the
module-level `TOOL_REGISTRY` singleton.  Add a new tool there.

### Commands (`citra.commands`)

Slash commands (`/q`, `/c`, `/help`, `/test`) follow a parallel pattern:
`Command` base class → `CommandResult` → `CommandRegistry`.  Registered
in `commands/default_registry.py`.

### Configuration

Configuration is loaded from a TOML file whose path is given by the
`CITRA_CONFIG_PATH` environment variable (set by `start.sh`).
The file must contain:

```toml
[model]
host = "https://…"      # OpenAI-compatible API base URL
api_key = "…"
id = "…"                # model name

[web-search]
host_url = "https://…"  # SearXNG base URL
```

---

## Coding conventions

- **No third-party deps beyond `jsonschema` and `prompt-toolkit`.**
  HTTP calls use `urllib`; no `requests`, no `aiohttp`.  The only
  `prompt_toolkit` import lives in `utils/terminal_input.py`.
- **Frozen dataclasses** for configuration and schema objects.
- **ABC + `@abstractmethod` / `@final`** for the tool/command base
  classes (`_execute`/`_run` are abstract; `execute`/`run` are final).
- **JSON-schema DSL** (`utils/json_schema.py`): build parameter schemas
  with `JsonSchema.object/string/integer/…` and `JsonProperty` — do not
  hand-write raw dicts.
- **Workspace paths**: always resolve user-provided paths through
  `utils/workspace.resolve_path(context, path)` and display them with
  `display_path(context, path)`.
- **Logging**: tools use `logging.getLogger(__name__)`; the base
  `Tool.execute` logs every call.
- **Error handling in the loop**: tool/command errors are caught and
  returned as textual error strings (to the model or terminal) rather
  than crashing the REPL.

---

## How to add a new tool

1. Create `tools/my_tool.py`, subclass `Tool`, define `DEFINITION`
   (a `ChatCompletionTool`), implement `_execute`.
2. Import it in `tools/default_registry.py` and call
   `TOOL_REGISTRY.register("my_tool", MyTool)`.
3. The tool is now visible to the model and instantiated automatically on
   every API call.

See `tools/AGENTS.md` for the full pattern and the list of existing tools.

---

## How to add a new command

1. Create `commands/my_command.py`, subclass `Command`, set `id` and
   `description`, implement `_run(args) → CommandResult`.
2. Import it in `commands/default_registry.py` and call
   `COMMAND_REGISTRY.register("my_command", MyCommand)`.

See `commands/AGENTS.md` for details.

---

## Testing

The test suite uses the stdlib `unittest` (no extra deps) and lives in
`tests/`:

```bash
PYTHONPATH=src CITRA_CONFIG_PATH=.config/config.toml \
  .venv/bin/python -m unittest discover -s tests
```

`terminal_input`'s inactivity-timeout logic is tested deterministically
by driving the internal `_IdleWatchdog` with a fake event loop rather
than real sleeps.  `PromptUser` is tested with the `terminal_input`
utility mocked.  The REPL integration test mocks both
`terminal_input.prompt` and the model API.
