# AGENTS.md — `citra` (root package)

> The main application package.  See the top-level `AGENTS.md` for the
> full architecture overview.

---

## `main.py` — REPL, agentic loop, and API interaction

This is the **entry point** (`python -m citra.main`).  It contains:

### Terminal UI helpers
- `render_markdown(text)` — minimal inline `**bold**` rendering.
- `tool_name_for_display(name)` — `"web_search"` → `"Web Search"`.
- `argument_preview(arguments)` / `result_preview(result)` — compact
  one-line previews for tool calls.

### API serialization
- `system_prompt(context)` — the system message sent to the model
  (`"Concise coding assistant. cwd: … os: …"`).
- `call_api(context, messages, tools)` — one POST to the
  chat-completions endpoint using `urllib`.
- `get_assistant_message(response)` — extracts and normalizes the
  assistant message from the API response.
- `serialize_tool_result(result)` — strings pass through; other types
  are JSON-encoded.
- `execute_tool_call(tools, tool_call)` — parses one tool call, looks up
  the tool, executes it, returns a string result (or error string).

### Agentic loop
- `run_agent_turn(messages)` — loops: create context → instantiate tools
  → call API → execute tool calls → append results → repeat until no
  more tool calls.

### Command handling
- `is_command(input)` — checks for `/` prefix.
- `handle_command(input, messages)` — delegates to the `COMMAND_REGISTRY`.

### REPL (`main()`)
- Prints a header (model id + workspace).
- Reads input in a loop through `terminal_input.prompt(...)` (no direct
  `input()`); dispatches to commands or the agentic loop.
- The main REPL prompt has **no** timeout (unlike `PromptUser`).
- Catches `KeyboardInterrupt` / `EOFError` to exit cleanly.

---

## Sub-packages

| Package        | Purpose                                          | Has AGENTS.md |
|----------------|--------------------------------------------------|---------------|
| `context/`     | ExecutionContext + config loading                | ✅            |
| `commands/`    | Slash-command framework + built-in commands      | ✅            |
| `tools/`       | Tool framework + concrete tools                  | ✅            |
| `utils/`       | Shared helpers (JSON schema DSL, terminal, etc.) | ✅            |

Read each sub-package's `AGENTS.md` before working in it.

---

## Notes for agents

- **Fresh context per API call**: `run_agent_turn` creates a new
  `ExecutionContext` and new tool instances on every loop iteration.
  Do not cache contexts or tools across calls.
- The system prompt is intentionally minimal.  If you need richer
  instructions, edit `system_prompt()`.
- Tool-call errors are returned to the model as `error: …` strings so
  the agent can self-correct.  Only unexpected exceptions in the REPL
  itself are shown to the user.
