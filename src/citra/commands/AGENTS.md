# AGENTS.md — `citra.commands`

> Slash-command framework for Citra's REPL.

Commands are user-facing actions triggered by typing `/name` at the
interactive prompt.  They run **outside** the agentic loop — they do not
involve the model and are executed synchronously by `main.py`'s
`handle_command()`.

---

## Core framework

### `command.py`

**`Command`** (abstract base class):

- Class attributes: `id` (str, the name without `/`), `description` (str).
- `__init__(context: ExecutionContext)` — stores the context.
- `run(args: str) → CommandResult` (`@final`) — catches exceptions,
  delegates to `_run`.
- `_run(args: str) → CommandResult` (`@abstractmethod`) — override in
  subclasses.

**`CommandResult`** (frozen dataclass):

| Field             | Type   | Effect                                           |
|-------------------|--------|--------------------------------------------------|
| `output`          | `str`  | Text printed to the terminal after execution.    |
| `clear_messages`  | `bool` | If `True`, the conversation history is cleared.  |
| `exit`            | `bool` | If `True`, the REPL terminates.                  |

**`CommandRegistry`**: stores command classes; `instantiate(id, context)`
returns a fresh `Command` or `None` if unknown.

### `default_registry.py` — `COMMAND_REGISTRY`

Module-level singleton.  **This is where you wire in new commands.**

### `__init__.py`

Re-exports `Command`, `CommandRegistry`, `CommandResult`,
`COMMAND_REGISTRY`.

---

## Existing commands

| Id    | Class          | File       | Effect                                  |
|-------|----------------|------------|-----------------------------------------|
| `q`   | `QuitCommand`  | `quit.py`  | Exits the REPL (`exit=True`).           |
| `c`   | `ClearCommand` | `clear.py` | Clears conversation (`clear_messages=True`). |
| `help`| `HelpCommand`  | `help.py`  | Lists all registered commands.          |
| `test`| `TestCommand`  | `test.py`  | Runs diagnostic checks (config, model API, web search, bash, workspace). |

Legacy alias: bare `/exit` is mapped to `/q` in `main.py`.

---

## How to add a new command

1. Create `commands/my_command.py`:
   ```python
   from .command import Command, CommandResult

   class MyCommand(Command):
       id = "my_command"
       description = "What it does."

       def _run(self, args: str) -> CommandResult:
           return CommandResult(output="result")
   ```
2. In `default_registry.py`:
   ```python
   from .my_command import MyCommand
   COMMAND_REGISTRY.register("my_command", MyCommand)
   ```

---

## Notes for agents

- Commands receive raw text **after** the command name (already
  stripped).  Most commands ignore `args`.
- `HelpCommand` reads from `COMMAND_REGISTRY.help_lines()` — new
  commands appear in `/help` automatically once registered.
- `TestCommand` is the most complex command; it demonstrates how to
  make direct HTTP calls (model API + SearXNG) and render check results
  with terminal colors.
