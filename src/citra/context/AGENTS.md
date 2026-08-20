# AGENTS.md — `citra.context`

> Execution context and configuration loading.

This package provides the runtime context that every tool and command
operates within.

---

## Files

### `execution_context.py`

`ExecutionContext` — a **frozen dataclass** created once per model API
call.  It captures:

| Property       | Source                                        |
|----------------|-----------------------------------------------|
| `os`           | `platform.system()` (normalized: `darwin`→`macos`) |
| `workspace`    | `os.getcwd()` at construction time            |
| `config`       | `CitraConfig.load(path)` from `CITRA_CONFIG_PATH` |
| `model_config` | shortcut → `config.model`                     |
| `web_search_config` | shortcut → `config.web_search`          |

**Key method:** `has_command(cmd)` — checks whether an executable is on
`PATH` (used by the `bash` tool).

The dataclass uses private (`__`-prefixed) fields set via
`object.__setattr__` in `__post_init__` because frozen dataclasses
prohibit normal assignment.

### `config_loader.py`

Three frozen dataclasses:

- **`ModelConfig`** — `host`, `api_key`, `id`, `max_tokens`.
- **`WebSearchConfig`** — `host_url`.
- **`CitraConfig`** — top-level config, loaded from a TOML file via
  `CitraConfig.load(path)`.  The file must contain `[model]` and
  `[web-search]` tables.  Missing keys raise `ValueError`.

TOML parsing uses the stdlib `tomllib` module.

### `__init__.py`

Re-exports `ExecutionContext`, `CitraConfig`, `ModelConfig`,
`WebSearchConfig`.

---

## Important notes for agents

- **Never construct `ExecutionContext` with positional args** — it reads
  from environment / cwd automatically.  Just call `ExecutionContext()`.
- The config file path comes from the **`CITRA_CONFIG_PATH`** env var,
  which is set by `start.sh`.  If it's missing, construction raises
  `RuntimeError`.
- `ExecutionContext` is **frozen** — you cannot mutate it after
  construction.  Create a new one if you need a different context.
