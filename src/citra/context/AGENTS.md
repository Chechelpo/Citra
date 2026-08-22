# AGENTS.md — `citra.context`

> Execution context and configuration loading.

This package provides process-lifetime configuration, workspace, staging, and
execution services used by every tool and command.

---

## Files

### `execution_context.py`

`ExecutionContext` is a **frozen dataclass** created once by
`CitraApplication` for the life of the process. It captures:

| Property       | Source                                        |
|----------------|-----------------------------------------------|
| `os`           | `platform.system()` (normalized: `darwin`→`macos`) |
| `workspace`    | lifecycle-scoped `WorkspaceContext`            |
| `config`       | supplied `CitraConfig`, or `CITRA_CONFIG_PATH` fallback |
| `model_config` | shortcut → `config.model`                     |
| `web_search_config` | shortcut → `config.web_search`          |
| `sandbox`      | Bubblewrap execution broker                    |
| `filesystem`   | fixed-operation sandbox filesystem client      |
| `lsp_manager`  | persistent language-server manager             |

**Key method:** `has_command(cmd)` — checks whether an executable is on
`PATH` (used by the `bash` tool).

The dataclass uses private (`__`-prefixed) fields set via
`object.__setattr__` in `__post_init__` because frozen dataclasses
prohibit normal assignment.

### `config_loader.py`

Configuration uses frozen dataclasses, including:

- **`ModelConfig`** — model identity, limits, reasoning, and retry policy.
- **`RetryConfig`** — attempts, request timeout, and backoff bounds.
- **`WebSearchConfig`** — `host_url`.
- **`WorkspaceContextConfig`** — source and temporary-root selection.
- **`LspContextConfig`** — enable flag and protocol timeouts.
- **`CitraConfig`** — top-level config, loaded from a TOML file via
  `CitraConfig.load()`. Missing required keys raise `ValueError`.

TOML parsing uses the stdlib `tomllib` module.

### `__init__.py`

Re-exports the public configuration, execution, workspace, and staging types.

---

## Important notes for agents

- `CitraApplication` normally constructs `ExecutionContext` with the one
  lifecycle workspace and shared services. Tests may inject those services.
- The config file path comes from the **`CITRA_CONFIG_PATH`** env var,
  which is set by `start.sh`, unless a parsed config is explicitly supplied.
- `ExecutionContext` is frozen. Long-lived mutable services are referenced by
  it rather than replaced.
- General filesystem tools must use `context.filesystem`; privileged
  materialization/staging are narrow domain brokers, not arbitrary I/O APIs.
