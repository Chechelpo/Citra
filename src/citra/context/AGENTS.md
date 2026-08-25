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

- **`ModelConfig` / `ModelConfigStore`** — named model profiles containing
  provider identity, limits, reasoning, credentials, and retry policy. The
  canonical TOML layout is `[models]` with one persisted `active` profile and
  any number of `[models.<name>]` tables. `CitraConfig.model()` resolves the
  active profile, while `CitraConfig.model(name)` resolves an explicit profile.
  Legacy single `[model]` files remain readable and migrate when a multi-profile
  operation requires the canonical layout.
- **`RetryConfig`** — attempts, request timeout, and backoff bounds.
- **`WebSearchConfig`** — `host_url`.
- **`WorkspaceContextConfig`** — source and temporary-root selection.
- **`LspContextConfig`** — enable flag and protocol timeouts.
- **`BashConfig`** — network permission and Bash defaults.
- **`SubprocessConfig`** — subprocess network permission and output caps.
- **`BrowserConfig`** — Playwright path, timeouts, and unsafe-action policy.
- **`CurlConfig`** — always-allow network, permission, and timeout limits.
- **`NotificationConfig`** — `prompt_bell`.
- **`SandboxContextConfig`** — Bubblewrap sandbox policy (binds, namespaces,
  environment handling).
- **`CitraConfig`** — top-level config, loaded from a TOML file via
  `CitraConfig.load()`. Missing required keys raise `ValueError`; the model,
  web-search, and workspace TOML sections are required.

`CitraConfig.load()` resolves the config path from the **`CITRA_CONFIG_PATH`**
env var and raises `RuntimeError` if it is not defined (Citra is normally
started through `start.sh`). TOML parsing uses the stdlib `tomllib` module.
Various positive-value validations are enforced (LSP/curl/bash/subprocess/
browser limits).

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
