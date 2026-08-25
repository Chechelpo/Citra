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
  canonical model file is `.citra/config/models.toml`; its TOML layout is
  `[models]` with one persisted `active` profile and any number of
  `[models.<name>]` tables. `CitraConfig.model()` resolves the active profile,
  while `CitraConfig.model(name)` resolves an explicit profile. Legacy single
  `[model]` files remain readable when `CITRA_CONFIG_PATH` explicitly points at
  an old monolithic config.
- **`RetryConfig`** — attempts, request timeout, and backoff bounds.
- **`WebSearchConfig`** — `host_url`.
- **`WorkspaceContextConfig`** — source and temporary-root selection.
- **`LspContextConfig`** — enable flag and protocol timeouts.
- **`LintContextConfig` / `LintRuleConfig`** — global fallback lint policy.
  Successful `edit`/`write` operations first detect supported lint policy in
  the permanent source project's `pyproject.toml`; when none is detected they
  use these global rules. If neither source nor global policy exists, linting
  is disabled. Policies are never merged. Commands run with network access
  disabled, alongside automatic LSP diagnostics.
- **`BashConfig`** — network permission and Bash defaults.
- **`SubprocessConfig`** — subprocess network permission and output caps.
- **`BrowserConfig`** — Playwright path, timeouts, and unsafe-action policy.
- **`CurlConfig`** — always-allow network, permission, and timeout limits.
- **`NotificationConfig`** — `prompt_bell`.
- **`SandboxContextConfig`** — Bubblewrap sandbox policy (binds, namespaces,
  environment handling).
- **`CitraConfig`** — top-level config assembled from the split configuration
  directory. Missing required keys raise `ValueError`; model, web-search, and
  workspace configuration remain required.

`CitraConfig.load()` resolves **`CITRA_CONFIG_PATH`** from the environment. The
canonical value is the `.citra/config` directory containing exactly these
configuration domains:

| File | Contents |
|------|----------|
| `tools.toml` | `[web-search]`, `[workspace]`, `[browser]`, `[sandbox]`, `[subprocess]`, `[bash]`, `[curl]`, `[lsp]`, `[notifications]`, and future non-model/non-lint operational sections. |
| `models.toml` | `[models]`, its `active` selector, named model profiles, and retry tables. |
| `linting.toml` | Optional global fallback `[lint]` and `[[lint.rules]]`. |

`tools.toml` and `models.toml` are required in the canonical directory layout;
`linting.toml` is optional. A historical single TOML file is still accepted
when `CITRA_CONFIG_PATH` explicitly points to a file. `start.sh` prefers the
split layout and only falls back to the old `.citra/config.toml` with a
migration warning. TOML parsing uses the stdlib `tomllib` module.

### Lint policy precedence

Post-edit linting is gated by the global `[lint].enabled` master switch and
uses one policy source, never a merge. When the switch is enabled:

1. Source-project linting detected from the nearest `@source/pyproject.toml`.
2. Global fallback rules from `.citra/config/linting.toml`.
3. No linting.

Linting is enabled by default. When `lint.enabled = false`, Citra performs no
linting at all, including source-project auto-detection. `lint.enabled = true`
is valid with zero global rules, which enables project-declared linting while
leaving the global fallback empty. If `linting.toml` is absent, this enabled,
empty-fallback behavior is used.

The source project is permanent/read-only to ordinary model tools, so staged
edits cannot weaken the policy that verifies those same edits. Ruff is the
first auto-detected project linter: `[tool.ruff.lint]` enables a per-file
`ruff check` using that `pyproject.toml`; `[tool.ruff.format]` additionally
enables `ruff format --check`. Direct file checks use `--force-exclude` so
project exclusions remain effective.

Global linting remains deliberately command-driven. A typical fallback can
declare:

```toml
[lint]
timeout = 30
max_output_length = 20000

[[lint.rules]]
name = "ruff"
command = [
  "ruff",
  "check",
  "--config",
  "{source}/pyproject.toml",
  "{path}",
]
include = ["**/*.py"]
exclude = ["generated/**"]
cwd = "@source"
```

Linting is enabled by default. `lint.enabled = false` is the master off
switch; `lint.enabled = true` may be used with no `[[lint.rules]]` entries to
permit project-declared linting without a global fallback. Commands are
executed directly as argv, never through a shell, and always with sandbox
networking disabled.

Supported command/cwd placeholders are:

| Placeholder | Meaning |
|-------------|---------|
| `{path}` | Absolute path to the edited file in Citra's writable workspace. |
| `{relative_path}` | Project-relative POSIX path of the edited file. |
| `{workspace}` | Citra's writable project workspace root. |
| `{source}` | Permanent read-only source workspace root. |

The explicit global command is responsible for selecting its lint policy. It
is only used when no supported source-project linter is detected. For example,
a global fallback may invoke Ruff, `npm run lint`, or another toolchain. Lint
rules apply only to files in the editable workspace; scratch paths such as
`@tmp` and the read-only `@source` tree are not linted.

### `__init__.py`

Re-exports the public configuration, execution, workspace, and staging types.

---

## Important notes for agents

- `CitraApplication` normally constructs `ExecutionContext` with the one
  lifecycle workspace and shared services. Tests may inject those services.
- The config directory comes from the **`CITRA_CONFIG_PATH`** env var, which is
  set by `start.sh`, unless a parsed config is explicitly supplied. Canonically
  it is `.citra/config`; direct legacy file paths remain supported.
- `ExecutionContext` is frozen. Long-lived mutable services are referenced by
  it rather than replaced.
- General filesystem tools must use `context.filesystem`; privileged
  materialization/staging are narrow domain brokers, not arbitrary I/O APIs.
