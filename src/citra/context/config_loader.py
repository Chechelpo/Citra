from __future__ import annotations

from .config import *

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from typing import Any

from .config import ModelConfigStore


TOOLS_CONFIG_FILE = "tools.toml"
MODELS_CONFIG_FILE = "models.toml"
LINTING_CONFIG_FILE = "linting.toml"

@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 12
    request_timeout: float = 120.0
    initial_backoff: float = 1.0
    max_backoff: float = 30.0

@dataclass(frozen=True)
class WebSearchConfig:
    host_url: str


@dataclass(frozen=True)
class CurlConfig:
    always_allow_network: bool = False
    permission_timeout: int = 30
    default_timeout: int = 30
    max_timeout: int = 300
    max_output_length: int = 100_000


@dataclass(frozen=True)
class BashConfig:
    always_allow_network: bool = False
    permission_timeout: int = 30


@dataclass(frozen=True)
class SubprocessConfig:
    always_allow_network: bool = False
    permission_timeout: int = 30
    max_output_length: int = 100_000


@dataclass(frozen=True)
class BrowserConfig:
    always_allow_network: bool = False
    permission_timeout: int = 30
    request_timeout: float = 30.0
    browsers_path: str = "~/.cache/ms-playwright"
    enabled_unsafe_actions: tuple[str, ...] = ()
    always_allow_unsafe_actions: bool = False


@dataclass(frozen=True)
class NotificationConfig:
    prompt_bell: bool = True

@dataclass(frozen=True)
class WorkspaceContextConfig:
    temporary_workspace: str | None
    permanent_workspace: str | None
    library: str | None


@dataclass(frozen=True)
class LspContextConfig:
    enabled: bool = True
    startup_timeout: float = 30.0
    request_timeout: float = 15.0
    diagnostics_timeout: float = 10.0
    cold_diagnostics_timeout: float = 45.0
    json_fallback: bool = True


@dataclass(frozen=True)
class LintRuleConfig:
    name: str
    command: tuple[str, ...]
    include: tuple[str, ...] = ("**/*",)
    exclude: tuple[str, ...] = ()
    cwd: str = "."


@dataclass(frozen=True)
class LintContextConfig:
    enabled: bool = True
    timeout: int = 30
    max_output_length: int = 20_000
    rules: tuple[LintRuleConfig, ...] = ()


@dataclass(frozen=True)
class SandboxContextConfig:
    """Operator-controlled Bubblewrap policy with secure defaults."""

    base_readonly_binds: tuple[str, ...] = ("/",)

    masked_host_dirs: tuple[str, ...] = (
        "/home",
        "/root",
        "/run",
        "/tmp",
        "/var/tmp",
        "/mnt",
        "/media",
        "/srv",
        "/boot",
    )

    masked_host_files: tuple[str, ...] = ()

    extra_readonly_binds: tuple[str, ...] = ()
    extra_writable_binds: tuple[str, ...] = ()
    extra_device_binds: tuple[str, ...] = ()

    private_files: tuple[str, ...] = ()

    auto_bind_citra_runtime: bool = True
    auto_bind_citra_config: bool = True

    citra_config_exclude: tuple[str, ...] = (
        "config.toml",
    )

    auto_bind_masked_path_entries: bool = True

    auto_bind_env_paths: tuple[str, ...] = (
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        "VIRTUAL_ENV",
        "JAVA_HOME",
        "GOROOT",
        "NVM_BIN",
        "NVM_DIR",
        "PYTHONPATH",
        "PYENV_ROOT",
        "RUSTUP_HOME",
    )

    auto_bind_resolv_conf_target: bool = True

    unshare_user_try: bool = True
    unshare_pid: bool = True
    unshare_ipc: bool = True
    unshare_uts: bool = True
    unshare_cgroup_try: bool = False

    new_terminal_session: bool = True

    disable_nested_user_namespaces: bool = False

    drop_environment_variables: tuple[str, ...] = (
        "DBUS_SESSION_BUS_ADDRESS",
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "GPG_AGENT_INFO",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    )

    drop_environment_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CitraConfig:
    """Load Citra configuration.

    Canonical configuration is split beneath ``.citra/config`` into
    required ``tools.toml`` and ``models.toml`` files, plus optional
    ``linting.toml`` global lint fallback policy.
    ``CITRA_CONFIG_PATH`` therefore normally points at that directory.

    A path to the historical single-file configuration remains supported for
    backwards compatibility.
    """

    model_config_store: ModelConfigStore
    web_search: WebSearchConfig
    workspace_context: WorkspaceContextConfig

    bash: BashConfig = BashConfig()
    subprocess: SubprocessConfig = SubprocessConfig()
    browser: BrowserConfig = BrowserConfig()
    notifications: NotificationConfig = NotificationConfig()
    curl: CurlConfig = CurlConfig()
    lsp: LspContextConfig = LspContextConfig()
    lint: LintContextConfig = LintContextConfig()
    sandbox: SandboxContextConfig = SandboxContextConfig()

    @classmethod
    def _config_path(cls) -> Path:
        config_path_raw = os.environ.get("CITRA_CONFIG_PATH")

        if not config_path_raw:
            raise RuntimeError(
            "CITRA_CONFIG_PATH is not defined. Citra should be started "
            "through start.sh or supplied a CitraConfig explicitly."
            )

        return Path(
            config_path_raw
        ).resolve()
    
    @staticmethod
    def _read_toml(path: Path) -> dict[str, Any]:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        if not isinstance(raw, dict):
            raise ValueError(f"Citra config must be a TOML table: {path}")
        return raw

    @classmethod
    def _load_raw_config(
        cls,
        config_path: Path,
    ) -> tuple[dict[str, Any], Path]:
        """Return the effective config and the file owned by model storage."""
        if config_path.is_file():
            # Historical one-file layout. Keep this readable so callers that
            # explicitly provide a legacy path do not regress.
            return cls._read_toml(config_path), config_path

        if not config_path.exists():
            raise FileNotFoundError(
                f"Citra config path not found: {config_path}"
            )
        if not config_path.is_dir():
            raise NotADirectoryError(
                f"Citra config path is neither a file nor directory: {config_path}"
            )

        tools_path = config_path / TOOLS_CONFIG_FILE
        models_path = config_path / MODELS_CONFIG_FILE
        linting_path = config_path / LINTING_CONFIG_FILE
        for required in (tools_path, models_path):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Citra config file not found: {required}"
                )

        tools_raw = cls._read_toml(tools_path)
        models_raw = cls._read_toml(models_path)
        linting_raw = (
            cls._read_toml(linting_path)
            if linting_path.is_file()
            else {}
        )

        misplaced_tools = {"model", "models", "lint"}.intersection(tools_raw)
        if misplaced_tools:
            raise ValueError(
                f"{TOOLS_CONFIG_FILE} contains section(s) that belong in "
                f"another config file: {', '.join(sorted(misplaced_tools))}"
            )

        model_keys = set(models_raw)
        if not model_keys or not model_keys.issubset({"model", "models"}):
            unexpected = sorted(model_keys - {"model", "models"})
            detail = (
                f" Unexpected root section(s): {', '.join(unexpected)}."
                if unexpected
                else ""
            )
            raise ValueError(
                f"{MODELS_CONFIG_FILE} must contain only [models] "
                f"(or legacy [model]).{detail}"
            )

        linting_keys = set(linting_raw)
        if not linting_keys.issubset({"lint"}):
            unexpected = sorted(linting_keys - {"lint"})
            raise ValueError(
                f"{LINTING_CONFIG_FILE} must contain only [lint]. "
                f"Unexpected root section(s): {', '.join(unexpected)}"
            )

        merged = dict(tools_raw)
        for source_name, source in (
            (MODELS_CONFIG_FILE, models_raw),
            (LINTING_CONFIG_FILE, linting_raw),
        ):
            duplicate = set(merged).intersection(source)
            if duplicate:
                raise ValueError(
                    f"Duplicate Citra config section(s) while loading "
                    f"{source_name}: {', '.join(sorted(duplicate))}"
                )
            merged.update(source)

        return merged, models_path

    @classmethod
    def load(cls) -> CitraConfig:
        config_path: Path = CitraConfig._config_path()
        raw, models_config_path = cls._load_raw_config(config_path)

        try:
            if "model" not in raw and "models" not in raw:
                raise KeyError("models")
            if "model" in raw and "models" in raw:
                raise ValueError(
                    "Citra config cannot contain both [model] and [models]."
                )

            web_search_raw = raw["web-search"]
            workspace_context_raw = raw["workspace"]

            lsp_raw = raw.get(
                "lsp",
                {},
            )

            lint_raw = raw.get(
                "lint",
                {},
            )

            bash_raw = raw.get(
                "bash",
                {},
            )

            subprocess_raw = raw.get(
                "subprocess",
                {},
            )

            browser_raw = raw.get(
                "browser",
                {},
            )

            notifications_raw = raw.get(
                "notifications",
                {},
            )

            curl_raw = raw.get(
                "curl",
                {},
            )

            sandbox_raw = raw.get(
                "sandbox",
                {},
            )

            if not isinstance(lsp_raw, dict):
                raise ValueError(
                    "'lsp' must be a TOML table."
                )

            if not isinstance(lint_raw, dict):
                raise ValueError(
                    "'lint' must be a TOML table."
                )

            if not isinstance(curl_raw, dict):
                raise ValueError(
                    "'curl' must be a TOML table."
                )

            if not isinstance(bash_raw, dict):
                raise ValueError(
                    "'bash' must be a TOML table."
                )

            if not isinstance(subprocess_raw, dict):
                raise ValueError(
                    "'subprocess' must be a TOML table."
                )

            if not isinstance(browser_raw, dict):
                raise ValueError(
                    "'browser' must be a TOML table."
                )

            if not isinstance(notifications_raw, dict):
                raise ValueError(
                    "'notifications' must be a TOML table."
                )

            if not isinstance(sandbox_raw, dict):
                raise ValueError(
                    "'sandbox' must be a TOML table."
                )

            sandbox_defaults = SandboxContextConfig()
            browser_defaults = BrowserConfig()

            def string_tuple(
                table: dict[str, object],
                *,
                section: str,
                name: str,
                default: tuple[str, ...],
            ) -> tuple[str, ...]:
                value = table.get(
                    name,
                    default,
                )

                if not isinstance(
                    value,
                    (list, tuple),
                ):
                    raise ValueError(
                        f"'{section}.{name}' must be an array of strings."
                    )

                if not all(
                    isinstance(item, str)
                    for item in value
                ):
                    raise ValueError(
                        f"'{section}.{name}' must contain only strings."
                    )

                return tuple(
                    value
                )

            model = ModelConfigStore(
                config_path=models_config_path
            )
            # Validate the active profile at load time so malformed model
            # configuration fails before the application starts.
            model.get()

            web_search = WebSearchConfig(
                host_url=web_search_raw[
                    "host_url"
                ],
            )

            workspace_context = WorkspaceContextConfig(
                temporary_workspace=workspace_context_raw.get(
                    "temporary_workspace"
                ),
                permanent_workspace=workspace_context_raw.get(
                    "permanent_workspace"
                ),
                library= workspace_context_raw.get(
                    "library"
                )
            )

            lsp = LspContextConfig(
                enabled=bool(
                    lsp_raw.get(
                        "enabled",
                        True,
                    )
                ),
                startup_timeout=float(
                    lsp_raw.get(
                        "startup_timeout",
                        30.0,
                    )
                ),
                request_timeout=float(
                    lsp_raw.get(
                        "request_timeout",
                        15.0,
                    )
                ),
                diagnostics_timeout=float(
                    lsp_raw.get(
                        "diagnostics_timeout",
                        10.0,
                    )
                ),
                cold_diagnostics_timeout=float(
                    lsp_raw.get(
                        "cold_diagnostics_timeout",
                        45.0,
                    )
                ),
                json_fallback=bool(
                    lsp_raw.get(
                        "json_fallback",
                        True,
                    )
                ),
            )

            lint_rules_raw = lint_raw.get(
                "rules",
                [],
            )
            if not isinstance(lint_rules_raw, list):
                raise ValueError(
                    "'lint.rules' must be an array of tables."
                )

            lint_rules: list[LintRuleConfig] = []
            lint_rule_names: set[str] = set()
            for index, lint_rule_raw in enumerate(lint_rules_raw):
                section = f"lint.rules[{index}]"
                if not isinstance(lint_rule_raw, dict):
                    raise ValueError(
                        f"'{section}' must be a TOML table."
                    )

                name = lint_rule_raw.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        f"'{section}.name' must be a non-empty string."
                    )
                name = name.strip()
                if name in lint_rule_names:
                    raise ValueError(
                        f"Duplicate lint rule name: {name}"
                    )
                lint_rule_names.add(name)

                command_raw = lint_rule_raw.get("command")
                if (
                    not isinstance(command_raw, list)
                    or not command_raw
                    or not all(
                        isinstance(argument, str) and argument
                        for argument in command_raw
                    )
                ):
                    raise ValueError(
                        f"'{section}.command' must be a non-empty array of non-empty strings."
                    )

                include = string_tuple(
                    lint_rule_raw,
                    section=section,
                    name="include",
                    default=("**/*",),
                )
                if not include:
                    raise ValueError(
                        f"'{section}.include' must contain at least one pattern."
                    )
                if not all(pattern.strip() for pattern in include):
                    raise ValueError(
                        f"'{section}.include' cannot contain empty patterns."
                    )

                exclude = string_tuple(
                    lint_rule_raw,
                    section=section,
                    name="exclude",
                    default=(),
                )
                if not all(pattern.strip() for pattern in exclude):
                    raise ValueError(
                        f"'{section}.exclude' cannot contain empty patterns."
                    )
                cwd = lint_rule_raw.get("cwd", ".")
                if not isinstance(cwd, str) or not cwd.strip():
                    raise ValueError(
                        f"'{section}.cwd' must be a non-empty string."
                    )

                lint_rules.append(
                    LintRuleConfig(
                        name=name,
                        command=tuple(command_raw),
                        include=include,
                        exclude=exclude,
                        cwd=cwd,
                    )
                )

            lint_enabled_raw = lint_raw.get(
                "enabled",
                True,
            )
            if not isinstance(lint_enabled_raw, bool):
                raise ValueError(
                    "'lint.enabled' must be a boolean."
                )

            lint_timeout_raw = lint_raw.get(
                "timeout",
                30,
            )
            if (
                not isinstance(lint_timeout_raw, int)
                or isinstance(lint_timeout_raw, bool)
            ):
                raise ValueError(
                    "'lint.timeout' must be an integer."
                )

            lint_output_limit_raw = lint_raw.get(
                "max_output_length",
                20_000,
            )
            if (
                not isinstance(lint_output_limit_raw, int)
                or isinstance(lint_output_limit_raw, bool)
            ):
                raise ValueError(
                    "'lint.max_output_length' must be an integer."
                )

            lint = LintContextConfig(
                enabled=lint_enabled_raw,
                timeout=lint_timeout_raw,
                max_output_length=lint_output_limit_raw,
                rules=tuple(lint_rules),
            )

            curl = CurlConfig(
                always_allow_network=bool(
                    curl_raw.get(
                        "always_allow_network",
                        False,
                    )
                ),
                permission_timeout=int(
                    curl_raw.get(
                        "permission_timeout",
                        30,
                    )
                ),
                default_timeout=int(
                    curl_raw.get(
                        "default_timeout",
                        30,
                    )
                ),
                max_timeout=int(
                    curl_raw.get(
                        "max_timeout",
                        300,
                    )
                ),
                max_output_length=int(
                    curl_raw.get(
                        "max_output_length",
                        100_000,
                    )
                ),
            )

            bash = BashConfig(
                always_allow_network=bool(
                    bash_raw.get(
                        "always_allow_network",
                        False,
                    )
                ),
                permission_timeout=int(
                    bash_raw.get(
                        "permission_timeout",
                        30,
                    )
                ),
            )

            subprocess_config = SubprocessConfig(
                always_allow_network=bool(
                    subprocess_raw.get(
                        "always_allow_network",
                        False,
                    )
                ),
                permission_timeout=int(
                    subprocess_raw.get(
                        "permission_timeout",
                        30,
                    )
                ),
                max_output_length=int(
                    subprocess_raw.get(
                        "max_output_length",
                        100_000,
                    )
                ),
            )

            browser = BrowserConfig(
                always_allow_network=bool(
                    browser_raw.get(
                        "always_allow_network",
                        browser_defaults.always_allow_network,
                    )
                ),
                permission_timeout=int(
                    browser_raw.get(
                        "permission_timeout",
                        browser_defaults.permission_timeout,
                    )
                ),
                request_timeout=float(
                    browser_raw.get(
                        "request_timeout",
                        browser_defaults.request_timeout,
                    )
                ),
                browsers_path=str(
                    browser_raw.get(
                        "browsers_path",
                        browser_defaults.browsers_path,
                    )
                ),
                enabled_unsafe_actions=string_tuple(
                    browser_raw,
                    section="browser",
                    name="enabled_unsafe_actions",
                    default=browser_defaults.enabled_unsafe_actions,
                ),
                always_allow_unsafe_actions=bool(
                    browser_raw.get(
                        "always_allow_unsafe_actions",
                        browser_defaults.always_allow_unsafe_actions,
                    )
                ),
            )

            notifications = NotificationConfig(
                prompt_bell=bool(
                    notifications_raw.get(
                        "prompt_bell",
                        True,
                    )
                ),
            )

            sandbox = SandboxContextConfig(
                base_readonly_binds=string_tuple(
                    sandbox_raw,
                    section="sandbox",
                    name="base_readonly_binds",
                    default=sandbox_defaults.base_readonly_binds,
                ),
                masked_host_dirs=string_tuple(
                    sandbox_raw,
                    section="sandbox",
                    name="masked_host_dirs",
                    default=sandbox_defaults.masked_host_dirs,
                ),
                masked_host_files=string_tuple(
                    sandbox_raw,
                    section="sandbox",
                    name="masked_host_files",
                    default=sandbox_defaults.masked_host_files,
                ),
                extra_readonly_binds=string_tuple(
                    sandbox_raw,
                    section="sandbox",
                    name="extra_readonly_binds",
                    default=sandbox_defaults.extra_readonly_binds,
                ),
                extra_writable_binds=string_tuple(
                    sandbox_raw,
                    section="sandbox",
                    name="extra_writable_binds",
                    default=sandbox_defaults.extra_writable_binds,
                ),
                extra_device_binds=string_tuple(
                    sandbox_raw,
                    section="sandbox",
                    name="extra_device_binds",
                    default=sandbox_defaults.extra_device_binds,
                ),
                private_files=string_tuple(
                    sandbox_raw,
                    section="sandbox",
                    name="private_files",
                    default=sandbox_defaults.private_files,
                ),
                auto_bind_citra_runtime=bool(
                    sandbox_raw.get(
                        "auto_bind_citra_runtime",
                        sandbox_defaults.auto_bind_citra_runtime,
                    )
                ),
                auto_bind_citra_config=bool(
                    sandbox_raw.get(
                        "auto_bind_citra_config",
                        sandbox_defaults.auto_bind_citra_config,
                    )
                ),
                citra_config_exclude=string_tuple(
                    sandbox_raw,
                    section="sandbox",
                    name="citra_config_exclude",
                    default=sandbox_defaults.citra_config_exclude,
                ),
                auto_bind_masked_path_entries=bool(
                    sandbox_raw.get(
                        "auto_bind_masked_path_entries",
                        sandbox_defaults.auto_bind_masked_path_entries,
                    )
                ),
                auto_bind_env_paths=string_tuple(
                    sandbox_raw,
                    section="sandbox",
                    name="auto_bind_env_paths",
                    default=sandbox_defaults.auto_bind_env_paths,
                ),
                auto_bind_resolv_conf_target=bool(
                    sandbox_raw.get(
                        "auto_bind_resolv_conf_target",
                        sandbox_defaults.auto_bind_resolv_conf_target,
                    )
                ),
                unshare_user_try=bool(
                    sandbox_raw.get(
                        "unshare_user_try",
                        sandbox_defaults.unshare_user_try,
                    )
                ),
                unshare_pid=bool(
                    sandbox_raw.get(
                        "unshare_pid",
                        sandbox_defaults.unshare_pid,
                    )
                ),
                unshare_ipc=bool(
                    sandbox_raw.get(
                        "unshare_ipc",
                        sandbox_defaults.unshare_ipc,
                    )
                ),
                unshare_uts=bool(
                    sandbox_raw.get(
                        "unshare_uts",
                        sandbox_defaults.unshare_uts,
                    )
                ),
                unshare_cgroup_try=bool(
                    sandbox_raw.get(
                        "unshare_cgroup_try",
                        sandbox_defaults.unshare_cgroup_try,
                    )
                ),
                new_terminal_session=bool(
                    sandbox_raw.get(
                        "new_terminal_session",
                        sandbox_defaults.new_terminal_session,
                    )
                ),
                disable_nested_user_namespaces=bool(
                    sandbox_raw.get(
                        "disable_nested_user_namespaces",
                        sandbox_defaults.disable_nested_user_namespaces,
                    )
                ),
                drop_environment_variables=string_tuple(
                    sandbox_raw,
                    section="sandbox",
                    name="drop_environment_variables",
                    default=sandbox_defaults.drop_environment_variables,
                ),
                drop_environment_prefixes=string_tuple(
                    sandbox_raw,
                    section="sandbox",
                    name="drop_environment_prefixes",
                    default=sandbox_defaults.drop_environment_prefixes,
                ),
            )

        except KeyError as error:
            raise ValueError(
                f"Missing required config value: {error.args[0]}"
            ) from error
            
        if min(
            lsp.startup_timeout,
            lsp.request_timeout,
            lsp.diagnostics_timeout,
            lsp.cold_diagnostics_timeout,
        ) <= 0:
            raise ValueError(
                "All LSP timeout values must be greater than zero."
            )

        if min(
            lint.timeout,
            lint.max_output_length,
        ) <= 0:
            raise ValueError(
                "All lint limits must be greater than zero."
            )

        if min(
            curl.permission_timeout,
            curl.default_timeout,
            curl.max_timeout,
            curl.max_output_length,
        ) <= 0:
            raise ValueError(
                "All curl limits must be greater than zero."
            )

        if curl.default_timeout > curl.max_timeout:
            raise ValueError(
                "'curl.default_timeout' cannot exceed 'curl.max_timeout'."
            )

        if bash.permission_timeout <= 0:
            raise ValueError(
                "'bash.permission_timeout' must be greater than zero."
            )

        if min(
            subprocess_config.permission_timeout,
            subprocess_config.max_output_length,
        ) <= 0:
            raise ValueError(
                "All subprocess limits must be greater than zero."
            )

        if min(
            browser.permission_timeout,
            browser.request_timeout,
        ) <= 0:
            raise ValueError(
                "All browser timeout values must be greater than zero."
            )

        if not browser.browsers_path.strip():
            raise ValueError(
                "'browser.browsers_path' cannot be empty."
            )

        return cls(
            model_config_store=model,
            web_search=web_search,
            workspace_context=workspace_context,
            bash=bash,
            subprocess=subprocess_config,
            browser=browser,
            notifications=notifications,
            curl=curl,
            lsp=lsp,
            lint=lint,
            sandbox=sandbox,
        )
  
    def model(self, name: str | None = None) -> ModelConfig:
        return self.model_config_store.get(name)

    def models(self) -> tuple[str, ...]:
        return self.model_config_store.names()
