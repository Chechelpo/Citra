from __future__ import annotations

from .config import *

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib

from .config import ModelConfigStore

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


@dataclass(frozen=True)
class LspContextConfig:
    enabled: bool = True
    startup_timeout: float = 30.0
    request_timeout: float = 15.0
    diagnostics_timeout: float = 10.0


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
    """
    Loads the config.toml file wherever it is
    (default: .citra/config.toml). Expects config-template.toml format,
    declared at root.
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
    
    @classmethod
    def load(cls) -> CitraConfig:
        config_path:Path = CitraConfig._config_path()
        
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Citra config file not found: {config_path}"
            )

        with config_path.open("rb") as file:
            raw = tomllib.load(file)

        try:
            model_raw = raw["model"]
            web_search_raw = raw["web-search"]
            workspace_context_raw = raw["workspace"]

            retry_raw = model_raw.get(
                "retry",
                {},
            )

            lsp_raw = raw.get(
                "lsp",
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

            if not isinstance(retry_raw, dict):
                raise ValueError(
                    "'model.retry' must be a TOML table."
                )

            if not isinstance(lsp_raw, dict):
                raise ValueError(
                    "'lsp' must be a TOML table."
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
                config_path=config_path
            )

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
        ) <= 0:
            raise ValueError(
                "All LSP timeout values must be greater than zero."
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
            sandbox=sandbox,
        )
  
    def model(self) -> ModelConfig:
        return self.model_config_store.get()
