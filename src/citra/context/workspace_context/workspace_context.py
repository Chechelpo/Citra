"""Process-lifetime Agent Runtime filesystem and lifecycle ownership.

The controller uses the input project only to create a complete Git-aware
working copy. Model-facing tools operate on that copy as the ordinary current
project. The input path is never exposed as a model-facing path and Citra does
not create an ``@source`` symlink, alias, or mountpoint.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from threading import Lock
from typing import Mapping, Sequence
import venv
from citra.sandbox.sandbox_mode import SandboxMode
from ..environment_fetching import EnvironmentInfo
from ..available_tools import default_runtime_assets, default_tool_definitions, discover_host_runtime
from .runtime import RuntimeAsset, RuntimeProcessSupervisor, RuntimeProvisioning, RuntimeProvisioner, ToolDefinition, write_json_atomic
from .source_baseline import SourceEntry, capture_source_baseline
_PATH_ALIAS_PATTERN = re.compile('^@([a-z_]+)(?:/(.*))?$')
_RUNTIME_DIRECTORY_PATTERN = re.compile('^citra-process-(?P<pid>[1-9][0-9]*)-(?P<nonce>[A-Za-z0-9_-]{6,})$')
logger = logging.getLogger(__name__)
_DEFAULT_PROVISIONING_COPY_BUDGET_BYTES = 64 * 1024 * 1024 * 1024
_DEFAULT_ENV_SOFT_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
_DEFAULT_CACHE_SOFT_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
_DEFAULT_TMP_SOFT_LIMIT_BYTES = 2 * 1024 * 1024 * 1024

class RuntimeClosingError(RuntimeError):
    """New runtime work was requested after shutdown began."""

class RuntimeState(str, Enum):
    """Lifecycle states for one process-lifetime Agent Runtime."""
    NEW = 'new'
    CREATING_FILESYSTEM = 'creating-filesystem'
    MATERIALIZING_WORKSPACE = 'materializing-workspace'
    PROVISIONING_RUNTIME = 'provisioning-runtime'
    BUILDING_ENVIRONMENT = 'building-environment'
    ACTIVE = 'active'
    CLOSING = 'closing'
    CLOSED = 'closed'
    FAILED = 'failed'

class _RuntimeLifecycle:
    """Small thread-safe lifecycle state holder."""

    def __init__(self) -> None:
        """Create a lifecycle in the ``NEW`` state."""
        self._lock = Lock()
        self._state = RuntimeState.NEW

    @property
    def state(self) -> RuntimeState:
        """Return the current lifecycle state under the state lock."""
        with self._lock:
            return self._state

    def set(self, state: RuntimeState) -> None:
        """Set the lifecycle state atomically."""
        with self._lock:
            self._state = state

    def begin_closing(self) -> bool:
        """Transition once to ``CLOSING`` and report whether it changed."""
        with self._lock:
            if self._state in {RuntimeState.CLOSING, RuntimeState.CLOSED}:
                return False
            self._state = RuntimeState.CLOSING
            return True

class AvailablePathAlias(str, Enum):
    """A model-facing Citra path alias, similar to ``~/`` or ``./``."""
    LIBRARY = 'library'
    HOME = 'home'
    TMP = 'tmp'
    CACHE = 'cache'
    CONFIG = 'config'
    DATA = 'data'
    RUNTIME = 'runtime'
    ENV = 'env'

    def as_alias(self) -> str:
        """Return this enum value in model-facing ``@name`` form."""
        return f'@{self.value}'

@dataclass(frozen=True)
class WorkspaceContext:
    """Own one process-lifetime Agent Runtime.

    ``source_workspace`` is controller-private. The model works on
    ``workspace``, which is always a materialized copy under ``root``.

    There is intentionally no direct-source mode and no ``@source`` alias.
    The real source tree is not exposed to model-facing filesystem tools.
    """
    source_workspace: Path
    library: Path
    workspace: Path
    root: Path
    home: Path
    tmp: Path
    cache: Path
    config: Path
    data: Path
    state: Path
    runtime: Path
    env: Path
    metadata: Path
    runtime_state: Path
    provisioning: RuntimeProvisioning
    sandbox_mode: SandboxMode
    private_source_paths: tuple[Path, ...]
    source_baseline: dict[str, SourceEntry] | None
    created_at: str
    workspace_initial_bytes: int
    startup_warnings: tuple[str, ...]
    processes: RuntimeProcessSupervisor
    _lifecycle: _RuntimeLifecycle
    environment_info: EnvironmentInfo
    aggressive_environment_normalization: bool
    environment_overrides: tuple[tuple[str, str], ...]
    env_soft_limit_bytes: int
    cache_soft_limit_bytes: int
    tmp_soft_limit_bytes: int

    @classmethod
    def create(cls, workspace: str | Path, *, temporary_workspace: str | Path | None=None, library: str | Path | None=None, tool_definitions: Sequence[ToolDefinition] | None=None, runtime_assets: Sequence[RuntimeAsset] | None=None, browser_path: str | Path | None=None, sandbox_mode: SandboxMode=SandboxMode.FULL_SANDBOX, provisioning_copy_budget_bytes: int=_DEFAULT_PROVISIONING_COPY_BUDGET_BYTES, remove_stale_process_roots: bool=True, aggressive_environment_normalization: bool=True, environment_overrides: Mapping[str, str] | None=None, env_soft_limit_bytes: int=_DEFAULT_ENV_SOFT_LIMIT_BYTES, cache_soft_limit_bytes: int=_DEFAULT_CACHE_SOFT_LIMIT_BYTES, tmp_soft_limit_bytes: int=_DEFAULT_TMP_SOFT_LIMIT_BYTES) -> WorkspaceContext:
        """Create an isolated runtime and materialize ``workspace`` into it.

        The supplied ``workspace`` path is the real controller-owned source.
        It is copied into ``<runtime-root>/workspace`` before any model-facing
        work begins.

        The input path is never added to ``allowed_roots`` and no source alias
        is created inside the copied project.
        """
        if provisioning_copy_budget_bytes < 0:
            raise ValueError('Provisioning copy budget cannot be negative.')
        if not isinstance(sandbox_mode, SandboxMode):
            raise TypeError('sandbox_mode must be a SandboxMode')
        for name, value in (('env_soft_limit_bytes', env_soft_limit_bytes), ('cache_soft_limit_bytes', cache_soft_limit_bytes), ('tmp_soft_limit_bytes', tmp_soft_limit_bytes)):
            if value < 0:
                raise ValueError(f'{name} cannot be negative.')
        source_workspace = Path(workspace).expanduser().resolve()
        if not source_workspace.is_dir():
            raise NotADirectoryError(f'Source workspace does not exist: {source_workspace}')
        temp_base = Path(temporary_workspace).expanduser().resolve() if temporary_workspace is not None else Path(tempfile.gettempdir()).resolve()
        if cls._is_within(source_workspace, temp_base):
            raise ValueError('The temporary Agent Runtime parent cannot be inside the source workspace.')
        temp_base.mkdir(parents=True, exist_ok=True)
        janitor_warnings: list[str] = []
        if remove_stale_process_roots:
            janitor_warnings.extend(cls.cleanup_stale_roots(temp_base))
        root = Path(tempfile.mkdtemp(prefix=f'citra-process-{os.getpid()}-', dir=str(temp_base))).resolve()
        lifecycle = _RuntimeLifecycle()
        processes = RuntimeProcessSupervisor()
        lifecycle.set(RuntimeState.CREATING_FILESYSTEM)
        workspace_path = root / 'workspace'
        runtime = root / 'runtime'
        env = root / 'env'
        cache = root / 'cache'
        tmp = root / 'tmp'
        home = root / 'home' / 'agent'
        metadata = root / 'metadata'
        config_dir = home / '.config'
        data = home / '.local' / 'share'
        state = metadata
        xdg_state = home / '.local' / 'state'
        runtime_state = home / '.run'
        if library is not None:
            library_path = Path(library).expanduser().resolve()
        else:
            citra_root_raw = os.environ.get('CITRA_ROOT')
            library_path = Path(citra_root_raw).expanduser().resolve() / 'library' if citra_root_raw else temp_base / 'citra-library'
        if cls._is_within(library_path, source_workspace) or cls._is_within(library_path, root) or cls._is_within(root, library_path):
            _remove_tree(root)
            raise ValueError('The controlled document library, source workspace, and Agent Runtime root must not contain one another.')
        private_source_paths = _private_source_exclusions(source_workspace, library_path)
        try:
            library_path.mkdir(parents=True, exist_ok=True)
            for directory in (workspace_path, runtime, env, cache, tmp, home, metadata, config_dir, data, xdg_state, runtime_state):
                directory.mkdir(parents=True, exist_ok=False)
            home.chmod(448)
            runtime_state.chmod(448)
            created_at = datetime.now(timezone.utc).isoformat()
            owner = {'schema_version': 1, 'runtime_id': root.name, 'owner_pid': os.getpid(), 'owner_process_start': _process_start_token(os.getpid()), 'created_at': created_at}
            write_json_atomic(metadata / 'owner.json', owner)
            lifecycle.set(RuntimeState.MATERIALIZING_WORKSPACE)
            copy_warnings, workspace_bytes = _materialize_source_workspace(source_workspace, workspace_path, excluded_roots=private_source_paths)
            try:
                source_baseline = capture_source_baseline(workspace_path)
            except (OSError, RuntimeError, ValueError) as error:
                source_baseline = None
                copy_warnings.append('Source apply baseline is unavailable: ' + str(error))
            lifecycle.set(RuntimeState.PROVISIONING_RUNTIME)
            discovery = discover_host_runtime() if tool_definitions is None or runtime_assets is None else None
            definitions = tuple(default_tool_definitions(mode=sandbox_mode, discovery=discovery) if tool_definitions is None else tool_definitions)
            standalone = tuple(default_runtime_assets(mode=sandbox_mode, discovery=discovery, browser_path=browser_path) if runtime_assets is None else runtime_assets)
            provisioning = RuntimeProvisioner(runtime_root=runtime, copy_budget_bytes=provisioning_copy_budget_bytes, mode=sandbox_mode).provision(definitions, standalone_assets=standalone)
            lifecycle.set(RuntimeState.BUILDING_ENVIRONMENT)
            dependency_warnings = _create_dependency_environment(env, provisioning)
            for directory in (env / 'npm', env / 'npm' / 'bin', env / 'cargo', env / 'cargo' / 'bin', env / 'rustup', env / 'gem', env / 'gem' / 'bin', env / 'go' / 'pkg' / 'mod', env / 'go' / 'bin', cache / 'xdg', cache / 'pip', cache / 'uv', cache / 'ruff', cache / 'npm', cache / 'go-build', cache / 'gradle', cache / 'playwright', cache / 'python'):
                directory.mkdir(parents=True, exist_ok=True)
            warnings = tuple(janitor_warnings + copy_warnings + list(provisioning.warnings) + dependency_warnings)
            instance = cls(source_workspace=source_workspace, library=library_path, workspace=workspace_path, root=root, home=home, tmp=tmp, cache=cache, config=config_dir, data=data, state=state, runtime=runtime, env=env, metadata=metadata, runtime_state=runtime_state, provisioning=provisioning, sandbox_mode=sandbox_mode, private_source_paths=private_source_paths, source_baseline=source_baseline, created_at=created_at, workspace_initial_bytes=workspace_bytes, startup_warnings=warnings, processes=processes, _lifecycle=lifecycle, environment_info=EnvironmentInfo.collect_environment(), aggressive_environment_normalization=aggressive_environment_normalization, environment_overrides=tuple(((str(name), str(value)) for name, value in (environment_overrides or {}).items())), env_soft_limit_bytes=env_soft_limit_bytes, cache_soft_limit_bytes=cache_soft_limit_bytes, tmp_soft_limit_bytes=tmp_soft_limit_bytes)
            lifecycle.set(RuntimeState.ACTIVE)
            instance.write_runtime_manifest(workspace_bytes=workspace_bytes)
            logger.info('Workspace context activated', extra={'origin': __name__, 'runtime_id': instance.runtime_id, 'sandbox_mode': sandbox_mode.name})
            return instance
        except Exception:
            lifecycle.set(RuntimeState.FAILED)
            _remove_tree(root)
            raise

    def python_runtime(self) -> Path:
        """Return the mutable Python dependency environment root."""
        return self.env / 'python'

    def python_bin(self) -> Path:
        """Return the mutable Python dependency environment's bin directory."""
        return self.env / 'python' / 'bin'

    @property
    def runtime_id(self) -> str:
        """Return the process-runtime directory identifier."""
        return self.root.name

    @property
    def lifecycle_state(self) -> RuntimeState:
        """Handle lifecycle state."""
        return self._lifecycle.state

    @property
    def is_closing(self) -> bool:
        """Return whether is closing."""
        return self.lifecycle_state in {RuntimeState.CLOSING, RuntimeState.CLOSED}

    def ensure_active(self) -> None:
        """Reject new runtime work once shutdown has begun."""
        if self.lifecycle_state is not RuntimeState.ACTIVE:
            raise RuntimeClosingError(f'Agent Runtime is not accepting work ({self.lifecycle_state.value}).')

    def begin_closing(self) -> bool:
        """Begin runtime shutdown and stop accepting child processes."""
        began = self._lifecycle.begin_closing()
        self.processes.begin_closing()
        return began

    @property
    def disabled_tool_ids(self) -> frozenset[str]:
        """Return lifecycle-level tool exclusions."""
        return frozenset()

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        """Return roots exposed to ordinary model-facing filesystem tools.

        ``source_workspace``, ``library``, and controller metadata are
        deliberately absent.
        """
        return (self.workspace, self.home, self.tmp, self.cache, self.env, self.runtime)

    @property
    def writable_roots(self) -> tuple[Path, ...]:
        """Return model-facing writable data-plane roots."""
        return (self.workspace, self.home, self.tmp, self.cache, self.env)

    @property
    def runtime_readonly_binds(self) -> tuple[tuple[Path, Path], ...]:
        """Compatibility passthrough for provisioned runtime binds."""
        return self.provisioning.readonly_binds

    def resolve_command(self, command: str) -> Path | None:
        """Resolve a provisioned command while the runtime is active."""
        if self.is_closing:
            return None
        return self.provisioning.resolve_command(command)

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a model-facing path and enforce the allowed-root boundary."""
        raw = str(path)
        alias_raw = raw
        while alias_raw.startswith('./'):
            alias_raw = alias_raw[2:]
        alias_match = _PATH_ALIAS_PATTERN.fullmatch(alias_raw)
        if alias_match:
            alias, remainder = alias_match.groups()
            base = self._alias_root(alias)
            resolved = (base if not remainder else base / remainder).resolve()
        elif raw == '~' or raw.startswith('~/'):
            remainder = '' if raw == '~' else raw[2:]
            resolved = (self.home if not remainder else self.home / remainder).resolve()
        else:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = self.workspace / candidate
            resolved = candidate.resolve()
        self.require_allowed_path(resolved)
        return resolved

    def require_allowed_path(self, path: str | Path) -> Path:
        """Require a path to belong to the model-facing runtime filesystem."""
        resolved = Path(path).resolve()
        if self._is_within(self.library, resolved):
            raise ValueError('Path belongs to the Citra document library and is not accessible through ordinary filesystem tools.')
        if self._is_within(self.source_workspace, resolved):
            raise ValueError('Path belongs to the controller-owned source workspace and is not model-facing.')
        if self.is_controller_private_source_path(resolved):
            raise ValueError('Path belongs to Citra controller configuration and is not model-facing.')
        if any((self._is_within(root, resolved) for root in self.allowed_roots)):
            return resolved
        raise ValueError(f'Path is outside the model-facing filesystem: {resolved}')

    def is_valid_read_path(self, path: str | Path) -> bool:
        """Return whether an ordinary model-facing read may access ``path``."""
        try:
            self.require_allowed_path(path)
            return True
        except ValueError:
            return False

    def is_controller_private_source_path(self, path: str | Path) -> bool:
        """Return whether a path belongs to excluded controller source state."""
        resolved = Path(path).resolve()
        return any((resolved == private or self._is_within(private, resolved) for private in self.private_source_paths))

    def require_writable_path(self, path: str | Path) -> Path:
        """Resolve ``path`` and require it to belong to a writable root."""
        resolved = self.resolve_path(path)
        if any((self._is_within(root, resolved) for root in self.writable_roots)):
            return resolved
        raise ValueError(f'Path is read-only: {self.display_path(resolved)}')

    def display_path(self, path: str | Path) -> str:
        """Render a controller/runtime path using model-facing aliases."""
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(self.workspace)
            return '.' if not relative.parts else relative.as_posix()
        except ValueError:
            pass
        aliases = ((AvailablePathAlias.LIBRARY.value, self.library), (AvailablePathAlias.TMP.value, self.tmp), (AvailablePathAlias.HOME.value, self.home), (AvailablePathAlias.CACHE.value, self.cache), (AvailablePathAlias.ENV.value, self.env), (AvailablePathAlias.CONFIG.value, self.config), (AvailablePathAlias.DATA.value, self.data), (AvailablePathAlias.RUNTIME.value, self.runtime))
        for alias, root in aliases:
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            return f'@{alias}' if not relative.parts else f'@{alias}/{relative.as_posix()}'
        return str(resolved)

    def environment(self, extra: Mapping[str, str] | None=None) -> dict[str, str]:
        """Build the canonical environment shared by runtime processes.

        The input project path is deliberately not exported. Processes receive
        only the copied project root.
        """
        inherited: dict[str, str] = {}
        for name in ('LANG', 'LC_ALL', 'LC_CTYPE', 'TERM', 'COLORTERM', 'TZ', 'SSL_CERT_FILE', 'SSL_CERT_DIR', 'JAVA_HOME', 'GOROOT', 'NODE_PATH'):
            value = os.environ.get(name)
            if value is not None:
                inherited[name] = value
        path_entries = [self.env / 'python' / 'bin', self.env / 'npm' / 'bin', self.env / 'cargo' / 'bin', self.env / 'gem' / 'bin', self.env / 'go' / 'bin', Path('/runtime/bin')]
        path_values = [str(path) for path in path_entries if path == Path('/runtime/bin') or path.exists()]
        inherited['PATH'] = os.pathsep.join(dict.fromkeys(path_values))
        for definition in self.provisioning.definitions.values():
            tool = self.provisioning.tools.get(definition.id)
            if tool is None or not tool.available:
                continue
            for name, value in definition.environment.items():
                inherited[name] = self._expand_environment_value(value)
        if self.aggressive_environment_normalization:
            inherited.update({'PIP_CACHE_DIR': str(self.cache / 'pip'), 'UV_CACHE_DIR': str(self.cache / 'uv'), 'RUFF_CACHE_DIR': str(self.cache / 'ruff'), 'npm_config_cache': str(self.cache / 'npm'), 'npm_config_prefix': str(self.env / 'npm'), 'CARGO_HOME': str(self.env / 'cargo'), 'RUSTUP_HOME': str(Path(os.environ.get('RUSTUP_HOME', str(Path.home() / '.rustup'))).expanduser().absolute()), 'GEM_HOME': str(self.env / 'gem'), 'GEM_PATH': str(self.env / 'gem'), 'GOCACHE': str(self.cache / 'go-build'), 'GOMODCACHE': str(self.env / 'go' / 'pkg' / 'mod'), 'GOBIN': str(self.env / 'go' / 'bin'), 'GRADLE_USER_HOME': str(self.cache / 'gradle'), 'PLAYWRIGHT_BROWSERS_PATH': str(self.provisioning.asset_path('playwright-browsers') or self.cache / 'playwright'), 'PYTHONPYCACHEPREFIX': str(self.cache / 'python')})
        for name, value in self.environment_overrides:
            inherited[name] = self._expand_environment_value(value)
        if extra:
            inherited.update({name: str(value) for name, value in extra.items()})
        inherited.update({'HOME': str(self.home), 'TMPDIR': str(self.tmp), 'TMP': str(self.tmp), 'TEMP': str(self.tmp), 'XDG_CONFIG_HOME': str(self.config), 'XDG_CACHE_HOME': str(self.cache / 'xdg'), 'XDG_DATA_HOME': str(self.data), 'XDG_STATE_HOME': str(self.home / '.local' / 'state'), 'XDG_RUNTIME_DIR': str(self.runtime_state), 'CITRA_PROJECT_ROOT': str(self.workspace), 'CITRA_AGENT_ROOT': str(self.root), 'CITRA_RUNTIME': str(Path('/runtime')), 'CITRA_ENV': str(self.env), 'CITRA_TMP': str(self.tmp), 'CITRA_CACHE': str(self.cache)})
        inherited.pop('CITRA_SOURCE', None)
        inherited.pop('CITRA_LIBRARY', None)
        python_environment = self.env / 'python'
        if python_environment.is_dir():
            inherited['VIRTUAL_ENV'] = str(python_environment)
        return inherited

    def _expand_environment_value(self, value: str) -> str:
        """Expand only model-facing/runtime aliases in an environment value."""
        replacements = {'workspace': self.workspace, 'runtime': self.runtime, 'env': self.env, 'cache': self.cache, 'tmp': self.tmp, 'home': self.home}
        result = value
        for name, path in replacements.items():
            result = result.replace(f'{{{name}}}', str(path))
            result = result.replace(f'${{@{name}}}', str(path))
            alias = f'@{name}'
            if result == alias:
                result = str(path)
            elif result.startswith(alias + '/'):
                result = str(path / result[len(alias) + 1:])
        return result

    def refresh_staged_command(self, command: str) -> Path | None:
        """Resolve a newly installed executable only from mutable env roots."""
        for directory in (self.env / 'python' / 'bin', self.env / 'npm' / 'bin', self.env / 'cargo' / 'bin', self.env / 'gem' / 'bin', self.env / 'go' / 'bin'):
            candidate = directory / command
            if candidate.is_file() and os.access(candidate, os.X_OK):
                self.provisioning.register_staged_command(command, candidate)
                self.write_runtime_manifest()
                return candidate
        return None

    def write_runtime_manifest(self, *, workspace_bytes: int | None=None) -> None:
        """Write controller-only runtime diagnostics and ownership metadata."""
        storage = self.storage_usage()
        payload: dict[str, object] = {'schema_version': 1, 'runtime_id': self.runtime_id, 'sandbox_mode': self.sandbox_mode.name, 'owner_pid': os.getpid(), 'created_at': self.created_at, 'source': str(self.source_workspace), 'workspace': str(self.workspace), 'workspace_mode': 'isolated-copy', 'state': self.lifecycle_state.value, 'active_child_processes': self.processes.active_count, 'workspace_initial_bytes': self.workspace_initial_bytes if workspace_bytes is None else workspace_bytes, 'environment': {'aggressive_normalization': self.aggressive_environment_normalization, 'override_names': [name for name, _ in self.environment_overrides]}, 'storage': storage, 'storage_soft_limits': {'env_bytes': self.env_soft_limit_bytes, 'cache_bytes': self.cache_soft_limit_bytes, 'tmp_bytes': self.tmp_soft_limit_bytes}, 'startup_warnings': list(self.startup_warnings), 'storage_warnings': list(self.soft_limit_warnings()), **self.provisioning.as_manifest()}
        write_json_atomic(self.metadata / 'runtime-manifest.json', payload)

    def storage_usage(self) -> dict[str, int]:
        """Return current usage of Citra-controlled mutable runtime areas."""
        return {'env_bytes': _directory_size(self.env), 'cache_bytes': _directory_size(self.cache), 'tmp_bytes': _directory_size(self.tmp)}

    def require_soft_capacity(self, area: str, *, expected_bytes: int=0) -> None:
        """Guard a Citra-controlled allocation against configured soft limits."""
        self.ensure_active()
        if expected_bytes < 0:
            raise ValueError('Expected allocation size cannot be negative.')
        limits = {'env': self.env_soft_limit_bytes, 'cache': self.cache_soft_limit_bytes, 'tmp': self.tmp_soft_limit_bytes}
        roots = {'env': self.env, 'cache': self.cache, 'tmp': self.tmp}
        if area not in limits:
            raise ValueError(f'Unknown Agent Runtime storage area: {area}')
        used = _directory_size(roots[area])
        limit = limits[area]
        if used + expected_bytes > limit:
            raise RuntimeError(f'Agent Runtime @{area} soft limit would be exceeded: used={used}, requested={expected_bytes}, limit={limit} bytes.')

    def soft_limit_warnings(self) -> tuple[str, ...]:
        """Return human-readable storage soft-limit warnings."""
        usage = self.storage_usage()
        limits = {'env_bytes': self.env_soft_limit_bytes, 'cache_bytes': self.cache_soft_limit_bytes, 'tmp_bytes': self.tmp_soft_limit_bytes}
        return tuple((f"{name.removesuffix('_bytes')} soft limit exceeded: {usage[name]} > {limit} bytes" for name, limit in limits.items() if usage[name] > limit))

    def runtime_diagnostics(self) -> dict[str, object]:
        """Return controller-facing runtime diagnostics."""
        return {'runtime_id': self.runtime_id, 'root': str(self.root), 'workspace': str(self.workspace), 'source': str(self.source_workspace), 'workspace_mode': 'isolated-copy', 'runtime': str(Path('/runtime')), 'dependency_environment': str(self.env), 'state': self.lifecycle_state.value, 'active_child_processes': self.processes.active_count, 'provisioning_budget_bytes': self.provisioning.budget_bytes, 'provisioning_copied_bytes': self.provisioning.copied_bytes, 'aggressive_normalization': self.aggressive_environment_normalization, 'storage': self.storage_usage(), 'tools': self.provisioning.as_manifest()['tools'], 'warnings': [*self.startup_warnings, *self.soft_limit_warnings()]}

    def cleanup(self, *, force: bool=False, preserve_workspace: bool=False) -> None:
        """Terminate children and remove runtime-only state.

        ``preserve_workspace`` keeps the copied project and its Git repository
        for the user. This is used by normal application shutdown because Citra
        never commits on the user's behalf.
        """
        if self.lifecycle_state is RuntimeState.CLOSED:
            return
        self.begin_closing()
        try:
            self.processes.terminate_all(force=force)
        except Exception as error:
            raise RuntimeError(f'Could not terminate children for Agent Runtime {self.root}: {error}') from error
        try:
            if preserve_workspace:
                for child in self.root.iterdir():
                    if child == self.workspace:
                        continue
                    if child.is_dir() and (not child.is_symlink()):
                        _remove_tree(child)
                    else:
                        child.unlink(missing_ok=True)
                marker = self.root / '.citra-workspace'
                marker.write_text('This project checkout is preserved for the user.\n', encoding='utf-8')
            else:
                _remove_tree(self.root)
        except Exception as error:
            raise RuntimeError(f'Could not remove Agent Runtime {self.root}: {error}') from error
        finally:
            if preserve_workspace or not self.root.exists():
                self._lifecycle.set(RuntimeState.CLOSED)

    def write_text_atomic(self, path: str | Path, text: str, *, encoding: str='utf-8') -> Path:
        """Atomically write text inside a model-facing writable root."""
        self.ensure_active()
        destination = self.require_writable_path(path)
        parent = self.require_writable_path(destination.parent)
        parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_raw = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=parent)
        temporary = Path(temporary_raw)
        try:
            with os.fdopen(descriptor, 'w', encoding=encoding) as file:
                file.write(text)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def _alias_root(self, alias: str) -> Path:
        """Resolve one model-facing path alias."""
        aliases: dict[str, Path] = {AvailablePathAlias.HOME.value: self.home, AvailablePathAlias.TMP.value: self.tmp, AvailablePathAlias.CACHE.value: self.cache, AvailablePathAlias.CONFIG.value: self.config, AvailablePathAlias.DATA.value: self.data, AvailablePathAlias.RUNTIME.value: self.runtime, AvailablePathAlias.ENV.value: self.env}
        try:
            return aliases[alias]
        except KeyError as error:
            raise ValueError(f'Unknown workspace path alias: @{alias}') from error

    @staticmethod
    def _is_within(root: Path, path: Path) -> bool:
        """Return whether ``path`` is equal to or below ``root``."""
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    @classmethod
    def cleanup_stale_roots(cls, parent: Path) -> tuple[str, ...]:
        """Conservatively remove roots with valid metadata and a dead owner."""
        warnings: list[str] = []
        try:
            candidates = tuple(parent.iterdir())
        except OSError as error:
            return (f'Could not scan runtime parent {parent}: {error}',)
        for candidate in candidates:
            match = _RUNTIME_DIRECTORY_PATTERN.fullmatch(candidate.name)
            if match is None or candidate.is_symlink() or (not candidate.is_dir()):
                continue
            if (candidate / '.citra-workspace').is_file():
                continue
            owner_path = candidate / 'metadata' / 'owner.json'
            try:
                payload = json.loads(owner_path.read_text(encoding='utf-8'))
                runtime_id = payload['runtime_id']
                owner_pid = payload['owner_pid']
                owner_start = payload.get('owner_process_start')
                if runtime_id != candidate.name or not isinstance(owner_pid, int) or owner_pid <= 0 or (int(match.group('pid')) != owner_pid) or (owner_start is not None and (not isinstance(owner_start, str))):
                    raise ValueError('ownership metadata does not match directory')
            except Exception as error:
                warnings.append(f'Left unverified runtime root {candidate}: {error}')
                continue
            if _process_matches(owner_pid, owner_start):
                continue
            try:
                _remove_tree(candidate)
            except Exception as error:
                warnings.append(f'Could not remove stale runtime {candidate}: {error}')
        return tuple(warnings)

    def resolve_library_path(self, path: str | Path) -> Path:
        """Resolve a dedicated @library path for controller library tools."""
        raw = str(path)
        if raw == '@library':
            return self.library
        prefix = '@library/'
        if not raw.startswith(prefix):
            raise ValueError("Library paths must begin with '@library'.")
        remainder = raw[len(prefix):]
        if not remainder:
            return self.library
        resolved = (self.library / remainder).resolve()
        if not self._is_within(self.library, resolved):
            raise ValueError('Library path escapes @library.')
        return resolved

    def list_library_documents(self, *, location: str='@library', recursive: bool=True) -> tuple[Path, ...]:
        """List controlled Citra documents under @library."""
        directory = self.resolve_library_path(location)
        if not directory.exists():
            return ()
        if not directory.is_dir():
            raise NotADirectoryError(f'Library location is not a directory: {self.display_path(directory)}')
        iterator = directory.rglob('*.citra.xml') if recursive else directory.glob('*.citra.xml')
        documents: list[Path] = []
        for path in iterator:
            if not path.is_file():
                continue
            resolved = path.resolve()
            if self._is_within(self.library, resolved):
                documents.append(resolved)
        return tuple(sorted(documents, key=lambda path: self.display_path(path).casefold()))

def _materialize_source_workspace(source_root: Path, workspace_root: Path, *, excluded_roots: Sequence[Path]=()) -> tuple[list[str], int]:
    """Copy the input project into the model-facing current project.

    The complete project, including version-control metadata, is copied.
    Source-owned regular files and symlinks are copied as project content;
    Citra itself does not add any source bridge or alias.
    """
    warnings: list[str] = []
    total_bytes = 0
    excluded = frozenset(excluded_roots)

    def copy_directory(source: Path, destination: Path, relative: Path) -> None:
        """Handle copy directory."""
        nonlocal total_bytes
        with os.scandir(source) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            child_source = Path(entry.path)
            child_relative = relative / entry.name
            if child_source in excluded:
                warnings.append(f'Skipped controller-private source entry: {child_relative.as_posix()}')
                continue
            child_destination = destination / entry.name
            metadata = entry.stat(follow_symlinks=False)
            mode = metadata.st_mode
            relative_text = child_relative.as_posix()
            if stat.S_ISDIR(mode):
                child_destination.mkdir()
                copy_directory(child_source, child_destination, child_relative)
                shutil.copystat(child_source, child_destination, follow_symlinks=False)
                child_destination.chmod(stat.S_IMODE(metadata.st_mode) | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                continue
            if stat.S_ISLNK(mode):
                child_destination.symlink_to(os.readlink(child_source))
                total_bytes += metadata.st_size
                continue
            if stat.S_ISREG(mode):
                _copy_regular_file(child_source, child_destination)
                total_bytes += metadata.st_size
                continue
            warnings.append(f'Skipped unsupported source entry: {relative_text}')
    copy_directory(source_root, workspace_root, Path())
    return (warnings, total_bytes)

def _private_source_exclusions(source_root: Path, library: Path) -> tuple[Path, ...]:
    """Return controller-owned paths that must never enter the copied source."""
    candidates = [library, source_root / '.citra.logs']
    for name in ('CITRA_ROOT', 'CITRA_CONFIG_PATH'):
        raw = os.environ.get(name)
        if raw:
            candidates.append(Path(raw).expanduser())
    result: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved == source_root:
            continue
        if WorkspaceContext._is_within(source_root, resolved):
            result.append(resolved)
    return tuple(dict.fromkeys(result))

def _copy_regular_file(source: Path, destination: Path) -> None:
    """Use a reflink when available, otherwise perform an ordinary copy."""
    cloned = False
    try:
        import fcntl
        ficlone = 1074041865
        with source.open('rb') as source_stream, destination.open('xb') as target:
            fcntl.ioctl(target.fileno(), ficlone, source_stream.fileno())
        cloned = True
    except (ImportError, OSError):
        destination.unlink(missing_ok=True)
    if cloned:
        shutil.copystat(source, destination, follow_symlinks=False)
    else:
        shutil.copy2(source, destination, follow_symlinks=False)
    destination.chmod(stat.S_IMODE(destination.stat().st_mode) | stat.S_IRUSR | stat.S_IWUSR)

def _create_dependency_environment(env_root: Path, provisioning: RuntimeProvisioning) -> list[str]:
    """Create the shared Python dependency environment when Python exists."""
    warnings: list[str] = []
    if not (provisioning.has_command('python3') or provisioning.has_command('python')):
        return warnings
    destination = env_root / 'python'
    try:
        venv.EnvBuilder(system_site_packages=True, clear=False, symlinks=True, with_pip=False).create(destination)
    except Exception as error:
        warnings.append(f'Could not create shared Python environment: {error}')
    return warnings

def _directory_size(root: Path) -> int:
    """Return an approximate recursive size without following symlinks."""
    total = 0
    if not root.exists():
        return total
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISDIR(metadata.st_mode):
                        stack.append(Path(entry.path))
                    else:
                        total += metadata.st_size
        except OSError:
            continue
    return total

def _process_start_token(pid: int) -> str | None:
    """Read the Linux process start token used to avoid PID-reuse mistakes."""
    try:
        raw = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')
        suffix = raw.rsplit(')', 1)[1].split()
        return suffix[19]
    except (IndexError, OSError):
        return None

def _process_matches(pid: int, expected_start: str | None) -> bool:
    """Return whether a PID still refers to the expected live process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    current_start = _process_start_token(pid)
    if expected_start is not None and current_start is not None:
        return current_start == expected_start
    return True

def _remove_tree(root: Path) -> None:
    """Best-effort permission repair followed by recursive runtime removal."""
    if not root.exists():
        return
    for directory, dirnames, filenames in os.walk(root, topdown=False):
        base = Path(directory)
        for name in filenames:
            path = base / name
            if path.is_symlink():
                continue
            try:
                path.chmod(stat.S_IMODE(path.stat().st_mode) | 384)
            except OSError:
                pass
        for name in dirnames:
            path = base / name
            if path.is_symlink():
                continue
            try:
                path.chmod(stat.S_IMODE(path.stat().st_mode) | 448)
            except OSError:
                pass
    try:
        root.chmod(stat.S_IMODE(root.stat().st_mode) | 448)
    except OSError:
        pass
    shutil.rmtree(root)
