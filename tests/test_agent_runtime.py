from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from threading import Event, Lock
from types import SimpleNamespace

import pytest
from citra.tools.lsp.installer import execute_install
from citra.tools.lsp.servers import SERVERS

from citra.agent import AgentSession
from citra.agent.runner import _configured_tools
from citra.application import CitraApplication
from citra.cli.repl import HardShutdownRequested, run_turn_with_steering
from citra.config.config_loader import (
    RuntimeConfig,
    RuntimeEnvironmentConfig,
    RuntimeStorageConfig,
    SandboxContextConfig,
    WorkspaceContextConfig,
    _parse_runtime_config,
)
from citra.context.runtime import (
    CopyPolicy,
    RuntimeAsset,
    RuntimeProcessSupervisor,
    RuntimeProvisioner,
    RuntimeProvisionError,
    ToolDefinition,
)
from citra.context.workspace_context import (
    RuntimeClosingError,
    WorkspaceContext,
)
from citra.context.workspace_changes import WorkspaceConflictError
from citra.sandbox import WorkspaceSandbox
from citra.tools.session_memory import TodoTool
from citra.tools.skills.skill_registry import SkillRegistry
from citra.tools.tool_registry import ToolRegistry
from citra.utils.prompt import _memory_guidance, _operating_model, _working_method


@contextmanager
def _runtime(tmp_path: Path, **kwargs: object):
    source = tmp_path / "source"
    parent = tmp_path / "runtimes"
    library = tmp_path / "library"
    source.mkdir(parents=True)
    context = WorkspaceContext.create(
        WorkspaceContextConfig(
            temporary_workspace=str(parent),
            library=str(library),
        ),
        source,
        tool_definitions=kwargs.pop("tool_definitions", ()),
        runtime_assets=kwargs.pop("runtime_assets", ()),
        **kwargs,
    )
    try:
        yield source, context
    finally:
        context.cleanup()


def test_runtime_has_unique_layout_and_complete_startup_copy(tmp_path: Path) -> None:
    source = tmp_path / "project"
    source.mkdir()
    (source / "nested").mkdir()
    (source / "nested" / "code.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "readonly.txt").write_text("locked\n", encoding="utf-8")
    (source / "readonly.txt").chmod(0o444)
    (source / "empty").mkdir()
    (source / "empty").chmod(0o555)
    (source / "link.py").symlink_to("nested/code.py")
    fifo = source / "pipe"
    os.mkfifo(fifo)

    context = WorkspaceContext.create(
        WorkspaceContextConfig(
            temporary_workspace=str(tmp_path / "runtime-parent"),
            library=str(tmp_path / "library"),
        ),
        source,
        tool_definitions=(),
        runtime_assets=(),
    )
    root = context.root
    try:
        assert re.fullmatch(
            rf"citra-process-{os.getpid()}-[A-Za-z0-9_-]{{6,}}",
            context.runtime_id,
        )
        assert (context.workspace / "nested" / "code.py").read_text() == "VALUE = 1\n"
        assert (context.workspace / "empty").is_dir()
        assert (context.workspace / "readonly.txt").stat().st_mode & 0o200
        assert (context.workspace / "empty").stat().st_mode & 0o300 == 0o300
        assert (context.workspace / "link.py").is_symlink()
        assert not (context.workspace / "pipe").exists()
        assert any("unsupported source entry: pipe" in item for item in context.startup_warnings)
        assert context.runtime not in context.writable_roots
        assert context.env in context.writable_roots
        assert context.metadata not in context.allowed_roots
        assert context.runtime.is_dir()
        assert context.env.is_dir()
        assert context.cache.is_dir()
        assert context.tmp.is_dir()
        assert context.home.is_dir()

        first_manifest = json.loads(
            (context.metadata / "runtime-manifest.json").read_text(encoding="utf-8")
        )
        context.write_runtime_manifest()
        refreshed_manifest = json.loads(
            (context.metadata / "runtime-manifest.json").read_text(encoding="utf-8")
        )
        assert refreshed_manifest["created_at"] == first_manifest["created_at"]
        assert refreshed_manifest["workspace_initial_bytes"] == first_manifest[
            "workspace_initial_bytes"
        ]
        assert refreshed_manifest["workspace_initial_bytes"] > 0

        (context.workspace / "nested" / "code.py").write_text(
            "VALUE = 2\n",
            encoding="utf-8",
        )
        assert (source / "nested" / "code.py").read_text() == "VALUE = 1\n"
    finally:
        context.cleanup()
    assert not root.exists()


def test_direct_source_mode_skips_copy_and_writes_authoritative_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project_file = source / "project.txt"
    project_file.write_text("before\n", encoding="utf-8")
    context = WorkspaceContext.create(
        WorkspaceContextConfig(
            temporary_workspace=str(tmp_path / "runtime-parent"),
            library=str(tmp_path / "library"),
            direct_source=True,
        ),
        source,
        tool_definitions=(),
        runtime_assets=(),
    )
    root = context.root
    try:
        assert context.direct_source
        assert context.workspace == source.resolve()
        assert not (root / "workspace").exists()
        assert context.changes is None
        assert context.disabled_tool_ids == frozenset({"commit"})
        assert context.environment()["CITRA_WORKSPACE"] == str(source.resolve())
        assert context.environment()["CITRA_SOURCE"] == str(source.resolve())
        assert context.require_writable_path("@source/project.txt") == project_file

        context.write_text_atomic("project.txt", "after\n")
        assert project_file.read_text(encoding="utf-8") == "after\n"
        diagnostics = context.runtime_diagnostics()
        assert diagnostics["workspace_mode"] == "direct-source"

        sandbox = WorkspaceSandbox(context)
        command = sandbox._build_bwrap_command(
            bwrap="/usr/bin/bwrap",
            command=("true",),
            cwd_path=context.workspace,
            network=False,
            env=context.environment(),
            turn_dirs=sandbox._prepare_lifecycle_directories(),
        )
        triples = tuple(zip(command, command[1:], command[2:]))
        assert ("--bind", str(source.resolve()), str(source.resolve())) in triples
        assert not any(
            option in {"--ro-bind", "--ro-bind-fd"}
            and target == str(source.resolve())
            for option, _source, target in triples
        )
    finally:
        context.cleanup()

    assert not root.exists()
    assert project_file.read_text(encoding="utf-8") == "after\n"


def test_direct_source_mode_filters_workspace_bridge_tools() -> None:
    context = SimpleNamespace(
        workspace=SimpleNamespace(
            disabled_tool_ids=frozenset({"commit"})
        )
    )
    core, deferred = _configured_tools(context)

    assert "read" in core
    assert "commit" not in core
    assert "commit" not in deferred


def test_disabled_memory_is_not_instantiated_or_prompted(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register("todo", TodoTool)
    session = AgentSession(memory_enabled=False)

    assert registry.instantiate(SimpleNamespace(), session) == {}
    assert session.memory.values() == ()
    assert _memory_guidance(False) == ""
    assert "memory" not in _working_method(
        direct_source=True,
        memory_enabled=False,
    ).lower()

    skills = SkillRegistry(
        agent_session=session,
        skills_root=tmp_path / "missing-skills",
        memory_enabled=False,
    )
    assert "task-recognition" not in skills.skills
    assert "sandbox-environment" in skills.skills


def test_direct_source_prompt_explains_immediate_writes() -> None:
    guidance = _operating_model(True)
    assert "authoritative writable project" in guidance
    assert "no Citra Commit or materialization workflow" in guidance


def test_runtime_rejects_unsafe_root_and_library_overlap(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="temporary Agent Runtime parent"):
        WorkspaceContext.create(
            WorkspaceContextConfig(
                temporary_workspace=str(source / "runtime-parent"),
                library=str(tmp_path / "library"),
            ),
            source,
            tool_definitions=(),
            runtime_assets=(),
        )

    library_parent = tmp_path / "controlled"
    nested_source = library_parent / "project"
    nested_source.mkdir(parents=True)
    runtime_parent = tmp_path / "runtimes"
    with pytest.raises(ValueError, match="must not contain"):
        WorkspaceContext.create(
            WorkspaceContextConfig(
                temporary_workspace=str(runtime_parent),
                library=str(library_parent),
            ),
            nested_source,
            tool_definitions=(),
            runtime_assets=(),
        )
    assert not tuple(runtime_parent.glob("citra-process-*"))


def test_controller_configuration_is_not_copied_from_selected_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "project"
    config_root = source / ".citra"
    config_root.mkdir(parents=True)
    (config_root / "config.toml").write_text("api_key='secret'\n", encoding="utf-8")
    (source / "visible.txt").write_text("ok\n", encoding="utf-8")
    monkeypatch.setenv("CITRA_ROOT", str(config_root))
    monkeypatch.setenv("CITRA_CONFIG_PATH", str(config_root / "config.toml"))

    context = WorkspaceContext.create(
        WorkspaceContextConfig(
            temporary_workspace=str(tmp_path / "runtime-parent"),
            library=str(tmp_path / "library"),
        ),
        source,
        tool_definitions=(),
        runtime_assets=(),
    )
    try:
        assert (context.workspace / "visible.txt").is_file()
        assert not (context.workspace / ".citra").exists()
        assert any("controller-private" in item for item in context.startup_warnings)

        sandbox = WorkspaceSandbox(context)
        command = sandbox._build_bwrap_command(
            bwrap="/usr/bin/bwrap",
            command=("true",),
            cwd_path=context.workspace,
            network=False,
            env=context.environment(),
            turn_dirs=sandbox._prepare_lifecycle_directories(),
        )
        pairs = tuple(zip(command, command[1:]))
        assert ("--tmpfs", str(config_root)) in pairs
        assert (
            "--tmpfs",
            str(context.workspace / "@source" / ".citra"),
        ) in pairs
        assert ("--tmpfs", str(context.library)) in pairs
    finally:
        context.cleanup()


def test_direct_source_mode_keeps_controller_configuration_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "project"
    config_root = source / ".citra"
    config_root.mkdir(parents=True)
    config_file = config_root / "config.toml"
    config_file.write_text("api_key='secret'\n", encoding="utf-8")
    monkeypatch.setenv("CITRA_ROOT", str(config_root))
    monkeypatch.setenv("CITRA_CONFIG_PATH", str(config_file))

    context = WorkspaceContext.create(
        WorkspaceContextConfig(
            temporary_workspace=str(tmp_path / "runtime-parent"),
            library=str(tmp_path / "library"),
            direct_source=True,
        ),
        source,
        tool_definitions=(),
        runtime_assets=(),
    )
    try:
        with pytest.raises(ValueError, match="controller configuration"):
            context.resolve_path(".citra/config.toml")

        sandbox = WorkspaceSandbox(context)
        command = sandbox._build_bwrap_command(
            bwrap="/usr/bin/bwrap",
            command=("true",),
            cwd_path=context.workspace,
            network=False,
            env=context.environment(),
            turn_dirs=sandbox._prepare_lifecycle_directories(),
        )
        pairs = tuple(zip(command, command[1:]))
        assert ("--tmpfs", str(config_root)) in pairs
    finally:
        context.cleanup()


def test_copy_budget_is_deterministic_and_falls_back_to_readonly_bind(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"1234")
    second.write_bytes(b"12345678")
    assets = (
        RuntimeAsset(
            id="first",
            source=first,
            destination=PurePosixPath("bin/first"),
            priority=20,
            policy=CopyPolicy.COPY_REQUIRED,
        ),
        RuntimeAsset(
            id="second",
            source=second,
            destination=PurePosixPath("bin/second"),
            priority=10,
            policy=CopyPolicy.COPY_OR_BIND,
            bind_target=second,
        ),
    )
    result = RuntimeProvisioner(
        runtime_root=tmp_path / "runtime",
        copy_budget_bytes=8,
    ).provision((), standalone_assets=assets)

    assert result.copied_bytes == 4
    assert result.assets["first"].mode == "copy"
    assert result.assets["second"].mode == "ro-bind"
    copied = result.assets["first"].runtime_path
    assert copied is not None
    assert copied.read_bytes() == b"1234"
    assert copied.stat().st_mode & 0o222 == 0

    with pytest.raises(RuntimeProvisionError, match="exceeds"):
        RuntimeProvisioner(
            runtime_root=tmp_path / "too-small",
            copy_budget_bytes=3,
        ).provision((), standalone_assets=(assets[0],))

    missing = RuntimeAsset(
        id="missing",
        source=tmp_path / "does-not-exist",
        destination=PurePosixPath("bin/missing"),
        policy=CopyPolicy.COPY_OR_BIND,
    )
    with pytest.raises(RuntimeProvisionError, match="unavailable"):
        RuntimeProvisioner(
            runtime_root=tmp_path / "missing-runtime",
            copy_budget_bytes=8,
        ).provision((), standalone_assets=(missing,))

    unsafe_tree = tmp_path / "unsafe-tree"
    unsafe_tree.mkdir()
    (unsafe_tree / "copied-first").write_text("partial", encoding="utf-8")
    (unsafe_tree / "escape").symlink_to("../outside")
    escaped = RuntimeAsset(
        id="escaped",
        source=unsafe_tree,
        destination=PurePosixPath("unsafe/tree"),
        policy=CopyPolicy.COPY_OR_BIND,
    )
    escaped_result = RuntimeProvisioner(
        runtime_root=tmp_path / "escaped-runtime",
        copy_budget_bytes=1024,
    ).provision((), standalone_assets=(escaped,))
    assert escaped_result.assets["escaped"].mode == "ro-bind"
    assert not (tmp_path / "escaped-runtime" / "unsafe" / "tree").exists()


def test_failed_runtime_health_check_is_not_advertised(tmp_path: Path) -> None:
    executable = tmp_path / "tool"
    executable.write_bytes(b"synthetic executable")
    executable.chmod(0o755)
    asset = RuntimeAsset(
        id="tool-binary",
        source=executable,
        destination=PurePosixPath("bin/tool"),
        policy=CopyPolicy.COPY_REQUIRED,
    )
    definition = ToolDefinition(
        id="tool",
        commands=("tool",),
        assets=(asset,),
        command_assets={"tool": asset.id},
        health_check=("{executable}", "--version"),
    )
    provisioning = RuntimeProvisioner(
        runtime_root=tmp_path / "runtime-health",
        copy_budget_bytes=1024,
    ).provision((definition,))

    class FailedSandbox:
        def run(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(returncode=1, output="broken")

    provisioning.health_check_tools(
        FailedSandbox(),
        cwd=tmp_path,
    )
    assert provisioning.resolve_command("tool") is None
    assert provisioning.tools["tool"].health_detail == "broken"


def test_lsp_install_uses_runtime_sandbox_and_network_policy(tmp_path: Path) -> None:
    definition = SERVERS["pyright"]
    candidate = next(
        item for item in definition.install_candidates if item.manager == "npm"
    )
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class Sandbox:
        def run(self, command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
            calls.append((tuple(command), kwargs))
            return SimpleNamespace(returncode=0, output="installed")

    result = execute_install(
        definition,
        candidate,
        dry_run=False,
        resolver=lambda command: (
            str(tmp_path / "env" / "bin" / command)
            if command == definition.executable
            else None
        ),
        sandbox=Sandbox(),
        cwd=tmp_path,
        environment={"HOME": str(tmp_path / "home")},
    )

    assert result.success
    assert calls[0][0] == candidate.command
    assert calls[0][1]["network"] is True
    assert calls[0][1]["cwd"] == tmp_path
    assert calls[0][1]["environment"] == {"HOME": str(tmp_path / "home")}


def test_environment_precedence_and_reserved_paths(tmp_path: Path) -> None:
    runtime_config = RuntimeConfig(
        environment=RuntimeEnvironmentConfig(
            aggressive_normalization=True,
            overrides=(
                ("PIP_CACHE_DIR", "@env/custom-pip"),
                ("MY_TOOL_MODE", "agent"),
                ("HOME", "/unsafe"),
            ),
        )
    )
    with _runtime(tmp_path, runtime_config=runtime_config) as (_, context):
        environment = context.environment(
            {"HOME": "/also-unsafe", "CITRA_ENV": "/outside"}
        )
        assert environment["PIP_CACHE_DIR"] == str(context.env / "custom-pip")
        assert environment["MY_TOOL_MODE"] == "agent"
        assert environment["HOME"] == str(context.home)
        assert environment["CITRA_ENV"] == str(context.env)
        assert environment["TMPDIR"] == str(context.tmp)

        sandbox = WorkspaceSandbox(
            context,
            config=SimpleNamespace(drop_environment_prefixes=("CITRA_",)),
        )
        sandbox_environment = sandbox._sandbox_environment(
            environment,
            explicit_environment=None,
            turn_dirs=sandbox._prepare_lifecycle_directories(),
        )
        assert sandbox_environment["XDG_CACHE_HOME"] == str(context.cache / "xdg")
        assert sandbox_environment["CITRA_ENV"] == str(context.env)

    disabled = RuntimeConfig(
        environment=RuntimeEnvironmentConfig(
            aggressive_normalization=False,
            overrides=(("MY_TOOL_MODE", "still-agent"),),
        )
    )
    with _runtime(tmp_path / "disabled", runtime_config=disabled) as (_, context):
        environment = context.environment()
        assert "PIP_CACHE_DIR" not in environment
        assert environment["MY_TOOL_MODE"] == "still-agent"


def test_runtime_config_parser_validates_limits_and_reserved_overrides() -> None:
    assert SandboxContextConfig().auto_bind_env_paths == ()
    config = _parse_runtime_config(
        {
            "storage": {"provisioning_copy_budget_bytes": 123},
            "environment": {
                "aggressive_normalization": False,
                "overrides": {"MY_MODE": "@cache/tool"},
            },
            "cleanup": {"remove_stale_process_roots": False},
        }
    )
    assert config.storage.provisioning_copy_budget_bytes == 123
    assert not config.environment.aggressive_normalization
    assert config.environment.overrides == (("MY_MODE", "@cache/tool"),)
    assert not config.cleanup.remove_stale_process_roots
    assert _parse_runtime_config(
        {"storage": {"provisioning_copy_budget_bytes": 0}}
    ).storage.provisioning_copy_budget_bytes == 0

    with pytest.raises(ValueError, match="positive integer"):
        _parse_runtime_config({"storage": {"tmp_soft_limit_bytes": 0}})
    with pytest.raises(ValueError, match="reserved"):
        _parse_runtime_config(
            {"environment": {"overrides": {"CITRA_RUNTIME": "/host"}}}
        )


def test_full_workspace_commit_and_conflict_detection(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "tracked.txt").write_text("one\ntwo\n", encoding="utf-8")
    context = WorkspaceContext.create(
        WorkspaceContextConfig(
            temporary_workspace=str(tmp_path / "runtime-parent"),
            library=str(tmp_path / "library"),
        ),
        source,
        tool_definitions=(),
        runtime_assets=(),
    )
    try:
        (context.workspace / "tracked.txt").write_text(
            "ONE\ntwo\n",
            encoding="utf-8",
        )
        context.changes.stage(("tracked.txt",))
        context.changes.apply()
        assert (source / "tracked.txt").read_text() == "ONE\ntwo\n"

        (context.workspace / "tracked.txt").write_text(
            "agent\n",
            encoding="utf-8",
        )
        context.changes.stage(("tracked.txt",))
        (source / "tracked.txt").write_text("external\n", encoding="utf-8")
        with pytest.raises(WorkspaceConflictError):
            context.changes.apply()
        assert (source / "tracked.txt").read_text() == "external\n"
    finally:
        context.cleanup()


def test_commit_rejects_parent_symlink_escape(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    context = WorkspaceContext.create(
        WorkspaceContextConfig(
            temporary_workspace=str(tmp_path / "runtime-parent"),
            library=str(tmp_path / "library"),
        ),
        source,
        tool_definitions=(),
        runtime_assets=(),
    )
    try:
        (context.workspace / "nested" / "new.txt").write_text(
            "agent\n",
            encoding="utf-8",
        )
        context.changes.stage(("nested/new.txt",))
        (source / "nested").rmdir()
        (source / "nested").symlink_to(outside, target_is_directory=True)

        with pytest.raises(WorkspaceConflictError, match="parent now escapes"):
            context.changes.apply()
        assert not (outside / "new.txt").exists()
    finally:
        context.cleanup()


def test_stale_janitor_removes_only_verified_dead_owner(tmp_path: Path) -> None:
    stale = tmp_path / "citra-process-99999999-deadbeef"
    (stale / "metadata").mkdir(parents=True)
    (stale / "metadata" / "owner.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime_id": stale.name,
                "owner_pid": 99999999,
                "owner_process_start": None,
            }
        ),
        encoding="utf-8",
    )
    unverified = tmp_path / "citra-process-99999998-unknown"
    unverified.mkdir()

    warnings = WorkspaceContext.cleanup_stale_roots(tmp_path)
    assert not stale.exists()
    assert unverified.exists()
    assert any("unverified runtime root" in warning for warning in warnings)


def test_sandbox_command_uses_explicit_agent_runtime_mounts(tmp_path: Path) -> None:
    os_asset = RuntimeAsset(
        id="usr",
        source=Path("/usr"),
        destination=PurePosixPath("compat/usr"),
        policy=CopyPolicy.BIND_ONLY,
        bind_target=Path("/usr"),
    )
    with _runtime(tmp_path, runtime_assets=(os_asset,)) as (_, context):
        sandbox = WorkspaceSandbox(context)
        directories = sandbox._prepare_lifecycle_directories()
        command = sandbox._build_bwrap_command(
            bwrap="/usr/bin/bwrap",
            command=("/usr/bin/true",),
            cwd_path=context.workspace,
            network=False,
            env=context.environment(),
            turn_dirs=directories,
        )
        triples = tuple(zip(command, command[1:], command[2:]))
        pairs = tuple(zip(command, command[1:]))
        assert ("--ro-bind", "/", "/") not in triples
        assert ("--tmpfs", str(context.root)) in pairs
        assert ("--ro-bind", str(context.runtime), str(context.runtime)) in triples
        for writable in (
            context.workspace,
            context.env,
            context.cache,
            context.home,
            context.tmp,
        ):
            assert ("--bind", str(writable), str(writable)) in triples
        assert ("--bind", str(context.runtime), str(context.runtime)) not in triples
        assert str(context.metadata) not in command
        assert ("--ro-bind", str(context.source_workspace), str(context.source_workspace)) in triples

        unsafe_sandbox = WorkspaceSandbox(
            context,
            config=SimpleNamespace(extra_readonly_binds=(str(context.root),)),
        )
        with pytest.raises(RuntimeError, match="controller metadata"):
            unsafe_sandbox._build_bwrap_command(
                bwrap="/usr/bin/bwrap",
                command=("true",),
                cwd_path=context.workspace,
                network=False,
                env=context.environment(),
                turn_dirs=unsafe_sandbox._prepare_lifecycle_directories(),
            )

        context.begin_closing()
        with pytest.raises(RuntimeClosingError):
            sandbox.run(("true",), timeout=1, network=False)


def test_second_interrupt_requests_hard_shutdown_without_waiting_for_worker() -> None:
    worker_release = Event()
    queued: list[str] = []
    hard_shutdowns: list[float] = []

    class Input:
        calls = 0

        def prompt_until(self, *_: object, **__: object) -> None:
            self.calls += 1
            raise KeyboardInterrupt

    application = SimpleNamespace(
        run_agent_turn=lambda: worker_release.wait(5),
        interactions=SimpleNamespace(take=lambda: None, has_pending=lambda: False),
        session=SimpleNamespace(queue_steering=lambda text: queued.append(text) or True),
        request_hard_shutdown=lambda: hard_shutdowns.append(time.monotonic()),
    )
    started = time.monotonic()
    try:
        with pytest.raises(HardShutdownRequested):
            run_turn_with_steering(application, input_service=Input())
        assert time.monotonic() - started < 1.0
        assert len(queued) == 1
        assert len(hard_shutdowns) == 1
    finally:
        worker_release.set()


def test_hard_application_close_removes_runtime_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _runtime(tmp_path) as (_, context):
        root = context.root
        force_values: list[bool] = []
        application = CitraApplication.__new__(CitraApplication)
        application._close_lock = Lock()
        application._closing = False
        application._closed = False
        application._hard_shutdown = Event()
        application.workspace = context
        application.interactions = SimpleNamespace(close=lambda: None)
        application.context = SimpleNamespace(
            close=lambda *, force=False: force_values.append(force)
        )
        application.session = object()
        monkeypatch.setattr(
            "citra.application.TOOL_REGISTRY.release_session",
            lambda _session: None,
        )

        application.request_hard_shutdown()

        assert application.hard_shutdown_requested
        assert force_values == [True]
        assert not root.exists()


def test_aggregate_process_supervisor_uses_bounded_termination() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    supervisor = RuntimeProcessSupervisor()
    assert supervisor.register(process)
    started = time.monotonic()
    supervisor.terminate_all(force=True)
    assert time.monotonic() - started < 1.0
    process.wait(timeout=1)
    assert process.poll() is not None


def test_controlled_allocations_honor_soft_limits(tmp_path: Path) -> None:
    config = RuntimeConfig(
        storage=RuntimeStorageConfig(
            provisioning_copy_budget_bytes=1024,
            env_soft_limit_bytes=1,
            cache_soft_limit_bytes=1,
            tmp_soft_limit_bytes=1,
        )
    )
    with _runtime(tmp_path, runtime_config=config) as (_, context):
        (context.env / "payload").write_bytes(b"xx")
        with pytest.raises(RuntimeError, match="soft limit"):
            context.require_soft_capacity("env")
        assert context.soft_limit_warnings()
