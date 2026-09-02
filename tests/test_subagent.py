"""
Tests for the subagent runtime.

The tests exercise the ``SubagentSupervisor`` against an in-memory
``_FakeAgentRunner`` (instead of the real ``AgentRunner``) so we can
verify the lifecycle, transcript mirror, and guidance plumbing without
invoking a model.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

from citra.tools.subagent.spec import (
    SubagentSpec,
    SubagentStatus,
    TranscriptEntry,
    resolve_write_path,
)
from citra.tools.subagent.guidance import (
    RequestGuidanceTool,
    SubagentGuidanceBridge,
)
from citra.tools.subagent.mode import build_subagent_mode
from citra.tools.subagent.supervisor import (
    ContextBuilder,
    SubagentSupervisor,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class _FakeAgentRunner:
    """Stand-in for ``AgentRunner`` that the supervisor can drive directly."""

    def __init__(
        self,
        context: Any,
        session: Any,
        *,
        api_call: Any,
    ) -> None:
        self.context = context
        self.session = session
        self.api_call = api_call
        self.invoked = False
        self.completed = False
        self.guidance_questions: list[str] = []
        self.steering: list[str] = []

    def run_turn(self) -> None:
        self.invoked = True
        # Drain any pending steering so the session leaves the
        # run_turn loop cleanly.
        for message in self.session.steering.drain():
            self.steering.append(message)
        self.completed = True


class _FakeExecutionContext:
    """Minimal stand-in for ``ExecutionContext``."""

    def __init__(self, workspace: Any) -> None:
        self.workspace = workspace

    def close(self, *, force: bool = False) -> None:  # pragma: no cover
        del force


def _make_workspace(tmp_path: Path) -> Any:
    workspace_root = tmp_path / "ws"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "src").mkdir(parents=True, exist_ok=True)
    (workspace_root / "src" / "module.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    workspace = SimpleNamespace(
        workspace=workspace_root,
        root=runtime_root,
        runtime=runtime_root,
    )
    return workspace


def _make_supervisor(
    tmp_path: Path,
) -> tuple[SubagentSupervisor, Any]:
    workspace = _make_workspace(tmp_path)
    captured: list[_FakeExecutionContext] = []

    def build_context(
        spec: SubagentSpec,
        write_path: Path,
        readonly_binds: tuple[Path, ...],
    ) -> _FakeExecutionContext:
        ctx = _FakeExecutionContext(
            workspace=SimpleNamespace(
                write_path=write_path,
                readonly_binds=readonly_binds,
            ),
        )
        captured.append(ctx)
        return ctx

    api_call: Any = SimpleNamespace()
    supervisor = SubagentSupervisor(
        parent_workspace=workspace,
        parent_root=workspace.runtime,
        api_call=cast(Any, api_call),
    )
    # The real AgentRunner is replaced after construction in tests that
    # want to drive a fake runner.
    supervisor._FakeAgentRunner = _FakeAgentRunner  # type: ignore[attr-defined]
    supervisor._fake_captured_contexts = captured  # type: ignore[attr-defined]
    return supervisor, workspace


def _simple_build_context(
    write_path: Path,
    readonly_binds: tuple[Path, ...] = (),
) -> ContextBuilder:
    """Return a simple ``ContextBuilder`` for the test runner."""

    def _builder(
        spec: SubagentSpec,
        wp: Path,
        rb: tuple[Path, ...],
    ) -> Any:
        return _FakeExecutionContext(
            workspace=SimpleNamespace(
                write_path=wp,
                readonly_binds=rb,
            ),
        )

    return cast(ContextBuilder, _builder)


# ---------------------------------------------------------------------------
# Spec tests
# ---------------------------------------------------------------------------

class TestSubagentSpec:
    def test_task_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError):
            SubagentSpec(task="   ", write_path="x")

    def test_write_path_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError):
            SubagentSpec(task="t", write_path="")

    def test_default_subagent_id_is_derived(self) -> None:
        spec = SubagentSpec(
            task="Implement foo",
            write_path="x",
        )
        assert spec.subagent_id == "implement-foo"

    def test_explicit_subagent_id_is_preserved(self) -> None:
        spec = SubagentSpec(
            task="t",
            write_path="x",
            subagent_id="custom",
        )
        assert spec.subagent_id == "custom"

    def test_blank_subagent_id_is_replaced(self) -> None:
        spec = SubagentSpec(
            task="t",
            write_path="x",
            subagent_id="   ",
        )
        assert spec.subagent_id
        assert spec.subagent_id != "   "

    def test_subagent_id_rejects_path_traversal(self) -> None:
        with pytest.raises(ValueError, match="Subagent id"):
            SubagentSpec(
                task="t",
                write_path="x",
                subagent_id="../../escape",
            )

    def test_readonly_binds_normalized(self) -> None:
        spec = SubagentSpec(
            task="t",
            write_path="x",
            readonly_binds=("a", "", "b", "  c  "),
        )
        assert spec.readonly_binds == ("a", "b", "c")

    def test_network_must_be_bool(self) -> None:
        with pytest.raises(TypeError):
            SubagentSpec(
                task="t",
                write_path="x",
                network="yes",  # type: ignore[arg-type]
            )

    def test_resolve_write_path_creates_missing_dirs(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "new" / "dir"
        resolved = resolve_write_path(tmp_path, "new/dir")
        assert resolved == target.resolve()
        assert resolved.is_dir()

    def test_resolve_write_path_rejects_existing_file(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "regular.txt"
        file_path.write_text("hi", encoding="utf-8")
        with pytest.raises(ValueError):
            resolve_write_path(tmp_path, "regular.txt")

    def test_resolve_write_path_rejects_workspace_escape(
        self,
        tmp_path: Path,
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with pytest.raises(ValueError, match="must remain inside"):
            resolve_write_path(workspace, "../escape")


# ---------------------------------------------------------------------------
# Transcript entry tests
# ---------------------------------------------------------------------------

class TestTranscriptEntry:
    def test_round_trip(self) -> None:
        entry = TranscriptEntry(
            role="user",
            content="hello",
            kind="guidance-request",
        )
        payload = entry.to_json()
        restored = TranscriptEntry.from_json(payload)
        assert restored.role == entry.role
        assert restored.content == entry.content
        assert restored.kind == entry.kind

    def test_jsonl_line_is_one_line(self) -> None:
        entry = TranscriptEntry(
            role="user",
            content="line1\nline2",
        )
        line = entry.to_jsonl_line()
        assert "\n" not in line


def test_request_guidance_uses_context_bridge() -> None:
    calls: list[tuple[str, str]] = []

    class _Supervisor:
        def request_guidance(self, subagent_id: str, question: str) -> str:
            calls.append((subagent_id, question))
            return "continue"

    context = SimpleNamespace(
        config=SimpleNamespace(
            model=lambda: SimpleNamespace(id="test-model"),
        ),
        subagents=SubagentGuidanceBridge(
            subagent_id="worker-1",
            supervisor=_Supervisor(),
        ),
    )
    tool = RequestGuidanceTool(cast(Any, context))

    assert tool.execute({"question": "Which API?"}) == "continue"
    assert calls == [("worker-1", "Which API?")]


def test_subagent_prompt_uses_context_selected_public_tool_names(
    tmp_path: Path,
) -> None:
    context = SimpleNamespace(
        config=SimpleNamespace(
            model=lambda: SimpleNamespace(id="gpt-5-codex"),
        ),
    )
    mode = build_subagent_mode(
        subagent_id="worker-1",
        task="Implement the component",
        write_path=tmp_path,
        readonly_binds=(),
        network=False,
        system_prompt_addendum="",
    )

    prompt = mode.get_system_prompt(cast(Any, context))
    assert "`exec_command`" in prompt
    assert "`request_guidance`" in prompt


# ---------------------------------------------------------------------------
# Supervisor tests
# ---------------------------------------------------------------------------

def _install_run_subagent(
    supervisor: SubagentSupervisor,
) -> None:
    """
    Install a minimal ``_run_subagent`` that drives ``ctx.run_turn()``.

    Tests that need a different runner thread body (steering, guidance,
    long-running completion) define their own ``_run_subagent`` instead.
    """

    def fake_run(
        record: Any,
        build_context: ContextBuilder,
    ) -> None:
        ctx = build_context(
            record.spec,
            record.write_path,
            record.readonly_binds,
        )
        record.context = ctx
        record.session = SimpleNamespace()
        record.status = SubagentStatus.RUNNING
        supervisor._append_entry(  # type: ignore[attr-defined]
            record,
            {
                "role": "user",
                "content": record.spec.task.strip(),
                "kind": "task",
                "subagent_id": record.subagent_id,
            },
        )
        run_turn = getattr(ctx, "run_turn", None)
        if callable(run_turn):
            run_turn()
        record.status = SubagentStatus.COMPLETED
        record.completion.set()

    supervisor._run_subagent = fake_run  # type: ignore[method-assign]


# Backwards-compatible alias.
_run_subagent_with_fake = _install_run_subagent


def test_create_starts_subagent_and_writes_transcript(
    tmp_path: Path,
) -> None:
    supervisor, workspace = _make_supervisor(tmp_path)
    _run_subagent_with_fake(supervisor)

    spec = SubagentSpec(
        task="do a thing",
        write_path="component",
    )
    subagent_id = supervisor.create(
        spec,
        build_context=cast(
            ContextBuilder,
            lambda s, w, r: _FakeExecutionContext(
                workspace=SimpleNamespace(write_path=w),
            ),
        ),
    )
    assert subagent_id == "do-a-thing"
    snapshot = supervisor.snapshot(subagent_id)
    assert snapshot is not None
    # Wait for the fake to finish.
    statuses = supervisor.wait((subagent_id,), timeout=5.0)
    assert statuses[subagent_id] == SubagentStatus.COMPLETED

    snapshot = supervisor.snapshot(subagent_id)
    assert snapshot is not None
    assert snapshot.status == SubagentStatus.COMPLETED
    assert any(
        entry.kind == "task" and entry.content == "do a thing"
        for entry in snapshot.transcript
    )

    transcript_path = (
        workspace.runtime / "subagents" / subagent_id / "transcript.jsonl"
    )
    assert transcript_path.exists()
    contents = transcript_path.read_text(encoding="utf-8")
    assert "do a thing" in contents


def test_supervisor_seeds_worker_task_and_records_runner_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from citra.agent.runner import AgentRunEvent
    import citra.tools.subagent.supervisor as supervisor_module

    supervisor, _ = _make_supervisor(tmp_path)
    seen_messages: list[dict[str, Any]] = []

    class _ObservedRunner:
        def __init__(
            self,
            context: Any,
            session: Any,
            *,
            api_call: Any,
            event_sink: Any,
            render_output: bool,
        ) -> None:
            del context, api_call
            assert render_output is False
            seen_messages.extend(session.get_messages())
            self.event_sink = event_sink

        def run_turn(self) -> None:
            self.event_sink(
                AgentRunEvent(
                    kind="assistant",
                    role="assistant",
                    content="finished component",
                )
            )

    monkeypatch.setattr(supervisor_module, "AgentRunner", _ObservedRunner)

    class _Workspace:
        def cleanup(self, *, force: bool = False) -> None:
            assert force

    context = _FakeExecutionContext(_Workspace())
    subagent_id = supervisor.create(
        SubagentSpec(
            task="implement component",
            write_path="component",
            subagent_id="worker-1",
        ),
        build_context=cast(
            ContextBuilder,
            lambda spec, write_path, binds: context,
        ),
    )

    statuses = supervisor.wait((subagent_id,), timeout=5.0)
    assert statuses[subagent_id] == SubagentStatus.COMPLETED
    assert any(
        message.get("role") == "user"
        and message.get("content") == "implement component"
        for message in seen_messages
    )
    snapshot = supervisor.snapshot(subagent_id)
    assert snapshot is not None
    assert any(
        entry.kind == "assistant"
        and entry.content == "finished component"
        for entry in snapshot.transcript
    )


def test_poll_returns_every_subagent(tmp_path: Path) -> None:
    supervisor, _ = _make_supervisor(tmp_path)
    _run_subagent_with_fake(supervisor)

    ids = [
        supervisor.create(
            SubagentSpec(
                task=f"job-{index}",
                write_path=f"job_{index}",
            ),
            build_context=cast(
                ContextBuilder,
                lambda s, w, r: _FakeExecutionContext(
                    workspace=SimpleNamespace(write_path=w),
                ),
            ),
        )
        for index in range(3)
    ]
    statuses = supervisor.wait(tuple(ids), timeout=5.0)
    assert all(
        statuses[sid] == SubagentStatus.COMPLETED for sid in ids
    )

    snapshots = supervisor.poll()
    seen = {snapshot.subagent_id for snapshot in snapshots}
    assert seen == set(ids)


def test_steer_enqueues_to_session(tmp_path: Path) -> None:
    supervisor, _ = _make_supervisor(tmp_path)

    delivered: list[str] = []
    completion = threading.Event()
    started = threading.Event()

    session_holder: dict[str, Any] = {}

    steering_messages: list[str] = []

    def build_context(
        spec: SubagentSpec,
        write_path: Path,
        readonly_binds: tuple[Path, ...],
    ) -> _FakeExecutionContext:
        def drain() -> list[str]:
            value = list(steering_messages)
            steering_messages.clear()
            return value

        steering_inbox = SimpleNamespace(drain=drain)

        class _Session:
            def __init__(self) -> None:
                self.steering = steering_inbox
                self.memory_enabled = False

            def queue_steering(self, message: str) -> bool:
                steering_messages.append(message)
                return True

        record_session = _Session()
        session_holder["session"] = record_session

        class _Ctx(_FakeExecutionContext):
            def run_turn(self) -> None:  # type: ignore[override]
                started.set()
                while not completion.is_set():
                    for message in record_session.steering.drain():
                        delivered.append(message)
                    time.sleep(0.01)

        return _Ctx(workspace=SimpleNamespace(write_path=write_path))

    def fake_run(
        record: Any,
        build_context: ContextBuilder,
    ) -> None:
        ctx = build_context(
            record.spec,
            record.write_path,
            record.readonly_binds,
        )
        record.context = ctx
        record.session = session_holder["session"]
        record.status = SubagentStatus.RUNNING
        ctx.run_turn()  # type: ignore[attr-defined]
        record.status = SubagentStatus.COMPLETED
        record.completion.set()

    supervisor._run_subagent = fake_run  # type: ignore[method-assign]

    spec = SubagentSpec(task="park", write_path="park")
    subagent_id = supervisor.create(
        spec,
        build_context=cast(ContextBuilder, build_context),
    )

    # Wait until the runner has actually started by polling the
    # supervisor; the very first run_turn will block on the
    # ``completion`` event, so we can then queue steering.
    if not started.wait(timeout=5.0):
        completion.set()
        supervisor.wait((subagent_id,), timeout=5.0)
        pytest.fail("subagent did not start in time")

    assert supervisor.steer(subagent_id, "redirect me")
    assert supervisor.steer(subagent_id, "redirect me again")
    time.sleep(0.2)
    completion.set()
    supervisor.wait((subagent_id,), timeout=5.0)
    assert "redirect me" in delivered
    assert "redirect me again" in delivered


def test_sleep_blocks_until_subagents_finish(tmp_path: Path) -> None:
    supervisor, _ = _make_supervisor(tmp_path)
    _run_subagent_with_fake(supervisor)

    ids = [
        supervisor.create(
            SubagentSpec(
                task=f"job-{index}",
                write_path=f"job_{index}",
            ),
            build_context=_simple_build_context(Path(f"job_{index}")),
        )
        for index in range(2)
    ]
    statuses = supervisor.wait(tuple(ids), timeout=5.0)
    assert all(
        statuses[sid] == SubagentStatus.COMPLETED for sid in ids
    )


def test_sleep_returns_partial_when_timeout_hits(tmp_path: Path) -> None:
    supervisor, _ = _make_supervisor(tmp_path)
    _install_run_subagent(supervisor)

    completion = threading.Event()

    def build_context(
        spec: SubagentSpec,
        write_path: Path,
        readonly_binds: tuple[Path, ...],
    ) -> _FakeExecutionContext:
        class _Ctx(_FakeExecutionContext):
            def run_turn(self) -> None:  # type: ignore[override]
                completion.wait(timeout=5.0)

        return _Ctx(workspace=SimpleNamespace(write_path=write_path))

    subagent_id = supervisor.create(
        SubagentSpec(task="long", write_path="long"),
        build_context=cast(ContextBuilder, build_context),
    )

    statuses = supervisor.wait(
        (subagent_id,),
        timeout=0.2,
    )
    assert statuses[subagent_id] in {
        SubagentStatus.PENDING,
        SubagentStatus.RUNNING,
    }
    completion.set()
    supervisor.wait((subagent_id,), timeout=5.0)


def test_request_guidance_orchestrator_answers(tmp_path: Path) -> None:
    supervisor, _ = _make_supervisor(tmp_path)
    _install_run_subagent(supervisor)
    question_delivered = threading.Event()
    response_returned = threading.Event()
    held_response: dict[str, str | None] = {"value": None}

    def build_context(
        spec: SubagentSpec,
        write_path: Path,
        readonly_binds: tuple[Path, ...],
    ) -> _FakeExecutionContext:
        # Build a minimal ExecutionContext-shaped namespace that the
        # ``RequestGuidanceTool`` would actually call. We just call the
        # supervisor's public ``request_guidance`` here, which is what
        # the tool would do.
        from citra.tools.subagent.supervisor import SubagentSupervisor as _SS

        subagent_id = spec.subagent_id

        class _Ctx(_FakeExecutionContext):
            def run_turn(self) -> None:  # type: ignore[override]
                question_delivered.set()
                response = _SS.request_guidance(
                    supervisor,
                    subagent_id,
                    "Which color?",
                )
                held_response["value"] = response
                response_returned.set()

        return _Ctx(workspace=SimpleNamespace(write_path=write_path))

    subagent_id = supervisor.create(
        SubagentSpec(task="ask", write_path="ask"),
        build_context=cast(ContextBuilder, build_context),
    )

    # Wait for the subagent to actually call request_guidance.
    assert question_delivered.wait(timeout=5.0)

    # The subagent's request is now pending; the orchestrator answers.
    assert supervisor.answer_guidance(subagent_id, "blue")

    assert response_returned.wait(timeout=5.0)
    assert held_response["value"] == "blue"

    # The transcript should contain both the question and the response.
    supervisor.wait((subagent_id,), timeout=5.0)
    snapshot = supervisor.snapshot(subagent_id)
    assert snapshot is not None
    kinds = {entry.kind for entry in snapshot.transcript}
    assert "guidance-request" in kinds
    assert "guidance-response" in kinds
    # After answering, the pending-guidance list must be drained.
    assert snapshot.pending_guidance == ()


def test_close_terminates_running_subagents(tmp_path: Path) -> None:
    supervisor, _ = _make_supervisor(tmp_path)
    completion = threading.Event()

    def build_context(
        spec: SubagentSpec,
        write_path: Path,
        readonly_binds: tuple[Path, ...],
    ) -> _FakeExecutionContext:
        class _Ctx(_FakeExecutionContext):
            def run_turn(self) -> None:
                completion.wait(timeout=5.0)

        return _Ctx(workspace=SimpleNamespace(write_path=write_path))

    subagent_id = supervisor.create(
        SubagentSpec(task="long", write_path="long"),
        build_context=cast(ContextBuilder, build_context),
    )

    # Allow the thread to start.
    time.sleep(0.05)
    supervisor.close()
    completion.set()

    snapshot = supervisor.snapshot(subagent_id)
    assert snapshot is not None
    assert snapshot.status.is_terminal


# ---------------------------------------------------------------------------
# Factory tests: subagent model selection
# ---------------------------------------------------------------------------


def test_subagent_config_pins_subagent_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subagent factory must pin the configured subagent model.

    Without a dedicated subagent profile, the factory still has to
    produce a config that resolves to the orchestrator profile.
    """
    from cryptography.fernet import Fernet

    from citra.context import CitraConfig
    from citra.tools.subagent.factory import _subagent_config
    from citra.tools.subagent.spec import SubagentSpec

    fernet = Fernet.generate_key()
    xdg = tmp_path / "xdg"
    (xdg / "citra").mkdir(parents=True)
    (xdg / "citra" / "encryption.key").write_bytes(fernet + b"\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    alpha_key = Fernet(fernet).encrypt(b"alpha-secret").decode("ascii")
    beta_key = Fernet(fernet).encrypt(b"beta-secret").decode("ascii")
    config_path = tmp_path / "config"
    config_path.mkdir()
    (config_path / "models.toml").write_text(
        f'''\
[models]
orchestrator = "alpha"
subagent = "beta"

[models.alpha]
host = "https://alpha.invalid/v1"
encrypted_key = "{alpha_key}"
id = "alpha-model"
max_input_tokens = 1000
max_output_tokens = 100

[models.beta]
host = "https://beta.invalid/v1"
encrypted_key = "{beta_key}"
id = "beta-model"
max_input_tokens = 2000
max_output_tokens = 200

''',
        encoding="utf-8",
    )
    (config_path / "tools.toml").write_text(
        "[web-search]\nhost_url = 'http://search.invalid'\n",
        encoding="utf-8",
    )
    (config_path / "sandbox.toml").write_text(
        "[sandbox]\nglobal_network_disallow = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CITRA_CONFIG_PATH", str(config_path))

    parent = CitraConfig.load()
    spec = SubagentSpec(task="verify", write_path=str(tmp_path / "out"))

    subagent_cfg = _subagent_config(parent, spec)
    # The subagent config keeps the same store so credentials stay
    # shared, but resolves the implicit model to the dedicated subagent
    # profile.
    assert subagent_cfg.model_config_store is parent.model_config_store
    assert subagent_cfg.model().name == "beta"
    assert subagent_cfg.model().id == "beta-model"
    # The orchestrator's view is unaffected.
    assert parent.model().name == "alpha"

    # When no dedicated subagent profile is set, the factory must not
    # pin a different default, so the orchestrator's profile is used.
    parent.model_config_store.set_subagent("alpha")
    default_cfg = _subagent_config(parent, spec)
    assert default_cfg.model().name == "alpha"
    # And memory/LSP/lint are turned off for the subagent.
    assert default_cfg.memory.enabled is False
    assert default_cfg.lsp.enabled is False
    assert default_cfg.lint.enabled is False


def test_subagent_runtime_forks_resolver_over_parent_assets(
    tmp_path: Path,
) -> None:
    from citra.context.runtime import (
        ProvisionedTool,
        RuntimeProvisioning,
    )
    from citra.tools.subagent.factory import _fork_runtime_provisioning

    runtime = tmp_path / "runtime"
    command = runtime / "bin" / "bash"
    parent_provisioning = RuntimeProvisioning(
        runtime_root=runtime,
        budget_bytes=100,
        copied_bytes=10,
        assets={},
        tools={
            "command:bash": ProvisionedTool(
                id="command:bash",
                commands={"bash": command},
                mode="copy",
            ),
            "staged:pytest": ProvisionedTool(
                id="staged:pytest",
                commands={"pytest": tmp_path / "env" / "pytest"},
                mode="dependency-environment",
            ),
        },
        definitions={},
    )
    parent_workspace = SimpleNamespace(
        runtime=runtime,
        provisioning=parent_provisioning,
    )

    forked = _fork_runtime_provisioning(cast(Any, parent_workspace))

    assert forked.runtime_root == runtime
    assert forked.resolve_command("bash") == command
    assert forked.tools is not parent_provisioning.tools
    assert forked.tools["command:bash"] is not parent_provisioning.tools[
        "command:bash"
    ]
    assert "staged:pytest" not in forked.tools


# Cleanup is automatic because the test runs in a tmp_path; the
# supervisor will not have removed the per-subagent runtime roots
# (because close() did, which is fine).
