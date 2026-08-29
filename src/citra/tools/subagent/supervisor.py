"""
Subagent lifecycle owner.

``SubagentSupervisor`` is the orchestrator-side process-lifetime service
that:

  * spawns a new subagent for every ``SubagentTool.create`` call;
  * runs the subagent on a dedicated daemon thread that drives
    ``AgentRunner.run_turn`` against a nested ``ExecutionContext``;
  * owns the shared transcript (in-memory mirror + on-disk JSONL file)
    and the guidance inbox;
  * signals completion through a per-subagent ``Event``;
  * cleans up the subagent's runtime root when the supervisor is closed.

The supervisor is owned by the orchestrator's ``ExecutionContext`` and
its lifecycle is bound to the surrounding ``WorkspaceContext``: closing
the supervisor terminates every running subagent and removes every
subagent runtime root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import logging
import shutil
import threading
from typing import Any, Callable
import uuid

from citra.agent import AgentSession
from citra.agent.runner import AgentRunner, ApiCall
from citra.context import ExecutionContext

from .spec import (
    SubagentSpec,
    SubagentStatus,
    SubagentSnapshot,
    TranscriptEntry,
    resolve_write_path,
    TRANSCRIPT_ENTRY_BUDGET,
)


logger = logging.getLogger(__name__)


# Default message returned to a subagent whose guidance request was never
# answered (typically because the orchestrator cancelled it).
_NO_RESPONSE_DEFAULT = (
    "[no response: the orchestrator did not answer before the subagent "
    "exited. Continue with your best judgment or surface the question in "
    "your final summary.]"
)


# ---------------------------------------------------------------------------
# Internal records
# ---------------------------------------------------------------------------

@dataclass
class _GuidanceWaiter:
    """Single in-flight request from a subagent, waiting for the orchestrator."""

    question: str
    response_event: threading.Event = field(default_factory=threading.Event)
    response: str | None = None

    def answer(
        self,
        response: str,
    ) -> bool:
        if self.response_event.is_set():
            return False
        self.response = response
        self.response_event.set()
        return True


@dataclass
class _SubagentRecord:
    """All per-subagent state owned by the supervisor."""

    subagent_id: str
    spec: SubagentSpec
    runtime_root: Path
    transcript_path: Path
    write_path: Path
    readonly_binds: tuple[Path, ...]
    status: SubagentStatus = SubagentStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    transcript: list[TranscriptEntry] = field(default_factory=list)
    pending_guidance: list[TranscriptEntry] = field(default_factory=list)
    pending_lock: threading.Lock = field(default_factory=threading.Lock)
    completion: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    workspace: Any = None
    context: ExecutionContext | None = None
    session: AgentSession | None = None
    guidance_waiters: dict[str, _GuidanceWaiter] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

ContextBuilder = Callable[
    [SubagentSpec, Path, tuple[Path, ...]],
    ExecutionContext,
]


class SubagentSupervisor:
    """
    Owns the orchestrator-side subagent lifecycle.

    One supervisor is constructed per orchestrator session and is
    reachable through ``ExecutionContext.subagents``.
    """

    def __init__(
        self,
        *,
        parent_workspace: Any,
        parent_root: Path,
        api_call: ApiCall,
    ) -> None:
        self.__parent_workspace = parent_workspace
        self.__parent_root = Path(parent_root)
        self.__api_call = api_call

        self.__records: dict[str, _SubagentRecord] = {}
        self.__records_lock = threading.Lock()
        self.__closed = False
        self.__close_lock = threading.Lock()

        self.__subagents_root = self.__parent_root / "subagents"
        self.__subagents_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_closed(self) -> bool:
        return self.__closed

    def create(
        self,
        spec: SubagentSpec,
        *,
        build_context: ContextBuilder,
    ) -> str:
        """
        Spawn a new subagent.

        ``build_context`` is the orchestrator's context builder: it is
        responsible for creating a nested ``WorkspaceContext`` and a
        constrained ``ExecutionContext`` for the new subagent. Splitting
        this out keeps the supervisor free of ``WorkspaceContext`` and
        ``Mode`` construction details.
        """
        with self.__close_lock:
            if self.__closed:
                raise RuntimeError(
                    "Subagent supervisor is closed."
                )

        subagent_id = spec.subagent_id

        with self.__records_lock:
            if subagent_id in self.__records:
                existing = self.__records[subagent_id]
                if not existing.status.is_terminal:
                    raise ValueError(
                        f"Subagent '{subagent_id}' is already running."
                    )
                # A previous subagent with this id has already finished;
                # remove its old runtime root so the new subagent gets a
                # clean slate.
                self._dispose_record(
                    existing,
                    force=True,
                )
                self.__records.pop(subagent_id, None)

            runtime_root = self.__subagents_root / subagent_id
            if runtime_root.exists():
                shutil.rmtree(
                    runtime_root,
                    ignore_errors=True,
                )
            runtime_root.mkdir(parents=True, exist_ok=True)
            transcript_path = runtime_root / "transcript.jsonl"

            write_path = resolve_write_path(
                self.__parent_workspace.workspace,
                spec.write_path,
            )

            readonly_binds: list[Path] = []
            for raw in spec.readonly_binds:
                candidate = Path(raw).expanduser().resolve()
                if candidate.exists():
                    readonly_binds.append(candidate)

            record = _SubagentRecord(
                subagent_id=subagent_id,
                spec=spec,
                runtime_root=runtime_root,
                transcript_path=transcript_path,
                write_path=write_path,
                readonly_binds=tuple(readonly_binds),
            )
            self.__records[subagent_id] = record

        thread = threading.Thread(
            target=self._run_subagent,
            args=(record, build_context),
            name=f"citra-subagent-{subagent_id}",
            daemon=True,
        )
        record.thread = thread
        thread.start()
        return subagent_id

    def poll(
        self,
        *,
        include_completed: bool = True,
    ) -> tuple[SubagentSnapshot, ...]:
        """Return one snapshot per subagent, including pending guidance."""
        with self.__records_lock:
            records = list(self.__records.values())

        snapshots: list[SubagentSnapshot] = []
        for record in records:
            if not include_completed and record.status.is_terminal:
                continue
            snapshots.append(self._snapshot(record))
        return tuple(snapshots)

    def snapshot(
        self,
        subagent_id: str,
    ) -> SubagentSnapshot | None:
        with self.__records_lock:
            record = self.__records.get(subagent_id)
        if record is None:
            return None
        return self._snapshot(record)

    def steer(
        self,
        subagent_id: str,
        message: str,
    ) -> bool:
        """
        Send ``message`` to the subagent's main session as a steering
        instruction.

        Returns ``True`` if the message was enqueued, ``False`` if the
        subagent is not known or has already finished.
        """
        message = (message or "").strip()
        if not message:
            return False

        with self.__records_lock:
            record = self.__records.get(subagent_id)
        if record is None or record.status.is_terminal:
            return False

        session = record.session
        if session is None:
            return False

        session.queue_steering(message)
        return True

    def answer_guidance(
        self,
        subagent_id: str,
        response: str,
    ) -> bool:
        """
        Answer the oldest pending guidance request for ``subagent_id``.

        Returns ``True`` if a waiter was woken up.
        """
        response = (response or "").strip()
        if not response:
            return False

        with self.__records_lock:
            record = self.__records.get(subagent_id)
        if record is None:
            return False

        with record.pending_lock:
            waiters = list(record.guidance_waiters.items())
        if not waiters:
            return False
        # FIFO: answer the oldest request.
        waiters.sort(key=lambda item: item[0])
        waiter_id, waiter = waiters[0]
        answered = waiter.answer(response)
        if not answered:
            return False
        with record.pending_lock:
            record.guidance_waiters.pop(waiter_id, None)
        # Pop the matching pending-guidance transcript entry so the
        # orchestrator's next ``poll`` no longer surfaces a request it
        # has already answered.
        record.pending_guidance = [
            entry
            for entry in record.pending_guidance
            if not _entry_matches_question(
                entry,
                waiter.question,
            )
        ]

        self._append_entry(
            record,
            {
                "role": "assistant",
                "content": (
                    f"[orchestrator -> subagent:{subagent_id}]\n\n{response}"
                ),
                "kind": "guidance-response",
                "subagent_id": subagent_id,
            },
        )
        return True

    def request_guidance(
        self,
        subagent_id: str,
        question: str,
    ) -> str:
        """
        Subagent-side entry point: ask the orchestrator a question and
        block until it answers.

        The call appends a ``guidance-request`` transcript entry, parks
        the caller on a ``threading.Event``, and returns the orchestrator's
        response (or a default message if the subagent is cancelled
        before the orchestrator can answer).
        """
        with self.__records_lock:
            record = self.__records.get(subagent_id)
        if record is None:
            return ""
        if record.status.is_terminal:
            return ""

        waiter = _GuidanceWaiter(question=question)
        with record.pending_lock:
            record.guidance_waiters[uuid.uuid4().hex] = waiter

        self._append_entry(
            record,
            {
                "role": "user",
                "content": (
                    f"[subagent:{subagent_id}] asks:\n\n{question}"
                ),
                "kind": "guidance-request",
                "subagent_id": subagent_id,
            },
        )

        try:
            waiter.response_event.wait()
        finally:
            with record.pending_lock:
                for key, existing in list(
                    record.guidance_waiters.items()
                ):
                    if existing is waiter:
                        record.guidance_waiters.pop(key, None)
                        break

        return waiter.response or _NO_RESPONSE_DEFAULT

    def wait(
        self,
        subagent_ids: tuple[str, ...],
        *,
        timeout: float | None = None,
    ) -> dict[str, SubagentStatus]:
        """
        Block until every named subagent is in a terminal state.

        ``timeout`` is in seconds. ``None`` blocks indefinitely. The
        returned map is keyed by the requested subagent ids and reports
        the status observed when the call returned (``COMPLETED``,
        ``FAILED`` or ``CANCELLED`` for finished subagents).
        """
        if not subagent_ids:
            return {}

        events: list[tuple[str, threading.Event]] = []
        with self.__records_lock:
            for sid in subagent_ids:
                record = self.__records.get(sid)
                if record is None:
                    continue
                events.append((sid, record.completion))

        if not events:
            return {
                sid: SubagentStatus.FAILED
                for sid in subagent_ids
            }

        if timeout is None:
            for _, event in events:
                event.wait()
        else:
            for _, event in events:
                if not event.wait(timeout=timeout):
                    break

        result: dict[str, SubagentStatus] = {}
        with self.__records_lock:
            for sid in subagent_ids:
                record = self.__records.get(sid)
                if record is None:
                    result[sid] = SubagentStatus.FAILED
                else:
                    result[sid] = record.status
        return result

    def close(self) -> None:
        """Cancel every running subagent and drop the supervisor state."""
        with self.__close_lock:
            if self.__closed:
                return
            self.__closed = True

        with self.__records_lock:
            records = list(self.__records.values())
        for record in records:
            self._cancel(record)

        # Best-effort wait for the subagent threads to drain. We do not
        # raise here: the orchestrator is shutting down, and ``force=True``
        # already terminated each subagent's runtime.
        for record in records:
            thread = record.thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=0.5)

        for record in records:
            self._dispose_record(record, force=True)

    # ------------------------------------------------------------------
    # Internal: subagent thread
    # ------------------------------------------------------------------

    def _run_subagent(
        self,
        record: _SubagentRecord,
        build_context: ContextBuilder,
    ) -> None:
        context: ExecutionContext | None = None
        try:
            context = build_context(
                record.spec,
                record.write_path,
                record.readonly_binds,
            )
            record.context = context
            record.session = AgentSession(
                memory_enabled=False,
            )
            record.status = SubagentStatus.RUNNING
            record.started_at = datetime.now(timezone.utc).isoformat()

            self._append_entry(
                record,
                {
                    "role": "user",
                    "content": record.spec.task.strip(),
                    "kind": "task",
                    "subagent_id": record.subagent_id,
                },
            )

            runner = AgentRunner(
                context,
                record.session,
                api_call=self.__api_call,
            )
            runner.run_turn()
            record.status = SubagentStatus.COMPLETED
        except Exception as error:
            record.error = (
                f"{type(error).__name__}: {error}"
            )
            record.status = SubagentStatus.FAILED
            logger.exception(
                "Subagent %s failed: %s",
                record.subagent_id,
                record.error,
            )
        finally:
            record.finished_at = datetime.now(timezone.utc).isoformat()
            self._append_entry(
                record,
                {
                    "role": "system",
                    "content": (
                        f"subagent {record.subagent_id} finished "
                        f"with status={record.status.value}"
                    ),
                    "kind": "lifecycle",
                    "subagent_id": record.subagent_id,
                },
            )
            with record.pending_lock:
                for waiter in record.guidance_waiters.values():
                    if not waiter.response_event.is_set():
                        waiter.answer(
                            "[no response: subagent exited before the "
                            "orchestrator could answer]"
                        )
                        waiter.response_event.set()
                record.guidance_waiters.clear()
            record.completion.set()
            if context is not None:
                try:
                    context.close(force=True)
                except Exception:
                    logger.exception(
                        "Failed to close subagent %s context",
                        record.subagent_id,
                    )

    def _cancel(
        self,
        record: _SubagentRecord,
    ) -> None:
        if record.status.is_terminal:
            return
        record.status = SubagentStatus.CANCELLED
        record.error = record.error or "cancelled by orchestrator shutdown"
        context = record.context
        if context is not None:
            try:
                context.workspace.begin_closing()
            except Exception:
                logger.exception(
                    "Failed to mark subagent %s workspace as closing",
                    record.subagent_id,
                )

    def _dispose_record(
        self,
        record: _SubagentRecord,
        *,
        force: bool,
    ) -> None:
        runtime_root = record.runtime_root
        if force and runtime_root.exists():
            try:
                shutil.rmtree(
                    runtime_root,
                    ignore_errors=True,
                )
            except Exception:
                logger.exception(
                    "Failed to remove subagent runtime %s",
                    runtime_root,
                )

    # ------------------------------------------------------------------
    # Internal: transcript + guidance plumbing
    # ------------------------------------------------------------------

    def _snapshot(
        self,
        record: _SubagentRecord,
    ) -> SubagentSnapshot:
        with record.pending_lock:
            guidance = list(record.pending_guidance)
        # ``transcript`` is append-only and read-only here; we hand out a
        # frozen copy so callers cannot mutate the in-memory mirror.
        transcript = tuple(record.transcript)
        return SubagentSnapshot(
            subagent_id=record.subagent_id,
            status=record.status,
            task=record.spec.task,
            write_path=str(record.write_path),
            transcript=transcript,
            pending_guidance=tuple(guidance),
            error=record.error,
            started_at=record.started_at,
            finished_at=record.finished_at,
        )

    def append_entry(
        self,
        subagent_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Append a transcript entry from outside the supervisor thread."""
        with self.__records_lock:
            record = self.__records.get(subagent_id)
        if record is None:
            return
        self._append_entry(record, payload)

    def _append_entry(
        self,
        record: _SubagentRecord,
        payload: dict[str, Any],
    ) -> None:
        entry = TranscriptEntry.from_json(payload)
        record.transcript.append(entry)
        if len(record.transcript) > TRANSCRIPT_ENTRY_BUDGET:
            del record.transcript[
                : len(record.transcript) - TRANSCRIPT_ENTRY_BUDGET
            ]
        if payload.get("kind") == "guidance-request":
            with record.pending_lock:
                record.pending_guidance.append(entry)

        try:
            with record.transcript_path.open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(entry.to_jsonl_line())
                handle.write("\n")
        except Exception:
            logger.exception(
                "Failed to persist subagent %s transcript",
                record.subagent_id,
            )


def _entry_matches_question(
    entry: TranscriptEntry,
    question: str,
) -> bool:
    """Return whether a guidance-request transcript entry corresponds
    to ``question``."""
    needle = f"asks:\n\n{question}"
    return entry.content.strip().endswith(needle.strip())
