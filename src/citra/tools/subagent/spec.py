"""
Data model for the subagent runtime.

This module owns the value types that cross the orchestrator <-> subagent
boundary. They are deliberately small, frozen, and JSON-serializable so a
subagent transcript can be replayed and inspected from outside the
subagent's own process state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
import json


# Hard cap on how much of a single subagent transcript is kept in memory.
# The on-disk JSONL file is the source of truth; this is just a budget for
# the in-memory mirror used by ``SubagentTool.poll`` and the supervisor.
TRANSCRIPT_MEMORY_BUDGET: int = 256

# Hard cap on how many entries the in-memory mirror keeps for a single
# subagent before the oldest ones are dropped. Older entries are still
# preserved on disk.
TRANSCRIPT_ENTRY_BUDGET: int = 512


class SubagentStatus(str, Enum):
    """Lifecycle state of one subagent."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            SubagentStatus.COMPLETED,
            SubagentStatus.FAILED,
            SubagentStatus.CANCELLED,
        }


@dataclass(frozen=True)
class SubagentSpec:
    """
    Inputs to ``SubagentTool.create``.

    The orchestrator is responsible for choosing an unambiguous ``subagent_id``
    (defaulting to a short, content-derived id) and the ``write_path`` that
    the subagent is allowed to mutate. ``readonly_binds`` may include any
    model-facing path the orchestrator wants to expose as read-only (typically
    source material needed to implement the component).
    """

    task: str
    write_path: str
    readonly_binds: tuple[str, ...] = ()
    subagent_id: str = ""
    network: bool = False
    system_prompt_addendum: str = ""

    def __post_init__(self) -> None:
        task = self.task.strip()
        if not task:
            raise ValueError("Subagent task cannot be empty.")
        object.__setattr__(self, "task", task)

        write_path = self.write_path.strip()
        if not write_path:
            raise ValueError("Subagent write_path cannot be empty.")
        object.__setattr__(self, "write_path", write_path)

        normalized: list[str] = []
        for raw in self.readonly_binds:
            path = str(raw).strip()
            if not path:
                continue
            normalized.append(path)
        object.__setattr__(self, "readonly_binds", tuple(normalized))

        subagent_id = self.subagent_id.strip()
        if not subagent_id:
            subagent_id = _default_subagent_id(task)
        if len(subagent_id) > 48 or any(
            not (character.isalnum() or character in {"-", "_"})
            for character in subagent_id
        ):
            raise ValueError(
                "Subagent id must be at most 48 characters and contain "
                "only letters, numbers, '-' or '_'."
            )
        object.__setattr__(self, "subagent_id", subagent_id)

        if not isinstance(self.network, bool):
            raise TypeError("SubagentSpec.network must be a bool.")

    def to_json(self) -> dict[str, Any]:
        return {
            "subagent_id": self.subagent_id,
            "task": self.task,
            "write_path": self.write_path,
            "readonly_binds": list(self.readonly_binds),
            "network": self.network,
            "system_prompt_addendum": self.system_prompt_addendum,
        }


@dataclass(frozen=True)
class TranscriptEntry:
    """One immutable line in a subagent's communication log."""

    role: str
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    kind: str = "message"
    tool: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "tool": self.tool,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "TranscriptEntry":
        try:
            role = str(payload["role"])
            content = str(payload["content"])
        except KeyError as error:
            raise ValueError(
                f"Transcript entry is missing required field: {error}"
            ) from error
        return cls(
            role=role,
            content=content,
            timestamp=str(
                payload.get("timestamp")
                or datetime.now(timezone.utc).isoformat()
            ),
            kind=str(payload.get("kind") or "message"),
            tool=(
                str(payload["tool"])
                if payload.get("tool") is not None
                else None
            ),
        )

    def to_jsonl_line(self) -> str:
        return json.dumps(
            self.to_json(),
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class SubagentSnapshot:
    """Read-only view of a subagent as observed by the orchestrator."""

    subagent_id: str
    status: SubagentStatus
    task: str
    write_path: str
    transcript: tuple[TranscriptEntry, ...]
    pending_guidance: tuple[TranscriptEntry, ...]
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    def to_json(self) -> dict[str, Any]:
        return {
            "subagent_id": self.subagent_id,
            "status": self.status.value,
            "task": self.task,
            "write_path": self.write_path,
            "transcript": [entry.to_json() for entry in self.transcript],
            "pending_guidance": [
                entry.to_json() for entry in self.pending_guidance
            ],
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _default_subagent_id(task: str) -> str:
    """Pick a short, filesystem-friendly id from a task description."""
    sanitized = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "-"
        for ch in task.lower()
    )
    sanitized = sanitized.strip("-")
    if not sanitized:
        sanitized = "subagent"
    return sanitized[:48]


def resolve_write_path(
    base_workspace: Path,
    write_path: str,
) -> Path:
    """
    Resolve a model-supplied write path against an orchestrator workspace.

    The result is always a directory. The subagent's runtime then exposes
    only this directory as writable.
    """
    workspace = base_workspace.expanduser().resolve()
    raw = Path(write_path).expanduser()
    if not raw.is_absolute():
        raw = workspace / raw
    resolved = raw.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError(
            "Subagent write path must remain inside the orchestrator "
            f"workspace: {resolved}"
        ) from error
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(
            f"Subagent write path is not a directory: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
