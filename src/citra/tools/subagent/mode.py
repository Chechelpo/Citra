"""
Constrained mode for subagents.

A subagent only sees a small, sandboxed world: a subset of the filesystem
tools, ``bash`` (already sandboxed by the runtime), the dedicated
``request_guidance`` tool, and no deferred tools, no document/diagram
tools, no git, no LSP, no other agents.

The mode is built from a dataclass instead of a static ``UserMode`` file
because its tool set, sandbox bindings, and system prompt are derived
per-subagent from the orchestrator's spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from citra.modes import Mode, SandboxConfig, TaskSteeringConfig
from citra.tools.default_registry import ToolSet
from citra.tools.skills.skill import Skill
from citra.tools.transient import Bash, Edit, Read, Write
from citra.sandbox.sandbox import SandboxMode

from .guidance import RequestGuidanceTool


# ---------------------------------------------------------------------------
# Subagent tool set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubagentToolset:
    """The narrow tool set exposed to a subagent."""

    read: type
    write: type
    edit: type
    bash: type
    request_guidance: type

    def to_tool_set(self) -> ToolSet:
        return ToolSet(
            core_tools=(
                self.read,
                self.write,
                self.edit,
                self.bash,
                self.request_guidance,
            ),
            deferred_tools=(),
        )


def default_subagent_toolset() -> SubagentToolset:
    return SubagentToolset(
        read=Read,
        write=Write,
        edit=Edit,
        bash=Bash,
        request_guidance=RequestGuidanceTool,
    )


# ---------------------------------------------------------------------------
# Subagent mode
# ---------------------------------------------------------------------------

class SubagentMode(Mode):
    """
    A mode that exposes only the subagent's narrow tool set.

    ``skill_registry`` is intentionally empty: skills are an orchestrator
    affordance, and a subagent is meant to receive its instructions from
    the orchestrator through its system prompt instead.
    """

    def __init__(
        self,
        *,
        name: str,
        system_prompt: str,
        toolset: SubagentToolset,
        sandbox_config: SandboxConfig,
    ) -> None:
        self._name = name
        self._description = (
            "Constrained mode for a subagent. Only filesystem read/write/"
            "edit, bash, and request_guidance are available."
        )
        self._system_prompt = system_prompt
        self._toolset = toolset
        self._sandbox_config = sandbox_config
        self._task_steering: TaskSteeringConfig | None = None
        self._initial_working_states: tuple[str, ...] = ()
        self._skills: tuple[Skill, ...] = ()
        self._tool_set = toolset.to_tool_set()
        self.validate()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def tool_set(self) -> ToolSet:
        return self._tool_set

    @property
    def skills(self) -> tuple[Skill, ...]:
        return self._skills

    @property
    def sandbox_config(self) -> SandboxConfig:
        return self._sandbox_config

    @property
    def task_steering(self) -> TaskSteeringConfig | None:
        return self._task_steering

    @property
    def initial_working_states(self) -> tuple[str, ...]:
        return self._initial_working_states

    def get_system_prompt(
        self,
        context: Any,
    ) -> str:
        del context
        return self._system_prompt


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_subagent_mode(
    *,
    subagent_id: str,
    task: str,
    write_path: Path,
    readonly_binds: tuple[Path, ...],
    network: bool,
    system_prompt_addendum: str,
    toolset: SubagentToolset | None = None,
) -> SubagentMode:
    """
    Construct the constrained mode for one subagent.

    The returned mode:

      * exposes only ``read``, ``write``, ``edit``, ``bash`` and the
        subagent's ``request_guidance`` tool;
      * has its sandbox bounded to ``write_path`` (writable) plus the
        declared ``readonly_binds`` (read-only) and the Citra runtime
        mounts that ``bwrap`` always re-exposes;
      * carries a system prompt that describes the subagent's task, its
        allowed environment, and its boundary contract with the
        orchestrator.
    """
    toolset = toolset or default_subagent_toolset()

    sandbox_config = _subagent_sandbox_config(
        write_path=write_path,
        readonly_binds=readonly_binds,
        network=network,
    )

    system_prompt = _build_system_prompt(
        subagent_id=subagent_id,
        task=task,
        write_path=write_path,
        readonly_binds=readonly_binds,
        network=network,
        addendum=system_prompt_addendum,
    )

    return SubagentMode(
        name=f"subagent:{subagent_id}",
        system_prompt=system_prompt,
        toolset=toolset,
        sandbox_config=sandbox_config,
    )


def _subagent_sandbox_config(
    *,
    write_path: Path,
    readonly_binds: tuple[Path, ...],
    network: bool,
) -> SandboxConfig:
    """
    Build the SandboxConfig used by a subagent.

    ``bwrap`` will still apply the orchestrator's baseline
    (``SANDBOX_WRITABLE_DIRS`` and the runtime's read-only binds), so we
    only need to declare the *additional* writable/ro binds.
    """
    additional_w_binds: list[Path] = [write_path]
    additional_ro_binds: list[Path] = []
    for raw in readonly_binds:
        resolved = Path(raw).expanduser().resolve()
        if not resolved.exists():
            # We still record it; the sandbox will silently ignore a
            # bind of a non-existent path, but the model at least sees
            # the declaration in its prompt.
            continue
        if resolved == write_path.resolve():
            continue
        additional_ro_binds.append(resolved)

    return SandboxConfig(
        mode=SandboxMode.PARTIAL_SANDBOX,
        additional_ro_binds=tuple(additional_ro_binds),
        additional_w_binds=tuple(additional_w_binds),
        global_network_disallow=not network,
    )


def _build_system_prompt(
    *,
    subagent_id: str,
    task: str,
    write_path: Path,
    readonly_binds: tuple[Path, ...],
    network: bool,
    addendum: str,
) -> str:
    lines: list[str] = [
        "# Subagent",
        "",
        f"You are subagent `{subagent_id}`, spawned by the orchestrator "
        "to complete a well-defined component of a larger task. You are "
        "not the orchestrator. Do not reason about the project as a whole; "
        "complete the task below using the tools available to you.",
        "",
        "## Task",
        "",
        task.strip(),
        "",
        "## Environment",
        "",
        f"- Write directory: `{write_path}` (the only place you may create or modify files).",
        "- Read-only binds:",
    ]

    if readonly_binds:
        for bind in readonly_binds:
            lines.append(f"  - `{bind}`")
    else:
        lines.append("  - (none)")

    lines.extend(
        [
            "- Sandbox: Bubblewrap with the orchestrator's runtime binds and "
            "the writable/read-only paths above. Other host locations are not "
            "reachable.",
            f"- Network: {'allowed' if network else 'denied'}.",
            "",
            "## Available tools",
            "",
            "- `read`, `write`, `edit`: operate relative to the write directory.",
            "- `bash`: runs inside the same sandbox; the Bubblewrap policy "
            "above determines what it can reach.",
            "- `request_guidance(question)`: ask the orchestrator a single, "
            "self-contained question and block until the orchestrator responds. "
            "Use this when an ambiguity in the task or environment prevents you "
            "from making progress.",
            "",
            "## Boundaries",
            "",
            "- Stay strictly inside the write directory. Do not attempt to "
            "read or write paths the sandbox or your tools do not expose.",
            "- You cannot open other tools. If you think you need a tool you "
            "do not have, ask the orchestrator through `request_guidance`.",
            "- Prefer small, well-scoped changes. Do not introduce unrelated "
            "refactors.",
            "- When you are done, end your turn with a short summary of what "
            "you changed. The orchestrator will collect your transcript.",
        ]
    )

    if addendum.strip():
        lines.extend(
            [
                "",
                "## Additional instructions from the orchestrator",
                "",
                addendum.strip(),
            ]
        )

    return "\n".join(lines) + "\n"
