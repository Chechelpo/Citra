"""
Constrained single-mode workflow for subagents.

A subagent only sees a small, sandboxed world: a subset of the filesystem
tools, safe file rollback, ``bash`` (already sandboxed by the runtime), the dedicated
``request_guidance`` tool, and no deferred tools, no document/diagram
tools, no git, no LSP, no other agents.

The workflow is built from a dataclass instead of static configuration
because its tool set, sandbox bindings, and system prompt are derived
per-subagent from the orchestrator's spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from citra.workflows import (
    SandboxConfig,
    SingleModeWorkflow,
    TaskSteeringConfig,
)
from citra.tools.default_registry import ToolSet
from citra.tools.skills.skill import Skill
from citra.tools.tool import Tool
from citra.tools.transient import Bash, Edit, Read, Workspace, Write
from citra.sandbox.sandbox import SandboxMode

from .guidance import RequestGuidanceTool

if TYPE_CHECKING:
    from citra.context import ExecutionContext


# ---------------------------------------------------------------------------
# Subagent tool set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubagentToolset:
    """The narrow tool set exposed to a subagent."""

    read: type[Tool]
    write: type[Tool]
    edit: type[Tool]
    bash: type[Tool]
    workspace: type[Tool]
    request_guidance: type[Tool]

    def __post_init__(self) -> None:
        for name in (
            "read",
            "write",
            "edit",
            "bash",
            "workspace",
            "request_guidance",
        ):
            tool = vars(self)[name]
            if not isinstance(tool, type) or not issubclass(tool, Tool):
                raise TypeError(f"{name} must be a Tool subclass")

    def to_tool_set(self) -> ToolSet:
        return ToolSet(
            core_tools=(
                self.read,
                self.write,
                self.edit,
                self.bash,
                self.workspace,
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
        workspace=Workspace,
        request_guidance=RequestGuidanceTool,
    )


# ---------------------------------------------------------------------------
# Subagent workflow
# ---------------------------------------------------------------------------

class SubagentWorkflow(SingleModeWorkflow):
    """
    A workflow that exposes only the subagent's narrow tool set.

    ``skill_registry`` is intentionally empty: skills are an orchestrator
    affordance, and a subagent is meant to receive its instructions from
    the orchestrator through its system prompt instead.
    """

    def __init__(
        self,
        *,
        name: str,
        subagent_id: str,
        task: str,
        write_path: Path,
        readonly_binds: tuple[Path, ...],
        network: bool,
        system_prompt_addendum: str,
        toolset: SubagentToolset,
        sandbox_config: SandboxConfig,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name cannot be empty")
        if not isinstance(subagent_id, str) or not subagent_id.strip():
            raise ValueError("subagent_id cannot be empty")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("subagent task cannot be empty")
        if not isinstance(write_path, Path):
            raise TypeError("write_path must be a Path")
        if not isinstance(readonly_binds, tuple):
            raise TypeError("readonly_binds must be a tuple")
        if any(not isinstance(path, Path) for path in readonly_binds):
            raise TypeError("readonly_binds must contain Path instances")
        if not isinstance(network, bool):
            raise TypeError("network must be boolean")
        if not isinstance(system_prompt_addendum, str):
            raise TypeError("system_prompt_addendum must be a string")
        if not isinstance(toolset, SubagentToolset):
            raise TypeError("toolset must be a SubagentToolset")
        if not isinstance(sandbox_config, SandboxConfig):
            raise TypeError("sandbox_config must be a SandboxConfig")
        self._name = name
        self._description = (
            "Constrained workflow for a subagent. Only filesystem read/write/"
            "edit, bash, and request_guidance are available."
        )
        self._subagent_id = subagent_id
        self._task = task
        self._write_path = write_path
        self._readonly_binds = readonly_binds
        self._network = network
        self._system_prompt_addendum = system_prompt_addendum
        self._toolset = toolset
        self._sandbox_config = sandbox_config
        self._task_steering = TaskSteeringConfig()
        self._initial_working_states: tuple[str, ...] = ()
        self._skills: tuple[Skill, ...] = ()
        self._tool_set = toolset.to_tool_set()
        self.validate()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
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
    def task_steering(self) -> TaskSteeringConfig:
        return self._task_steering

    @property
    def initial_working_states(self) -> tuple[str, ...]:
        return self._initial_working_states

    def get_system_prompt(
        self,
        context: ExecutionContext,
    ) -> str:
        return _build_system_prompt(
            subagent_id=self._subagent_id,
            task=self._task,
            write_path=self._write_path,
            readonly_binds=self._readonly_binds,
            network=self._network,
            addendum=self._system_prompt_addendum,
            toolset=self._toolset,
            context=context,
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_subagent_workflow(
    *,
    subagent_id: str,
    task: str,
    write_path: Path,
    readonly_binds: tuple[Path, ...],
    network: bool,
    system_prompt_addendum: str,
    toolset: SubagentToolset,
) -> SubagentWorkflow:
    """
    Construct the constrained workflow for one subagent.

    The returned workflow:

      * exposes only ``read``, ``write``, ``edit``, ``workspace``, ``bash`` and the
        subagent's ``request_guidance`` tool;
      * has its sandbox bounded to ``write_path`` (writable) plus the
        declared ``readonly_binds`` (read-only) and the Citra runtime
        mounts that ``bwrap`` always re-exposes;
      * carries a system prompt that describes the subagent's task, its
        allowed environment, and its boundary contract with the
        orchestrator.
    """
    sandbox_config = _subagent_sandbox_config(
        write_path=write_path,
        readonly_binds=readonly_binds,
        network=network,
    )

    return SubagentWorkflow(
        name=f"subagent:{subagent_id}",
        subagent_id=subagent_id,
        task=task,
        write_path=write_path,
        readonly_binds=readonly_binds,
        network=network,
        system_prompt_addendum=system_prompt_addendum,
        toolset=toolset,
        sandbox_config=sandbox_config,
    )


# Transitional names for callers that have not migrated to workflow wording.
SubagentMode = SubagentWorkflow


def build_subagent_mode(
    *,
    subagent_id: str,
    task: str,
    write_path: Path,
    readonly_binds: tuple[Path, ...],
    network: bool,
    system_prompt_addendum: str,
    toolset: SubagentToolset | None = None,
) -> SubagentWorkflow:
    return build_subagent_workflow(
        subagent_id=subagent_id,
        task=task,
        write_path=write_path,
        readonly_binds=readonly_binds,
        network=network,
        system_prompt_addendum=system_prompt_addendum,
        toolset=toolset or default_subagent_toolset(),
    )


def _subagent_sandbox_config(
    *,
    write_path: Path,
    readonly_binds: tuple[Path, ...],
    network: bool,
) -> SandboxConfig:
    """
    Build the SandboxConfig owned by a subagent's single-mode workflow.

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
    toolset: SubagentToolset,
    context: ExecutionContext,
) -> str:
    public_names = {
        name: tool_type.resolve_definition_for_context(context).function.name
        for name, tool_type in (
            ("read", toolset.read),
            ("write", toolset.write),
            ("edit", toolset.edit),
            ("bash", toolset.bash),
            ("workspace", toolset.workspace),
            ("request_guidance", toolset.request_guidance),
        )
    }
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
        "- Current project: `.` (the only place you may create or modify "
        "files).",
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
            f"- `{public_names['read']}`, `{public_names['write']}`, "
            f"`{public_names['edit']}`: operate relative to the current project.",
            f"- `{public_names['workspace']}`: roll back an exact tracked file "
            "when an attempted edit is wrong.",
            f"- `{public_names['bash']}`: runs inside the same sandbox; "
            "the Bubblewrap policy "
            "above determines what it can reach. Do not use it for Git mutation.",
            f"- `{public_names['request_guidance']}`: ask the orchestrator a "
            "single, "
            "self-contained question and block until the orchestrator responds. "
            "Use this when an ambiguity in the task or environment prevents you "
            "from making progress.",
            "",
            "## Boundaries",
            "",
            "- Stay strictly inside the current project. Do not attempt to "
            "read or write paths the sandbox or your tools do not expose.",
            "- Do not stage or commit changes; repository history belongs to "
            "the user.",
            "- You cannot open other tools. If you think you need a tool you "
            "do not have, ask the orchestrator through "
            f"`{public_names['request_guidance']}`.",
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
