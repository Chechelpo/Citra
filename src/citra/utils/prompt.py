"""
System-prompt construction utilities for Citra.

This module owns the dynamic environment context injected into the
model's system prompt.

The prompt should describe Citra's role and operating environment,
while individual tool definitions remain responsible for explaining
their own detailed behavior.
"""

from citra.tools.skills.skill import Skill
from collections.abc import Iterable
import platform
from dataclasses import dataclass
from datetime import datetime

from ..context import ExecutionContext


__all__ = [
    "build_system_prompt",
]

@dataclass(frozen=True)
class PromptEnvironment:
    """
    Dynamic environment information exposed to the model.
    """

    workspace: str
    source_workspace: str
    os: str
    architecture: str
    python_version: str
    datetime: str
    timezone: str
    git_repository: bool
    direct_source: bool
    memory_enabled: bool

    def as_prompt_section(self) -> str:
        return "\n".join(
            (
                "## Environment",
                "",
                f"- Workspace: {self.workspace}",
                f"- Source workspace: {self.source_workspace}",
                f"- OS: {self.os}",
                f"- Architecture: {self.architecture}",
                f"- Python: {self.python_version}",
                f"- Date/time: {self.datetime}",
                f"- Timezone: {self.timezone}",
                f"- Git repository: {'yes' if self.git_repository else 'no'}",
                f"- Direct source access: {'yes' if self.direct_source else 'no'}",
                f"- Memory enabled: {'yes' if self.memory_enabled else 'no'}",
            )
        )


def format_skills(skills: Iterable[Skill]) -> str:
    return "\n".join(
        f"- **{skill.name}**: {skill.description}"
        for skill in skills
    )
def build_system_prompt(
    context: ExecutionContext,
) -> str:
    """Build the active mode's system prompt for one model API call."""
    return context.mode.get_system_prompt(context)


def _project_description(direct_source: bool) -> str:
    if direct_source:
        return (
            "The user's project is exposed directly as the writable active "
            "project root; project edits affect the authoritative source "
            "immediately."
        )
    return (
        "The user's project is already present as a complete writable "
        "snapshot in the disposable agent workspace."
    )


def _source_environment_line(environment: PromptEnvironment) -> str:
    if environment.direct_source:
        return (
            "- **Source access:** direct and writable "
            f"(`{environment.source_workspace}`)"
        )
    return f"- **Read-only source:** `{environment.source_workspace}`"


def _operating_model(direct_source: bool) -> str:
    runtime_state = (
        "Runtime dependency/cache state persists for this Citra process and "
        "is discarded when Citra exits."
    )
    if direct_source:
        return "\n\n".join(
            (
                "`@workspace` and `@source` both resolve to the authoritative "
                "writable project. Changes are immediate; there is no Citra "
                "Commit or materialization workflow in this mode.",
                "Make changes, install staged dependencies, run diagnostics, "
                f"build, and test in the active project root. {runtime_state}",
            )
        )
    return "\n\n".join(
        (
            "`@source` is authoritative and read-only.",
            "The complete project is already present in the workspace. Make "
            "changes, install staged dependencies, run diagnostics, build, "
            f"and test there. {runtime_state}",
            "Apply verified project changes to `@source` only through "
            "Citra's Commit workflow.",
        )
    )


def _memory_guidance(enabled: bool) -> str:
    if not enabled:
        return ""
    return """\
## Session memory

Conversation history may be aggressively trimmed. Session memory is durable
task state and is expected for non-trivial work.

Memory is not a transcript.

Use:

* **Working State** for unresolved reasoning, hypotheses, investigations, or
  tentative interpretations that must survive context trimming.
* **TODO** for required work.
* **Requirement** for acceptance conditions and their verified satisfaction.
* **Fact** for verified information.
* **Decision** for choices later work should remain consistent with.
* **Constraint** for active boundaries and invariants.
* **Checkpoint** for a compact resume point when unfinished work may continue
  later.

TODOs, facts, decisions, and constraints may be created directly when they
already qualify as durable memory.

Working State is optional. Do not manufacture it merely to create another
memory entry.

When genuine Working State produces a durable consequence, promotion may be
used to preserve that provenance. One Working State may produce multiple
durable entries.

Keep memory concise, truthful, and synchronized with current reality.

Follow the **Task Recognition** skill for detailed memory semantics and
maintenance rules.
"""


def _working_method(*, direct_source: bool, memory_enabled: bool) -> str:
    steps = [
        "Inspect the relevant source. Use tree for discovery; do not rely "
        "solely on read.",
    ]
    if memory_enabled:
        steps.extend(
            (
                "Inspect existing memory and record durable TODOs, facts, "
                "decisions, or constraints when they become useful.",
                "Create Working State only when unresolved investigation "
                "itself must survive context trimming.",
            )
        )
    steps.extend(
        (
            "Investigate safely, using `@tmp` for disposable work.",
            (
                "Implement the smallest coherent solution directly in the "
                "active source."
                if direct_source
                else "Implement the smallest coherent solution in the "
                "existing workspace copy."
            ),
            "Diagnose and test the result in the shared Agent Runtime environment.",
        )
    )
    if not direct_source:
        steps.extend(
            (
                "Inspect the intended diff and stage only the required changes.",
                "Apply verified changes through Commit.",
            )
        )
    if memory_enabled:
        steps.append("Reconcile memory with the resulting state.")

    rendered = "\n".join(
        f"{index}. {step}"
        for index, step in enumerate(steps, start=1)
    )
    suffix = ""
    if memory_enabled:
        suffix = (
            "\n\nWhen an investigation represented by Working State resolves, "
            "promote useful consequences when provenance matters, then "
            "resolve or discard the Working State."
        )
    return (
        "For non-trivial work, generally:\n\n"
        f"{rendered}{suffix}\n\n"
        "Adapt this workflow when a simpler path is sufficient."
    )


def _completion_requirements(
    *,
    direct_source: bool,
    memory_enabled: bool,
) -> str:
    items = [
        (
            "ensure requested project edits are present in the authoritative "
            "source;"
            if direct_source
            else "ensure requested project changes were applied through Commit;"
        )
    ]
    if memory_enabled:
        items.extend(
            (
                "ensure every valid requirement is satisfied;",
                "ensure every valid TODO is complete;",
                "resolve or discard obsolete Working State;",
                "remove stale or incorrect memory;",
                "ensure retained facts, decisions, and constraints reflect reality;",
                "clear or refresh the checkpoint as appropriate;",
            )
        )
    return "\n".join(f"* {item}" for item in items)




def collect_environment(
    context: ExecutionContext,
) -> PromptEnvironment:
    now = datetime.now().astimezone()

    return PromptEnvironment(
        workspace=str(
            context.workspace.workspace
        ),
        source_workspace=str(
            context.workspace.source_workspace
        ),
        os=context.os,
        architecture=platform.machine() or "unknown",
        python_version=platform.python_version(),
        datetime=now.isoformat(
            timespec="seconds"
        ),
        timezone=_timezone_name(now),
        git_repository=_is_git_repository(
            context
        ),
        direct_source=bool(
            getattr(context.workspace, "direct_source", False)
        ),
        memory_enabled=bool(
            getattr(
                getattr(context.config, "memory", None),
                "enabled",
                True,
            )
        ),
    )


def _is_git_repository(
    context: ExecutionContext,
) -> bool:
    """Reuse the trusted workspace-commit service's startup detection."""
    changes = context.workspace.changes
    if changes is not None:
        return changes.source_is_git_repository
    return (context.workspace.workspace / ".git").is_dir()


def _timezone_name(
    value: datetime,
) -> str:
    name = value.tzname()

    if name:
        return name

    offset = value.strftime("%z")

    if offset:
        return offset

    return "unknown"
