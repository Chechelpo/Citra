"""
Factory that builds a subagent's ``ExecutionContext``.

A subagent runs inside its own recursive ``WorkspaceContext`` so the
Bubblewrap sandbox and provisioning logic can be reused unchanged. The
factory is kept separate from the supervisor to avoid an import cycle
between ``supervisor`` and ``context``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from citra.context import (
    CitraConfig,
    ExecutionContext,
    WorkspaceContext,
    WorkspaceContextConfig,
)
from citra.sandbox import WorkspaceSandbox
from citra.tools.skills.skill_registry import SkillRegistry

from .spec import SubagentSpec
from .mode import build_subagent_mode, default_subagent_toolset


def build_subagent_context(
    *,
    parent_workspace: WorkspaceContext,
    parent_config: CitraConfig,
    parent_skills: SkillRegistry,
    api_call: Any,
    spec: SubagentSpec,
    write_path: Path,
    readonly_binds: tuple[Path, ...],
) -> ExecutionContext:
    """
    Construct a fresh ``ExecutionContext`` for one subagent.

    The subagent's workspace is rooted under the orchestrator's runtime
    root (``<parent.runtime>/subagents/<id>``). The subagent has no
    access to ``@source``; the only writable mount is ``write_path`` and
    the only declared read-only binds are those the orchestrator
    explicitly passed in.
    """
    runtime_root = parent_workspace.root / "subagents" / spec.subagent_id
    runtime_root.mkdir(parents=True, exist_ok=True)

    workspace_config = WorkspaceContextConfig(
        temporary_workspace=str(runtime_root),
        permanent_workspace=str(write_path),
        library=str(parent_workspace.library),
        direct_source=True,
    )

    workspace = WorkspaceContext.create(
        config=workspace_config,
        workspace=write_path,
    )

    sandbox = WorkspaceSandbox(
        workspace,
        config=parent_config.sandbox,
        mode_config=_subagent_sandbox_mode(
            readonly_binds=readonly_binds,
            write_path=write_path,
            network=spec.network,
        ),
    )

    mode = build_subagent_mode(
        subagent_id=spec.subagent_id,
        task=spec.task,
        write_path=write_path,
        readonly_binds=readonly_binds,
        network=spec.network,
        system_prompt_addendum=spec.system_prompt_addendum,
        toolset=default_subagent_toolset(),
    )

    return ExecutionContext(
        workspace,
        skills=parent_skills,
        provided_config=_subagent_config(parent_config, spec),
        provided_mode=mode,
        provided_sandbox=sandbox,
    )


def _subagent_sandbox_mode(
    *,
    readonly_binds: tuple[Path, ...],
    write_path: Path,
    network: bool,
):
    """Build the ``SandboxConfig`` the subagent's mode will install."""
    from citra.modes import SandboxConfig
    from citra.sandbox.sandbox import SandboxMode

    return SandboxConfig(
        mode=SandboxMode.PARTIAL_SANDBOX,
        additional_ro_binds=tuple(
            dict.fromkeys(
                (
                    *readonly_binds,
                    *(
                        bind
                        for bind in (
                            Path("/usr"),
                            Path("/etc"),
                            Path("/var"),
                        )
                        if bind.exists()
                    ),
                )
            )
        ),
        additional_w_binds=(write_path,),
        global_network_disallow=not network,
    )


def _subagent_config(
    parent_config: CitraConfig,
    spec: SubagentSpec,
) -> CitraConfig:
    """
    Derive a subagent-specific config from the orchestrator's config.

    The subagent's mode already restricts which tools are exposed; the
    config tweaks here are belt-and-suspenders so even if a future change
    re-introduces a tool, the subagent cannot reach outside its sandbox.
    The subagent's model is pinned to ``models.subagent`` (falling back
    to the orchestrator profile) so the orchestrator's model choice
    cannot silently leak into nested agent calls.
    """
    new_memory = replace(
        parent_config.memory,
        enabled=False,
    )
    new_sandbox = replace(
        parent_config.sandbox,
        global_network_disallow=not spec.network,
    )
    new_lsp = replace(
        parent_config.lsp,
        enabled=False,
    )
    new_lint = replace(
        parent_config.lint,
        enabled=False,
    )
    new_default_profile = _resolve_subagent_profile(parent_config)

    return replace(
        parent_config,
        memory=new_memory,
        sandbox=new_sandbox,
        lsp=new_lsp,
        lint=new_lint,
        default_model_profile=new_default_profile,
    )


def _resolve_subagent_profile(parent_config: CitraConfig) -> str | None:
    """Pick the profile a subagent should use.

    When ``models.subagent`` is omitted the subagent shares the
    orchestrator's profile; otherwise the dedicated profile is
    returned. The orchestrator's ``default_model_profile`` is honored
    so explicit test overrides continue to flow through.
    """
    store = parent_config.model_config_store
    try:
        subagent = store.subagent_name()
    except (KeyError, ValueError, RuntimeError):
        return parent_config.default_model_profile
    if subagent == store.orchestrator_name():
        return None
    return subagent
