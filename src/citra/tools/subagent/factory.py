"""
Factory that builds a subagent's ``ExecutionContext``.

A subagent owns separate mutable runtime state and process supervision while
reusing the orchestrator's immutable provisioned command layer. The factory is
kept separate from the supervisor to avoid an import cycle between
``supervisor`` and ``context``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from citra.context import (
    CitraConfig,
    ExecutionContext,
    WorkspaceContext,
)
from citra.context.runtime import RuntimeProvisioning
from citra.tools.skills.skill_registry import SkillRegistry
from citra.workflows import SingleModeWorkflow, WorkflowRuntime

from .guidance import SubagentGuidanceBridge
from .spec import SubagentSpec
from .mode import build_subagent_mode, default_subagent_toolset


def build_subagent_context(
    *,
    parent_workspace: WorkspaceContext,
    parent_config: CitraConfig,
    parent_skills: SkillRegistry,
    supervisor: Any,
    spec: SubagentSpec,
    write_path: Path,
    readonly_binds: tuple[Path, ...],
) -> ExecutionContext:
    """
    Construct a fresh ``ExecutionContext`` for one subagent.

    The subagent's mutable runtime is rooted under the orchestrator's runtime
    root (``<parent.root>/subagents/<id>``). The only project write mount is
    ``write_path`` and the only additional read-only binds are the validated
    model-facing paths the orchestrator explicitly passed in.
    """
    runtime_root = parent_workspace.root / "subagents" / spec.subagent_id
    runtime_root.mkdir(parents=True, exist_ok=True)

    nested_workspace = WorkspaceContext.create(
        workspace=write_path,
        temporary_workspace=runtime_root,
        library=parent_workspace.library,
        tool_definitions=(),
        runtime_assets=(),
    )

    # Worker runtimes own separate mutable home/cache/tmp/process state, but
    # share the parent's already-provisioned immutable command layer.  This
    # avoids rebuilding the same toolchain for every delegated task while
    # retaining independent process supervision and cleanup.
    workspace = replace(
        nested_workspace,
        # The parent already materialized the project. Subagents must edit the
        # selected directory in that same checkout, not a second copy whose
        # changes would disappear when the worker runtime is cleaned up.
        source_workspace=parent_workspace.source_workspace,
        workspace=write_path.resolve(),
        runtime=parent_workspace.runtime,
        provisioning=_fork_runtime_provisioning(parent_workspace),
        private_source_paths=parent_workspace.private_source_paths,
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
    # Subagents follow the same ownership rule as foreground agents: a
    # one-mode workflow owns their concrete sandbox.
    workflow = SingleModeWorkflow(
        name=f"subagent:{spec.subagent_id}",
        description="One constrained subagent turn.",
        mode=mode,
        sandbox_config=mode.sandbox_config,
    )
    policy = parent_config.sandbox_policy.clone()
    policy.apply_mode_config(mode.sandbox_config)
    policy.add_readonly_bind(workspace.runtime)
    for path in readonly_binds:
        policy.add_readonly_bind(path)
    for root in workspace.writable_roots:
        if root != workspace.workspace:
            policy.add_writable_bind(root)
    workflow_runtime = WorkflowRuntime(
        workflow=workflow,
        workspace=workspace,
        policy=policy,
    )
    sandbox = workflow_runtime.sandbox

    return ExecutionContext(
        workspace,
        skills=parent_skills,
        subagents=SubagentGuidanceBridge(
            subagent_id=spec.subagent_id,
            supervisor=supervisor,
        ),
        workflow_runtime=workflow_runtime,
        provided_config=_subagent_config(parent_config, spec),
        provided_mode=mode,
        provided_workflow=workflow,
        provided_sandbox=sandbox,
    )


def _fork_runtime_provisioning(
    parent_workspace: WorkspaceContext,
) -> RuntimeProvisioning:
    """Create a worker-local resolver over the parent's immutable assets."""
    parent = parent_workspace.provisioning
    return RuntimeProvisioning(
        runtime_root=parent_workspace.runtime,
        budget_bytes=parent.budget_bytes,
        copied_bytes=parent.copied_bytes,
        assets=dict(parent.assets),
        tools={
            tool_id: replace(
                tool,
                commands=dict(tool.commands),
            )
            for tool_id, tool in parent.tools.items()
            if not tool_id.startswith("staged:")
        },
        definitions={
            tool_id: replace(
                definition,
                health_check=None,
            )
            for tool_id, definition in parent.definitions.items()
            if not tool_id.startswith("staged:")
        },
        warnings=list(parent.warnings),
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
    new_sandbox = parent_config.sandbox_policy.clone()
    new_sandbox.global_disallow_network = bool(
        parent_config.sandbox_policy.global_disallow_network
        or not spec.network
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
        sandbox_policy=new_sandbox,
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
