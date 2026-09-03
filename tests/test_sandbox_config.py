from __future__ import annotations

from pathlib import Path

from citra.config import SandboxPolicy
from citra.sandbox import SandboxMode, WorkspaceSandbox
from citra.workflows import SandboxConfig


def test_workflow_policy_adds_to_operator_policy(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    operator_ro = tmp_path / "operator-ro"
    workflow_ro = tmp_path / "workflow-ro"
    workflow_rw = tmp_path / "workflow-rw"
    for path in (operator_ro, workflow_ro, workflow_rw):
        path.mkdir()

    policy = SandboxPolicy(extra_ro_binds=[operator_ro])
    policy.apply_workflow_config(
        SandboxConfig(
            mode=SandboxMode.PARTIAL_SANDBOX,
            additional_ro_binds=(workflow_ro,),
            additional_w_binds=(workflow_rw,),
            global_network_disallow=True,
        )
    )
    sandbox = WorkspaceSandbox(project, policy, base_environment={})

    assert sandbox.mode is SandboxMode.PARTIAL_SANDBOX
    assert sandbox.readonly_binds()[:2] == (workflow_ro, operator_ro)
    assert sandbox.writable_binds()[:2] == (project, workflow_rw)
    assert sandbox.allows_network(True) is False
