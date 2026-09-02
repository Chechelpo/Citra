from __future__ import annotations

from pathlib import Path

from citra.config import SandboxPolicy
from citra.modes import SandboxConfig
from citra.sandbox import SandboxMode, WorkspaceSandbox


def test_mode_policy_adds_to_operator_policy(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    operator_ro = tmp_path / "operator-ro"
    mode_ro = tmp_path / "mode-ro"
    mode_rw = tmp_path / "mode-rw"
    for path in (operator_ro, mode_ro, mode_rw):
        path.mkdir()

    policy = SandboxPolicy(extra_ro_binds=[operator_ro])
    policy.apply_mode_config(
        SandboxConfig(
            mode=SandboxMode.PARTIAL_SANDBOX,
            additional_ro_binds=(mode_ro,),
            additional_w_binds=(mode_rw,),
            global_network_disallow=True,
        )
    )
    sandbox = WorkspaceSandbox(project, policy, base_environment={})

    assert sandbox.mode is SandboxMode.PARTIAL_SANDBOX
    assert sandbox.readonly_binds() == (mode_ro, operator_ro)
    assert sandbox.writable_binds() == (project, mode_rw)
    assert sandbox.allows_network(True) is False
