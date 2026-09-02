"""Tests for the model-facing sandbox explanation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from citra.config import SandboxPolicy
from citra.sandbox import SandboxMode
from citra.tools.skills.sandbox_explanation import SandboxEnvironment


def _context(mode: SandboxMode = SandboxMode.FULL_SANDBOX) -> SimpleNamespace:
    policy = SandboxPolicy(
        mode=mode,
        base_readonly_binds=[Path("/host/runtime")],
        global_disallow_network=True,
    )
    return SimpleNamespace(sandbox=SimpleNamespace(policy=policy))


def test_skill_explains_one_current_project() -> None:
    rendered = SandboxEnvironment().get_md(_context())
    assert "current project root is the writable directory `.`" in rendered
    assert "workspace` tool" in rendered
    assert "commits remain the user's responsibility" in rendered
    assert "@source" not in rendered


def test_skill_reports_finalized_policy() -> None:
    rendered = SandboxEnvironment().get_md(_context(SandboxMode.PARTIAL_SANDBOX))
    assert "PARTIAL_SANDBOX" in rendered
    assert "global_disallow_network" in rendered
    assert "/host/runtime" in rendered
