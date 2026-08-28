"""Tests for the sandbox-environment skill and the SandboxMode-aware rendering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from citra.sandbox.sandbox import SandboxEnvironmentInfo, SandboxMode
from citra.tools.skills.sandbox_explanation import SandboxEnvironment


def _make_context(mode: SandboxMode) -> MagicMock:
    context = MagicMock()
    context.workspace.workspace = Path("/agent/workspace")
    context.workspace.source_workspace = Path("/user/source")
    context.workspace.home = Path("/agent/home")
    context.workspace.tmp = Path("/agent/tmp")
    context.workspace.cache = Path("/agent/cache")
    context.workspace.env = Path("/agent/env")
    context.workspace.runtime = Path("/agent/runtime")
    context.sandbox.environment_info.return_value = SandboxEnvironmentInfo(
        mode=mode,
        extra_readonly_binds=(Path("/host/extra"),),
    )
    return context


class TestSandboxSkill:
    def test_skill_is_named_and_described(self) -> None:
        skill = SandboxEnvironment()
        assert skill.name == "sandbox-environment"
        assert "SandboxMode" in skill.description

    def test_full_access_mode_mentions_no_sandbox(self) -> None:
        rendered = SandboxEnvironment().get_md(_make_context(SandboxMode.FULL_ACCESS))
        assert "FULL_ACCESS" in rendered
        assert "full-access" in rendered
        assert "full authority" in rendered
        assert "Commit" in rendered  # the operating discipline still applies
        assert "read-only `@source`" not in rendered

    def test_only_source_mode_says_source_is_writable(self) -> None:
        rendered = SandboxEnvironment().get_md(_make_context(SandboxMode.ONLY_SOURCE))
        assert "ONLY_SOURCE" in rendered
        assert "authoritative project is exposed directly" in rendered
        assert "source immediately" in rendered

    def test_partial_sandbox_mode_describes_partial_visibility(self) -> None:
        rendered = SandboxEnvironment().get_md(
            _make_context(SandboxMode.PARTIAL_SANDBOX)
        )
        assert "PARTIAL_SANDBOX" in rendered
        assert "Bubblewrap" in rendered
        assert "read-only `@source`" in rendered
        assert "Commit" in rendered
        assert "partial-sandbox" in rendered

    def test_full_sandbox_mode_describes_strict_isolation(self) -> None:
        rendered = SandboxEnvironment().get_md(_make_context(SandboxMode.FULL_SANDBOX))
        assert "FULL_SANDBOX" in rendered
        assert "Bubblewrap" in rendered
        assert "read-only `@source`" in rendered
        assert "does not normally expose the complete host root" in rendered
        assert "full-sandbox" in rendered

    def test_extra_readonly_binds_are_always_listed(self) -> None:
        for mode in SandboxMode:
            rendered = SandboxEnvironment().get_md(_make_context(mode))
            assert "/host/extra" in rendered, (
                f"Extra read-only bind not listed for mode {mode.name}"
            )

    def test_all_modes_share_process_lifetime_filesystem_block(self) -> None:
        # The runtime roots (home/tmp/cache/env/runtime) are explained in every
        # mode, not just the sandboxed ones.
        for mode in SandboxMode:
            rendered = SandboxEnvironment().get_md(_make_context(mode))
            assert "/agent/workspace" in rendered
            assert "/agent/home" in rendered
            assert "/agent/tmp" in rendered
            assert "/agent/cache" in rendered
            assert "/agent/env" in rendered
            assert "/agent/runtime" in rendered
