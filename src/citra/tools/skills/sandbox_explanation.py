"""Model-facing explanation of the finalized process sandbox."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

from .skill import Skill

if TYPE_CHECKING:
    from citra.context import ExecutionContext


class SandboxEnvironment(Skill):
    """Explain the finalized filesystem and isolated runtime to the agent."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(
            "sandbox-environment",
            "Explains the active filesystem, process, and network boundaries.",
            Path(),
        )

    @override
    def get_md(self, context: ExecutionContext) -> str:
        """Render the active sandbox policy without leaking host sources."""
        policy = context.sandbox.policy
        readonly = (
            "\n".join(
                f"- `{target}`"
                for _source, target in policy.readonly_mounts
            )
            or "- None configured."
        )
        runtime_description = (
            "Host runtime assets were copied into the immutable `/runtime` "
            "layer before this session started."
            if policy.mode.name == "FULL_SANDBOX"
            else "Host runtime assets are exposed through immutable read-only mounts."
        )
        return f"""
# Sandbox

The current project root is the writable directory `.`.

Use relative project paths. Use `@tmp`, `@home`, `@cache`, and `@env` only for
disposable runtime state. Use the `workspace` tool to roll back exact tracked
files when needed. Repository commits remain the user's responsibility.

The active policy level is `{policy.mode.name}`. {runtime_description}
Discovered commands are available through `/runtime/bin`; commands absent
from that runtime fail normally. Writable dependency commands installed into
`@env` are prepended to the same isolated `PATH`.

Network access is disabled by default and is globally denied when
`global_disallow_network` is true.

Explicit read-only host dependencies:

{readonly}
""".strip()
