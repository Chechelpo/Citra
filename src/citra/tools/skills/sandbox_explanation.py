"""Model-facing explanation of the finalized process sandbox."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

from .skill import Skill

if TYPE_CHECKING:
    from citra.context import ExecutionContext


class SandboxEnvironment(Skill):
    def __init__(self) -> None:
        super().__init__(
            "sandbox-environment",
            "Explains the active filesystem, process, and network boundaries.",
            Path(),
        )

    @override
    def get_md(self, context: ExecutionContext) -> str:
        policy = context.sandbox.policy
        readonly = (
            "\n".join(f"- `{path}`" for path in policy.readonly_binds)
            or "- None configured."
        )
        return f"""
# Sandbox

The current project root is the writable directory `.`.

Use relative project paths. Use `@tmp`, `@home`, `@cache`, and `@env` only for
disposable runtime state. Use the `workspace` tool to roll back exact tracked
files when needed. Repository commits remain the user's responsibility.

The active policy level is `{policy.mode.name}`. Network access is disabled by
default and is globally denied when `global_disallow_network` is true.

Explicit read-only host dependencies:

{readonly}
""".strip()
