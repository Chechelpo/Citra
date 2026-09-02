"""Sandbox public API.

The implementation depends on :mod:`citra.config`, while configuration only
needs :class:`SandboxMode`.  Lazy exports keep those two package initializers
from importing each other recursively.
"""

from citra.sandbox.sandbox_mode import SandboxMode

__all__ = [
    "SandboxMode",
    "SandboxResult",
    "SandboxedFilesystem",
    "WorkspaceSandbox",
]


def __getattr__(name: str):
    if name in {"SandboxResult", "WorkspaceSandbox"}:
        from citra.sandbox.sandbox import SandboxResult, WorkspaceSandbox

        return {
            "SandboxResult": SandboxResult,
            "WorkspaceSandbox": WorkspaceSandbox,
        }[name]
    if name == "SandboxedFilesystem":
        from citra.sandbox.sandboxed_filesystem import SandboxedFilesystem

        return SandboxedFilesystem
    raise AttributeError(name)
