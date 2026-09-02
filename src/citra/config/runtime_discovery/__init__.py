from ._prettier_runtime_discovery_v1 import PrettierRuntimeDiscovery
from ._eslint_runtime_discovery_v1 import EslintRuntimeDiscovery
from ._base import RuntimeDiscoveryResult, StandardDiscovery

from ._pyrefly_runtime_discovery import PyreflyRuntimeDiscovery
from ._ruff_runtime_discovery import RuffRuntimeDiscovery

def get_ro_binds() -> tuple[RuntimeDiscoveryResult, ...]:
    """
    In charge of discovering ro-binds for utilities that the sandbox will use.
    """
    return (
        PyreflyRuntimeDiscovery().discover(),
        PrettierRuntimeDiscovery().discover(),
        RuffRuntimeDiscovery().discover(),
        EslintRuntimeDiscovery().discover(),
        StandardDiscovery().discover()
    )

RO_BINDS = get_ro_binds()

__all__ = [
    "RO_BINDS",
    "RuntimeDiscoveryResult",
    "get_ro_binds",
]
