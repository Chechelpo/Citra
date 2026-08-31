from citra.sandbox.runtime_discovery.base import StandardDiscovery
from citra.sandbox.runtime_discovery.prettier_runtime_discovery_v1 import PrettierRuntimeDiscovery
from citra.sandbox.runtime_discovery._eslint_runtime_discovery_v1 import EslintRuntimeDiscovery
from citra.sandbox.runtime_discovery.base import RuntimeDiscovery,RuntimeDiscoveryResult

from citra.sandbox.runtime_discovery.pyrefly_runtime_discovery import PyreflyRuntimeDiscovery
from citra.sandbox.runtime_discovery.ruff_runtime_discovery import RuffRuntimeDiscovery

def get_ro_binds() -> tuple[RuntimeDiscoveryResult, ...]:
    return (
        PyreflyRuntimeDiscovery().discover(),
        PrettierRuntimeDiscovery().discover(),
        PyreflyRuntimeDiscovery().discover(),
        RuffRuntimeDiscovery().discover(),
        StandardDiscovery().discover()
    )

__all__ = [
    "RuntimeDiscovery",
    "RuntimeDiscoveryResult"
]

