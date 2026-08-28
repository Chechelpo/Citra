from citra.sandbox.runtime_discovery.prettier_runtime_discovery_v1 import PrettierRuntimeDiscovery
from citra.sandbox.runtime_discovery.eslint_runtime_discovery_v1 import EslintRuntimeDiscovery
from citra.sandbox.runtime_discovery.base import RuntimeDiscovery,RuntimeDiscoveryResult

from citra.sandbox.runtime_discovery.pyrefly_runtime_discovery import PyreflyRuntimeDiscovery
from citra.sandbox.runtime_discovery.ruff_runtime_discovery import RuffRuntimeDiscovery

RUNTIME_DISCOVERY: tuple[type[RuntimeDiscovery], ...] = (
    PyreflyRuntimeDiscovery,
    RuffRuntimeDiscovery,
    EslintRuntimeDiscovery,
    PrettierRuntimeDiscovery
)
__all__ = [
    "RuntimeDiscovery",
    "RUNTIME_DISCOVERY",
    "RuntimeDiscoveryResult"
]

