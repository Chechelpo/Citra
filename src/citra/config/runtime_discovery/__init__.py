"""Extensible host-runtime discovery registry."""

from __future__ import annotations

import logging
from pathlib import Path

from ._base import RuntimeDiscovery, RuntimeDiscoveryResult, StandardDiscovery
from ._language import (
    CRuntimeDiscovery,
    DotNetRuntimeDiscovery,
    GitRuntimeDiscovery,
    GoRuntimeDiscovery,
    JvmRuntimeDiscovery,
    NodeRuntimeDiscovery,
    PythonRuntimeDiscovery,
    RubyRuntimeDiscovery,
    RustRuntimeDiscovery,
)


logger = logging.getLogger(__name__)

# Runtime discovery is modular by construction: add another base-class object
# to this tuple without changing the aggregation or provisioning code.
DISCOVERIES: tuple[RuntimeDiscovery, ...] = (
    PythonRuntimeDiscovery(),
    NodeRuntimeDiscovery(),
    RustRuntimeDiscovery(),
    CRuntimeDiscovery(),
    GoRuntimeDiscovery(),
    JvmRuntimeDiscovery(),
    DotNetRuntimeDiscovery(),
    RubyRuntimeDiscovery(),
    GitRuntimeDiscovery(),
    StandardDiscovery(),
)


def get_ro_binds(
    discoveries: tuple[RuntimeDiscovery, ...] | None = None,
) -> tuple[RuntimeDiscoveryResult, ...]:
    """Run every registered discovery object in deterministic order."""
    registered = DISCOVERIES if discoveries is None else discoveries
    results: list[RuntimeDiscoveryResult] = []
    for discovery in registered:
        discovery_name = type(discovery).__name__
        logger.debug(
            "Starting registered runtime discovery",
            extra={"origin": __name__, "discovery": discovery_name},
        )
        try:
            results.append(discovery.discover())
        except Exception:
            logger.exception(
                "Registered runtime discovery failed",
                extra={"origin": __name__, "discovery": discovery_name},
            )
            raise
    completed = tuple(results)
    logger.info(
        "Runtime discovery registry completed",
        extra={"origin": __name__, "discoveries": len(completed)},
    )
    return completed


def aggregate_results(
    results: tuple[RuntimeDiscoveryResult, ...],
) -> RuntimeDiscoveryResult:
    """Merge discovery results while preserving first-resolved commands."""
    binds: dict[Path, None] = {}
    commands: dict[str, Path] = {}
    for result in results:
        for path in result.readonly_binds:
            binds.setdefault(path.expanduser().absolute(), None)
        for command, path in result.command_paths:
            commands.setdefault(command, path.expanduser().absolute())
    aggregate = RuntimeDiscoveryResult(
        readonly_binds=tuple(sorted(binds, key=str)),
        available_commands=tuple(commands),
        command_paths=tuple(commands.items()),
    )
    logger.debug(
        "Aggregated runtime discovery results",
        extra={
            "origin": __name__,
            "discoveries": len(results),
            "commands": len(aggregate.command_paths),
            "binds": len(aggregate.readonly_binds),
        },
    )
    return aggregate


# Kept as a lazy compatibility value. SandboxPolicy no longer imports or
# materializes it at module import time.
RO_BINDS: tuple[RuntimeDiscoveryResult, ...] = ()

__all__ = [
    "RO_BINDS",
    "DISCOVERIES",
    "RuntimeDiscovery",
    "RuntimeDiscoveryResult",
    "aggregate_results",
    "get_ro_binds",
]
