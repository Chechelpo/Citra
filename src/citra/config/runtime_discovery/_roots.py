"""Safety predicates for bounded host-runtime discovery roots."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SYSTEM_TREES = (
    Path("/usr"),
    Path("/bin"),
    Path("/lib"),
    Path("/lib64"),
)

_BROAD_INSTALL_PREFIXES = frozenset(
    path.resolve()
    for path in (
        Path("/"),
        Path("/usr"),
        Path("/usr/local"),
        Path("/opt"),
        Path("/home"),
        Path("/root"),
        Path.home(),
    )
)

_UNSAFE_RECURSIVE_RUNTIME_ROOTS = _BROAD_INSTALL_PREFIXES | frozenset(
    path.resolve()
    for path in (
        Path("/bin"),
        Path("/sbin"),
        Path("/lib"),
        Path("/lib64"),
        Path("/usr/bin"),
        Path("/usr/sbin"),
        Path("/usr/lib"),
        Path("/usr/local/bin"),
        Path("/usr/local/lib"),
        Path("/usr/share"),
        Path("/etc"),
        Path("/var"),
    )
)


def is_system_prefix(path: Path) -> bool:
    """Return whether a path belongs to a shared operating-system tree."""
    resolved = path.expanduser().resolve()
    for root in _SYSTEM_TREES:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def is_broad_install_prefix(path: Path) -> bool:
    """Return whether a language prefix represents broad host state."""
    return path.expanduser().resolve() in _BROAD_INSTALL_PREFIXES


def is_unsafe_recursive_runtime_root(path: Path) -> bool:
    """Return whether a discovered directory is too broad to provision."""
    return path.expanduser().resolve() in _UNSAFE_RECURSIVE_RUNTIME_ROOTS


def is_runtime_prefix(path: Path) -> bool:
    """Return whether a script prefix is a bounded runtime installation."""
    if is_system_prefix(path):
        logger.debug(
            "Skipped shared system prefix for script runtime",
            extra={"origin": __name__, "prefix": str(path)},
        )
        return False

    markers = (
        path / "pyvenv.cfg",
        path / "bin" / "python",
        path / "bin" / "python3",
        path / "bin" / "node",
        path / "lib" / "node_modules",
    )
    bounded = any(marker.exists() for marker in markers)
    if not bounded:
        logger.debug(
            "Skipped unrecognized script installation prefix",
            extra={"origin": __name__, "prefix": str(path)},
        )
    return bounded
