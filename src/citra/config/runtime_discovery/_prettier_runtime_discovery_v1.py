"""Host-runtime discovery for Prettier sandbox compatibility."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shlex
import shutil

from ._base import (
    RuntimeDiscovery,
    RuntimeDiscoveryResult,
)


logger = logging.getLogger(__name__)

_COMMAND = "prettier"


class PrettierRuntimeDiscovery(RuntimeDiscovery):
    """Expose the host Prettier launcher and supporting runtime read-only.

    Prettier is resolved before Bubblewrap is created. If the host executable
    is unavailable, discovery is advisory: a warning is logged and no bind is
    contributed.
    """

    @classmethod
    def discover(cls) -> RuntimeDiscoveryResult:
        """Discover Prettier and return its required read-only host paths."""

        executable_raw = shutil.which(_COMMAND)
        if executable_raw is None:
            logger.warning(
                "Runtime discovery could not find %r on the host PATH; "
                "Prettier will not receive automatic read-only sandbox binds.",
                _COMMAND,
                extra={"origin": __name__},
            )
            return RuntimeDiscoveryResult()

        result = RuntimeDiscoveryResult(
            readonly_binds=_runtime_binds(Path(executable_raw)),
        )
        logger.debug(
            "Discovered Prettier runtime",
            extra={"origin": __name__, "executable": executable_raw},
        )
        return result


def _runtime_binds(executable: Path) -> tuple[Path, ...]:
    """Return minimal host paths needed to launch one discovered tool."""
    executable = executable.expanduser().absolute()
    resolved = executable.resolve()

    candidates: list[Path] = [
        executable.parent,
        resolved.parent,
    ]

    _append_python_environment_root(candidates, executable)
    _append_python_environment_root(candidates, resolved)
    _append_node_runtime_root(candidates, executable)
    _append_node_runtime_root(candidates, resolved)
    _append_node_modules_prefix(candidates, executable)
    _append_node_modules_prefix(candidates, resolved)

    interpreter = _shebang_interpreter(executable)
    if interpreter is not None:
        interpreter = interpreter.expanduser().absolute()
        resolved_interpreter = interpreter.resolve()

        candidates.extend(
            (
                interpreter.parent,
                resolved_interpreter.parent,
            )
        )

        _append_python_environment_root(candidates, interpreter)
        _append_python_environment_root(candidates, resolved_interpreter)
        _append_node_runtime_root(candidates, interpreter)
        _append_node_runtime_root(candidates, resolved_interpreter)
        _append_node_modules_prefix(candidates, interpreter)
        _append_node_modules_prefix(candidates, resolved_interpreter)

    return _minimal_existing_paths(candidates)


def _append_python_environment_root(
    candidates: list[Path],
    executable: Path,
) -> None:
    """Include an owning Python virtual environment when detectable."""
    parent = executable.parent
    if parent.name not in {"bin", "Scripts"}:
        return

    environment_root = parent.parent
    if (environment_root / "pyvenv.cfg").is_file():
        candidates.append(environment_root)


def _append_node_runtime_root(
    candidates: list[Path],
    executable: Path,
) -> None:
    """Include a conventional Node installation root when detectable.

    This primarily covers layouts such as NVM installations where ``node``
    and globally installed package launchers live beneath one version root.
    """
    parent = executable.parent
    if parent.name != "bin":
        return

    root = parent.parent
    if (
        (root / "bin" / "node").exists()
        or (root / "lib" / "node_modules").is_dir()
    ):
        candidates.append(root)


def _append_node_modules_prefix(
    candidates: list[Path],
    executable: Path,
) -> None:
    """Include the prefix owning a ``node_modules`` tree when present."""
    parts = executable.parts
    try:
        index = parts.index("node_modules")
    except ValueError:
        return

    if index <= 0:
        return

    if executable.is_absolute():
        prefix = Path(executable.anchor, *parts[1:index])
    else:
        prefix = Path(*parts[:index])

    if prefix.exists():
        candidates.append(prefix)


def _shebang_interpreter(executable: Path) -> Path | None:
    """Resolve the interpreter referenced by a script shebang.

    Absolute interpreters are returned directly. ``/usr/bin/env`` shebangs
    are resolved through the host PATH when they name an interpreter
    explicitly, including ``env -S`` forms.
    """
    try:
        with executable.open("rb") as stream:
            first_line = stream.readline(4096)
    except (OSError, ValueError):
        return None

    if not first_line.startswith(b"#!"):
        return None

    try:
        shebang = first_line[2:].decode("utf-8").strip()
        parts = shlex.split(shebang)
    except (UnicodeDecodeError, ValueError):
        return None

    if not parts:
        return None

    interpreter = Path(parts[0])
    if interpreter.name != "env":
        return interpreter if interpreter.is_absolute() else None

    command_parts = parts[1:]
    if command_parts[:1] == ["-S"]:
        command_parts = command_parts[1:]

    while command_parts and command_parts[0].startswith("-"):
        command_parts = command_parts[1:]

    if not command_parts:
        return None

    resolved = shutil.which(command_parts[0])
    return Path(resolved) if resolved is not None else None


def _minimal_existing_paths(
    candidates: list[Path],
) -> tuple[Path, ...]:
    """Deduplicate existing paths and remove children covered by a parent."""
    unique: list[Path] = []
    seen: set[str] = set()

    for candidate in candidates:
        path = candidate.absolute()
        if not path.exists():
            continue

        key = os.path.normpath(str(path))
        if key in seen:
            continue

        seen.add(key)
        unique.append(path)

    return tuple(
        path
        for path in unique
        if not any(
            other != path
            and other.is_dir()
            and _is_inside(path, other)
            for other in unique
        )
    )


def _is_inside(path: Path, parent: Path) -> bool:
    """Return whether a path is contained by a candidate parent path."""
    try:
        path.absolute().relative_to(parent.absolute())
        return True
    except ValueError:
        return False
