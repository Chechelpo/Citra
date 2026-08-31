"""Host-runtime discovery for Ruff sandbox compatibility."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shlex
import shutil
from typing import TYPE_CHECKING

from .base import RuntimeDiscovery, RuntimeDiscoveryResult

if TYPE_CHECKING:
    from citra.context.turn_workspace import WorkspaceContext


logger = logging.getLogger(__name__)

_COMMAND = "ruff"


class RuffRuntimeDiscovery(RuntimeDiscovery):
    """Expose the host Ruff launcher and supporting runtime read-only.

    Ruff is resolved before Bubblewrap is created. If the host executable is
    unavailable, discovery is advisory: a warning is logged and no bind is
    contributed.
    """

    @classmethod
    def discover(cls) -> RuntimeDiscoveryResult:
        executable_raw = shutil.which(_COMMAND)
        if executable_raw is None:
            logger.warning(
                "Runtime discovery could not find %r on the host PATH; "
                "Ruff will not receive automatic read-only sandbox binds.",
                _COMMAND,
            )
            return RuntimeDiscoveryResult()

        return RuntimeDiscoveryResult(
            readonly_binds=_runtime_binds(Path(executable_raw)),
        )


def _runtime_binds(executable: Path) -> tuple[Path, ...]:
    """Return the minimal host paths needed to launch one discovered tool."""
    executable = executable.expanduser().absolute()
    resolved = executable.resolve()

    candidates: list[Path] = [
        executable.parent,
        resolved.parent,
    ]

    _append_python_environment_root(candidates, executable)
    _append_python_environment_root(candidates, resolved)

    interpreter = _shebang_interpreter(executable)
    if interpreter is not None:
        interpreter = interpreter.expanduser().absolute()
        candidates.extend((interpreter.parent, interpreter.resolve().parent))
        _append_python_environment_root(candidates, interpreter)
        _append_python_environment_root(candidates, interpreter.resolve())

    return _minimal_existing_paths(candidates)


def _append_python_environment_root(
    candidates: list[Path],
    executable: Path,
) -> None:
    """Include an owning Python virtual environment when one is detectable."""
    parent = executable.parent
    if parent.name not in {"bin", "Scripts"}:
        return

    environment_root = parent.parent
    if (environment_root / "pyvenv.cfg").is_file():
        candidates.append(environment_root)


def _shebang_interpreter(executable: Path) -> Path | None:
    """Resolve an absolute interpreter referenced by a script shebang.

    This covers wrappers installed in locations such as ``~/.local/bin`` whose
    shebang points into a uv/pip-managed virtual environment hidden by the
    sandbox. ``/usr/bin/env`` shebangs are resolved through the host PATH when
    they name an interpreter explicitly.
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


def _minimal_existing_paths(candidates: list[Path]) -> tuple[Path, ...]:
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
    try:
        path.absolute().relative_to(parent.absolute())
        return True
    except ValueError:
        return False
