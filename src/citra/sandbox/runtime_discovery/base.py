from __future__ import annotations

from abc import ABC, abstractmethod

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from citra.context.session_context import WorkspaceContext


_COMMANDS_TO_DISCOVER: tuple[str, ...] = (
    "curl",
)


@dataclass(frozen=True)
class RuntimeDiscoveryResult:
    """Filesystem requirements discovered before sandbox creation."""

    readonly_binds: tuple[Path, ...] = ()


class RuntimeDiscovery(ABC):
    """Base class for one host-runtime discovery utility."""

    @classmethod
    @abstractmethod
    def discover(cls) -> RuntimeDiscoveryResult:
        """Return host paths that must be exposed read-only."""


class StandardDiscovery(RuntimeDiscovery):
    """
    Discovers host runtime requirements for standard system commands.

    It resolves:
    - executable locations
    - symbolic links
    - dynamic linker dependencies

    Returned paths can be mounted read-only into the sandbox.
    """

    commands: tuple[str, ...] = _COMMANDS_TO_DISCOVER

    @classmethod
    def discover(cls) -> RuntimeDiscoveryResult:
        paths: set[Path] = set()

        for command in cls.commands:
            executable = cls._resolve_command(command)

            if executable is None:
                continue

            paths.update(cls._resolve_symlink_chain(executable))
            paths.update(cls._discover_shared_dependencies(executable))

        return RuntimeDiscoveryResult(
            readonly_binds=tuple(sorted(paths)),
        )

    @staticmethod
    def _resolve_command(command: str) -> Path | None:
        """
        Resolve command through PATH.
        """

        location = shutil.which(command)

        if location is None:
            return None

        return Path(location).resolve()

    @staticmethod
    def _resolve_symlink_chain(path: Path) -> set[Path]:
        """
        Collect every path involved in a symlink chain.

        Example:
            /usr/bin/curl
            -> /usr/bin/curl.real
            -> /usr/libexec/curl

        All are returned.
        """

        result: set[Path] = set()

        current = path

        while True:
            result.add(current)

            if not current.is_symlink():
                break

            target = current.readlink()

            if not target.is_absolute():
                current = current.parent / target
            else:
                current = target

            current = current.resolve()

        return result

    @staticmethod
    def _discover_shared_dependencies(
        executable: Path,
    ) -> set[Path]:
        """
        Use ldd to discover ELF runtime dependencies.

        Example output:

            libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
            libssl.so => /lib/x86_64-linux-gnu/libssl.so

        """

        result: set[Path] = set()

        try:
            output = subprocess.check_output(
                ["ldd", str(executable)],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return result

        for line in output.splitlines():
            parts = line.split("=>")

            if len(parts) != 2:
                continue

            candidate = parts[1].strip().split()[0]

            path = Path(candidate)

            if path.exists():
                result.add(path.resolve())

        return result