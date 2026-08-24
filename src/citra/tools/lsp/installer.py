"""Explicit host-side language-server installation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
from typing import Iterable

from .servers.base import InstallCandidate, ServerDefinition


_MANAGER_PRIORITY = (
    "pacman",
    "paru",
    "yay",
    "apt",
    "apt-get",
    "dnf",
    "brew",
    "npm",
    "gem",
    "go",
    "cargo",
)


@dataclass(frozen=True)
class InstallResult:
    server_id: str
    command: tuple[str, ...] | None
    dry_run: bool
    returncode: int | None
    output: str
    executable_found: str | None

    @property
    def success(self) -> bool:
        if self.dry_run:
            return True
        return self.returncode == 0 and self.executable_found is not None


def available_managers() -> tuple[str, ...]:
    return tuple(manager for manager in _MANAGER_PRIORITY if shutil.which(manager) is not None)


def candidate_for(
    definition: ServerDefinition,
    managers: Iterable[str] | None = None,
) -> InstallCandidate | None:
    available = set(managers if managers is not None else available_managers())
    candidates = {candidate.manager: candidate for candidate in definition.install_candidates}
    for manager in _MANAGER_PRIORITY:
        if manager in available and manager in candidates:
            return candidates[manager]
    return None


def execute_install(
    definition: ServerDefinition,
    candidate: InstallCandidate,
    *,
    dry_run: bool,
) -> InstallResult:
    command = candidate.command
    if dry_run:
        return InstallResult(
            server_id=definition.id,
            command=command,
            dry_run=True,
            returncode=None,
            output="dry-run: " + " ".join(command),
            executable_found=shutil.which(definition.executable),
        )

    output_parts: list[str] = []
    print("$ " + " ".join(command), flush=True)
    try:
        process = subprocess.Popen(
            command,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output_parts.append(line)
        returncode = process.wait()
    except OSError as error:
        returncode = 127
        output_parts.append(str(error))

    executable = shutil.which(definition.executable)
    return InstallResult(
        server_id=definition.id,
        command=command,
        dry_run=False,
        returncode=returncode,
        output="".join(output_parts).strip(),
        executable_found=executable,
    )
