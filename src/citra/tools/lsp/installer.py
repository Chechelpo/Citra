"""Sandboxed Agent Runtime language-server installation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

from .servers.base import InstallCandidate, ServerDefinition


_MANAGER_PRIORITY = (
    "npm",
    "gem",
    "go",
    "cargo",
)


class _Sandbox(Protocol):
    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None,
        timeout: int,
        network: bool,
        environment: dict[str, str] | None = None,
    ) -> object: ...


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


def available_managers(
    resolver: Callable[[str], str | None],
) -> tuple[str, ...]:
    return tuple(manager for manager in _MANAGER_PRIORITY if resolver(manager) is not None)


def candidate_for(
    definition: ServerDefinition,
    managers: Iterable[str] | None = None,
) -> InstallCandidate | None:
    available = set(managers or ())
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
    resolver: Callable[[str], str | None],
    sandbox: _Sandbox | None = None,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    timeout: int = 300,
) -> InstallResult:
    command = candidate.command
    if dry_run:
        return InstallResult(
            server_id=definition.id,
            command=command,
            dry_run=True,
            returncode=None,
            output="dry-run: " + " ".join(command),
            executable_found=resolver(definition.executable),
        )

    if sandbox is None or cwd is None:
        raise RuntimeError(
            "Language-server installation requires an Agent Runtime sandbox."
        )

    print("$ " + " ".join(command), flush=True)
    try:
        completed = sandbox.run(
            command,
            cwd=cwd,
            timeout=timeout,
            network=True,
            environment=environment,
        )
        returncode = int(getattr(completed, "returncode", 1))
        output = str(getattr(completed, "output", ""))
    except Exception as error:
        returncode = 127
        output = str(error)

    executable = resolver(definition.executable)
    return InstallResult(
        server_id=definition.id,
        command=command,
        dry_run=False,
        returncode=returncode,
        output=output.strip(),
        executable_found=executable,
    )
