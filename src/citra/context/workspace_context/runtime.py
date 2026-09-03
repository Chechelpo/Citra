"""Declarative process-lifetime runtime provisioning.

The Agent Runtime is not a container image.  This module plans a small,
inspectable immutable tool layer from explicitly declared assets and records
the host paths that must instead be exposed as read-only fallback binds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import stat
import subprocess
from threading import Lock
import time
from typing import Any, Callable, Mapping, Sequence

from citra.sandbox.sandbox import WorkspaceSandbox


class RuntimeProvisionError(RuntimeError):
    """A required runtime asset could not be provisioned safely."""


class RuntimeProcessSupervisor:
    """Aggregate ownership for every process started inside one runtime."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._processes: dict[int, subprocess.Popen[Any]] = {}
        self._closing = False

    def register(self, process: subprocess.Popen[Any]) -> bool:
        with self._lock:
            if self._closing:
                accepted = False
            else:
                self._processes[process.pid] = process
                accepted = True
        if not accepted:
            self._signal(process, signal.SIGKILL)
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        return accepted

    def unregister(self, process: subprocess.Popen[Any]) -> None:
        with self._lock:
            self._processes.pop(process.pid, None)

    def begin_closing(self) -> None:
        with self._lock:
            self._closing = True

    @property
    def active_count(self) -> int:
        with self._lock:
            completed = [
                pid
                for pid, process in self._processes.items()
                if process.poll() is not None
            ]
            for pid in completed:
                self._processes.pop(pid, None)
            return len(self._processes)

    def terminate_all(
        self,
        *,
        force: bool,
        grace_seconds: float = 1.0,
    ) -> None:
        """Terminate all process groups within one aggregate time bound."""
        self.begin_closing()
        with self._lock:
            processes = tuple(self._processes.values())
        signal_value = signal.SIGKILL if force else signal.SIGTERM
        for process in processes:
            if process.poll() is None:
                self._signal(process, signal_value)

        deadline = time.monotonic() + (
            0.2 if force else max(0.0, grace_seconds)
        )
        while time.monotonic() < deadline:
            if all(process.poll() is not None for process in processes):
                break
            time.sleep(0.02)

        if not force:
            for process in processes:
                if process.poll() is None:
                    self._signal(process, signal.SIGKILL)
            kill_deadline = time.monotonic() + 0.2
            while time.monotonic() < kill_deadline:
                if all(process.poll() is not None for process in processes):
                    break
                time.sleep(0.02)

        survivors: list[int] = []
        for process in processes:
            # poll() reaps a completed child without adding a per-process wait
            # to the aggregate shutdown bound.
            if process.poll() is None:
                survivors.append(process.pid)
            else:
                self.unregister(process)
        if survivors:
            raise RuntimeError(
                "Agent Runtime child processes survived SIGKILL: "
                + ", ".join(str(pid) for pid in survivors)
            )

    @staticmethod
    def _signal(process: subprocess.Popen[Any], value: signal.Signals) -> None:
        try:
            os.killpg(process.pid, value)
        except ProcessLookupError:
            pass


class CopyPolicy(str, Enum):
    """How one semantic runtime asset may be provisioned."""

    COPY_REQUIRED = "copy-required"
    COPY_OR_BIND = "copy-or-bind"
    BIND_ONLY = "bind-only"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class RuntimeAsset:
    """One indivisible runtime asset declared by a tool adapter.

    ``destination`` is relative to the immutable runtime layer.  A fallback
    bind normally retains the host-visible path spelling in ``bind_target``;
    this is important for shebangs and runtimes containing absolute paths.
    """

    id: str
    source: Path
    destination: PurePosixPath
    priority: int = 100
    policy: CopyPolicy = CopyPolicy.COPY_OR_BIND
    required: bool = True
    bind_target: Path | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Runtime asset id cannot be empty.")
        if (
            not self.destination.parts
            or self.destination.is_absolute()
            or ".." in self.destination.parts
        ):
            raise ValueError(
                f"Runtime asset destination must remain relative: {self.destination}"
            )


@dataclass(frozen=True)
class ToolDefinition:
    """Declarative discovery/provisioning contract for one tool."""

    id: str
    commands: tuple[str, ...]
    assets: tuple[RuntimeAsset, ...]
    command_assets: Mapping[str, str]
    health_check: tuple[str, ...] | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    copy_priority: int = 100

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Tool definition id cannot be empty.")
        if not self.commands:
            raise ValueError(f"Tool {self.id!r} must declare at least one command.")
        asset_ids = {asset.id for asset in self.assets}
        missing = [
            asset_id
            for asset_id in self.command_assets.values()
            if asset_id not in asset_ids
        ]
        if missing:
            raise ValueError(
                f"Tool {self.id!r} references undeclared command assets: {missing}"
            )


@dataclass(frozen=True)
class AssetProvision:
    id: str
    source: Path
    mode: str
    size_bytes: int
    runtime_path: Path | None = None
    bind_target: Path | None = None
    reason: str | None = None

    def visible_path(self) -> Path | None:
        if self.mode == "copy":
            return self.runtime_path
        if self.mode == "ro-bind":
            return self.bind_target
        return None


@dataclass
class ProvisionedTool:
    id: str
    commands: dict[str, Path]
    mode: str
    health: str = "not-checked"
    health_detail: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.commands) and self.health != "failed"


@dataclass
class RuntimeProvisioning:
    """Resolved runtime manifest plus the process-local command resolver."""

    runtime_root: Path
    budget_bytes: int
    copied_bytes: int
    assets: dict[str, AssetProvision]
    tools: dict[str, ProvisionedTool]
    definitions: dict[str, ToolDefinition]
    warnings: list[str] = field(default_factory=list)

    def has_command(self, command: str) -> bool:
        return self.resolve_command(command) is not None

    def resolve_command(self, command: str) -> Path | None:
        for tool in self.tools.values():
            if tool.available and command in tool.commands:
                return tool.commands[command]
        return None

    def asset_path(self, asset_id: str) -> Path | None:
        provision = self.assets.get(asset_id)
        return provision.visible_path() if provision is not None else None

    def register_staged_command(self, command: str, path: Path) -> None:
        """Advertise a command installed into the mutable dependency layer."""
        self.tools[f"staged:{command}"] = ProvisionedTool(
            id=f"staged:{command}",
            commands={command: path},
            mode="dependency-environment",
            health="not-configured",
        )

    @property
    def readonly_binds(self) -> tuple[tuple[Path, Path], ...]:
        seen: set[tuple[str, str]] = set()
        binds: list[tuple[Path, Path]] = []
        for asset in self.assets.values():
            if asset.mode != "ro-bind" or asset.bind_target is None:
                continue
            key = (str(asset.source), str(asset.bind_target))
            if key in seen:
                continue
            seen.add(key)
            binds.append((asset.source, asset.bind_target))
        return tuple(binds)

    def health_check_tools(
        self,
        sandbox: WorkspaceSandbox,
        *,
        cwd: Path,
        timeout: int = 5,
    ) -> None:
        """Run declared checks inside the same sandbox used by agent tools."""
        for tool_id, definition in self.definitions.items():
            provisioned = self.tools.get(tool_id)
            if provisioned is None or not provisioned.commands:
                continue
            if definition.health_check is None:
                provisioned.health = "not-configured"
                continue
            executable = next(iter(provisioned.commands.values()))
            command = tuple(
                argument.replace("{executable}", str(executable))
                for argument in definition.health_check
            )
            try:
                result = sandbox.run(
                    command,
                    cwd=cwd,
                    timeout=timeout,
                    network=False,
                )
                returncode = result.returncode
                output = result.output.strip()
                if returncode == 0:
                    provisioned.health = "passed"
                    provisioned.health_detail = output[:500] or None
                else:
                    provisioned.health = "failed"
                    provisioned.health_detail = (
                        output[:500] or f"exit code {returncode}"
                    )
            except Exception as error:
                provisioned.health = "failed"
                provisioned.health_detail = str(error)[:500]

    def as_manifest(self) -> dict[str, object]:
        return {
            "provisioning_budget_bytes": self.budget_bytes,
            "provisioning_copied_bytes": self.copied_bytes,
            "assets": {
                asset_id: {
                    "source": str(asset.source),
                    "mode": asset.mode,
                    "size_bytes": asset.size_bytes,
                    "runtime_path": (
                        str(asset.runtime_path) if asset.runtime_path else None
                    ),
                    "bind_target": (
                        str(asset.bind_target) if asset.bind_target else None
                    ),
                    "reason": asset.reason,
                }
                for asset_id, asset in sorted(self.assets.items())
            },
            "tools": {
                tool_id: {
                    "available": tool.available,
                    "mode": tool.mode,
                    "commands": {
                        name: str(path)
                        for name, path in sorted(tool.commands.items())
                    },
                    "health": tool.health,
                    "health_detail": tool.health_detail,
                }
                for tool_id, tool in sorted(self.tools.items())
            },
            "warnings": list(self.warnings),
        }


class RuntimeProvisioner:
    """Plan and materialize declared assets against a hard copy budget."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        copy_budget_bytes: int,
        measure: Callable[[Path], int] | None = None,
    ) -> None:
        if copy_budget_bytes < 0:
            raise ValueError("Runtime provisioning copy budget cannot be negative.")
        self.runtime_root = runtime_root
        self.copy_budget_bytes = copy_budget_bytes
        self._measure = measure or measure_asset

    def provision(
        self,
        definitions: Sequence[ToolDefinition],
        *,
        standalone_assets: Sequence[RuntimeAsset] = (),
    ) -> RuntimeProvisioning:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        definition_map = {definition.id: definition for definition in definitions}
        if len(definition_map) != len(definitions):
            raise ValueError("Tool definition ids must be unique.")

        declared: dict[str, RuntimeAsset] = {}
        for asset in standalone_assets:
            self._register_asset(declared, asset)
        for definition in definitions:
            for asset in definition.assets:
                self._register_asset(declared, asset)
        self._validate_destinations(tuple(declared.values()))

        copied_bytes = 0
        provisions: dict[str, AssetProvision] = {}
        warnings: list[str] = []

        for asset in sorted(
            declared.values(),
            key=lambda item: (-item.priority, item.id),
        ):
            source = asset.source.expanduser().absolute()
            if not os.path.lexists(source):
                message = f"Runtime asset {asset.id!r} is unavailable: {source}"
                if asset.required and asset.policy is not CopyPolicy.OPTIONAL:
                    raise RuntimeProvisionError(message)
                else:
                    provisions[asset.id] = AssetProvision(
                        id=asset.id,
                        source=source,
                        mode="unavailable",
                        size_bytes=0,
                        reason="optional asset not present",
                    )
                continue

            if asset.policy is CopyPolicy.BIND_ONLY:
                provisions[asset.id] = self._bind(
                    asset,
                    source,
                    0,
                    "bind-only asset; copy size not measured",
                )
                continue

            try:
                size = self._measure(source)
            except Exception as error:
                if asset.policy is CopyPolicy.OPTIONAL:
                    provisions[asset.id] = AssetProvision(
                        id=asset.id,
                        source=source,
                        mode="unavailable",
                        size_bytes=0,
                        reason=f"optional asset could not be measured: {error}",
                    )
                    continue
                if asset.policy in {CopyPolicy.COPY_OR_BIND, CopyPolicy.BIND_ONLY}:
                    provisions[asset.id] = self._bind(asset, source, 0, str(error))
                    warnings.append(
                        f"Could not measure {asset.id!r}; using read-only bind: {error}"
                    )
                    continue
                raise RuntimeProvisionError(
                    f"Could not measure required runtime asset {asset.id!r}: {error}"
                ) from error

            fits = copied_bytes + size <= self.copy_budget_bytes
            if asset.policy is CopyPolicy.OPTIONAL and not fits:
                provisions[asset.id] = AssetProvision(
                    id=asset.id,
                    source=source,
                    mode="unavailable",
                    size_bytes=size,
                    reason="optional asset exceeds remaining copy budget",
                )
                continue
            if asset.policy is CopyPolicy.COPY_REQUIRED and not fits:
                raise RuntimeProvisionError(
                    f"Required copied asset {asset.id!r} ({size} bytes) exceeds "
                    f"the remaining provisioning budget "
                    f"({self.copy_budget_bytes - copied_bytes} bytes)."
                )
            if not fits:
                provisions[asset.id] = self._bind(
                    asset,
                    source,
                    size,
                    "copy budget exceeded",
                )
                continue

            destination = self.runtime_root.joinpath(*asset.destination.parts)
            try:
                copy_asset(source, destination)
            except Exception as error:
                discard_asset(destination)
                if asset.policy is CopyPolicy.OPTIONAL:
                    provisions[asset.id] = AssetProvision(
                        id=asset.id,
                        source=source,
                        mode="unavailable",
                        size_bytes=size,
                        reason=f"optional asset copy failed: {error}",
                    )
                    continue
                if asset.policy is CopyPolicy.COPY_OR_BIND:
                    provisions[asset.id] = self._bind(
                        asset,
                        source,
                        size,
                        f"copy failed: {error}",
                    )
                    warnings.append(
                        f"Could not copy {asset.id!r}; using read-only bind: {error}"
                    )
                    continue
                raise RuntimeProvisionError(
                    f"Could not copy required runtime asset {asset.id!r}: {error}"
                ) from error

            copied_bytes += size
            provisions[asset.id] = AssetProvision(
                id=asset.id,
                source=source,
                mode="copy",
                size_bytes=size,
                runtime_path=destination,
            )

        tools: dict[str, ProvisionedTool] = {}
        for definition in definitions:
            required_failed = any(
                asset.required
                and asset.policy is not CopyPolicy.OPTIONAL
                and provisions[asset.id].mode == "unavailable"
                for asset in definition.assets
            )
            commands: dict[str, Path] = {}
            if not required_failed:
                for command in definition.commands:
                    asset_id = definition.command_assets.get(command)
                    if asset_id is None:
                        continue
                    visible = provisions[asset_id].visible_path()
                    if visible is not None:
                        commands[command] = visible
            modes = {
                provisions[asset.id].mode
                for asset in definition.assets
                if provisions[asset.id].mode != "unavailable"
            }
            mode = next(iter(modes)) if len(modes) == 1 else "mixed"
            tool = ProvisionedTool(
                id=definition.id,
                commands=commands,
                mode=mode if modes else "unavailable",
                health="not-checked" if commands else "failed",
                health_detail=None if commands else "required runtime assets unavailable",
            )
            tools[definition.id] = tool

        freeze_runtime_layer(self.runtime_root)
        return RuntimeProvisioning(
            runtime_root=self.runtime_root,
            budget_bytes=self.copy_budget_bytes,
            copied_bytes=copied_bytes,
            assets=provisions,
            tools=tools,
            definitions=definition_map,
            warnings=warnings,
        )

    @staticmethod
    def _register_asset(
        declared: dict[str, RuntimeAsset],
        asset: RuntimeAsset,
    ) -> None:
        existing = declared.get(asset.id)
        if existing is not None and existing != asset:
            raise ValueError(f"Runtime asset id {asset.id!r} is declared inconsistently.")
        declared[asset.id] = asset

    @staticmethod
    def _validate_destinations(assets: Sequence[RuntimeAsset]) -> None:
        ordered = sorted(assets, key=lambda asset: asset.destination.parts)
        for index, asset in enumerate(ordered):
            for other in ordered[index + 1 :]:
                if (
                    asset.destination == other.destination
                    or asset.destination in other.destination.parents
                    or other.destination in asset.destination.parents
                ):
                    raise ValueError(
                        "Runtime asset destinations must not overlap: "
                        f"{asset.id!r}={asset.destination}, "
                        f"{other.id!r}={other.destination}"
                    )

    @staticmethod
    def _bind(
        asset: RuntimeAsset,
        source: Path,
        size: int,
        reason: str | None = None,
    ) -> AssetProvision:
        target = (asset.bind_target or source).expanduser().absolute()
        return AssetProvision(
            id=asset.id,
            source=source,
            mode="ro-bind",
            size_bytes=size,
            bind_target=target,
            reason=reason,
        )


def measure_asset(path: Path) -> int:
    """Measure regular files and symlinks without following directory links."""
    metadata = path.lstat()
    if stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        return metadata.st_size
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Unsupported runtime asset type: {path}")
    total = 0
    stack = [path]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    stack.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    total += info.st_size
                else:
                    raise ValueError(
                        f"Unsupported entry in runtime asset: {entry.path}"
                    )
    return total


def copy_asset(source: Path, destination: Path) -> None:
    """Copy one asset without following symlinks or recreating special files."""
    boundary = source if source.is_dir() and not source.is_symlink() else source.parent
    _copy_asset(source, destination, source_boundary=boundary.resolve())


def _copy_asset(
    source: Path,
    destination: Path,
    *,
    source_boundary: Path,
) -> None:
    metadata = source.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(source)
        if os.path.isabs(target):
            raise ValueError(f"Runtime asset symlink escapes its asset: {source}")
        resolved_target = (source.parent / target).resolve(strict=False)
        try:
            resolved_target.relative_to(source_boundary)
        except ValueError as error:
            raise ValueError(
                f"Runtime asset symlink escapes its asset: {source} -> {target}"
            ) from error
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(target)
        return
    if stat.S_ISREG(metadata.st_mode):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=False)
        return
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Unsupported runtime asset type: {source}")
    destination.mkdir(parents=True, exist_ok=False)
    with os.scandir(source) as entries:
        for entry in entries:
            _copy_asset(
                Path(entry.path),
                destination / entry.name,
                source_boundary=source_boundary,
            )
    shutil.copystat(source, destination, follow_symlinks=False)


def discard_asset(path: Path) -> None:
    """Remove an incomplete asset without following a destination symlink."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        for directory, _dirnames, _filenames in os.walk(path, topdown=False):
            current = Path(directory)
            if current.is_symlink():
                continue
            try:
                current.chmod(stat.S_IMODE(current.stat().st_mode) | 0o700)
            except OSError:
                pass
        try:
            path.chmod(stat.S_IMODE(path.stat().st_mode) | 0o700)
        except OSError:
            pass
        shutil.rmtree(path)


def freeze_runtime_layer(root: Path) -> None:
    """Remove write bits from copied runtime content after provisioning."""
    for directory, dirnames, filenames in os.walk(root, topdown=False):
        base = Path(directory)
        for filename in filenames:
            path = base / filename
            if path.is_symlink():
                continue
            mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod(mode & ~0o222)
        for dirname in dirnames:
            path = base / dirname
            if path.is_symlink():
                continue
            mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod(mode & ~0o222)
    mode = stat.S_IMODE(root.stat().st_mode)
    root.chmod(mode & ~0o222)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write controller metadata without leaving a partially written manifest."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
