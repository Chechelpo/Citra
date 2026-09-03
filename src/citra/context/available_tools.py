"""Translate modular host discovery into sandbox runtime assets.

The same discovery result drives both modes. ``FULL_SANDBOX`` copies every
discovered runtime root into ``/runtime/rootfs`` and exposes copied
compatibility paths. ``PARTIAL_SANDBOX`` exposes the corresponding host roots
through read-only mounts. Both modes publish commands exclusively through
``/runtime/bin``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path, PurePosixPath
from typing import Iterable

from citra.config.runtime_discovery import (
    RuntimeDiscoveryResult,
    aggregate_results,
    get_ro_binds,
)
from citra.sandbox.sandbox_mode import SandboxMode

from .workspace_context.runtime import CopyPolicy, RuntimeAsset, ToolDefinition


logger = logging.getLogger(__name__)


def discover_host_runtime() -> RuntimeDiscoveryResult:
    """Run and aggregate every registered runtime discovery object."""
    return aggregate_results(get_ro_binds())


def default_tool_definitions(
    *,
    mode: SandboxMode = SandboxMode.FULL_SANDBOX,
    discovery: RuntimeDiscoveryResult | None = None,
) -> tuple[ToolDefinition, ...]:
    """Create one declarative tool definition per discovered host command."""
    result = discovery or discover_host_runtime()
    assets = _discovered_assets(mode=mode, discovery=result)
    definitions: list[ToolDefinition] = []
    for command, executable in result.command_paths:
        asset = _containing_asset(executable, assets)
        if asset is None:
            logger.error(
                "Discovered command has no runtime asset",
                extra={
                    "origin": __name__,
                    "command": command,
                    "executable": str(executable),
                },
            )
            continue
        definitions.append(
            ToolDefinition(
                id=f"command:{command}",
                commands=(command,),
                assets=(asset,),
                command_assets={command: asset.id},
                command_sources={command: executable},
                health_check=None,
            )
        )
    logger.info(
        "Runtime command definitions created",
        extra={"origin": __name__, "mode": mode.name, "count": len(definitions)},
    )
    return tuple(definitions)


def default_runtime_assets(
    *,
    mode: SandboxMode = SandboxMode.FULL_SANDBOX,
    discovery: RuntimeDiscoveryResult | None = None,
    browser_path: str | Path | None = None,
) -> tuple[RuntimeAsset, ...]:
    """Declare discovered runtimes plus Citra and platform compatibility data."""
    result = discovery or discover_host_runtime()
    assets = list(_discovered_assets(mode=mode, discovery=result))
    policy = _copy_policy(mode)
    compatibility: list[tuple[str, Path, bool]] = [
        ("citra-package", Path(__file__).resolve().parents[2], True),
        ("etc-ssl", Path("/etc/ssl"), False),
        ("etc-ca-certificates", Path("/etc/ca-certificates"), False),
        ("usr-share-ca-certificates", Path("/usr/share/ca-certificates"), False),
        ("etc-passwd", Path("/etc/passwd"), False),
        ("etc-group", Path("/etc/group"), False),
        ("etc-nsswitch", Path("/etc/nsswitch.conf"), False),
        ("etc-ld-cache", Path("/etc/ld.so.cache"), False),
        ("etc-ld-conf", Path("/etc/ld.so.conf"), False),
        ("etc-ld-conf-d", Path("/etc/ld.so.conf.d"), False),
        ("etc-hosts", Path("/etc/hosts"), False),
        ("etc-resolv", Path("/etc/resolv.conf"), False),
        ("etc-localtime", Path("/etc/localtime"), False),
    ]
    for asset_id, source, required in compatibility:
        absolute = source.expanduser().absolute()
        if not absolute.exists():
            continue
        copy_source = absolute.resolve() if absolute.is_symlink() else absolute
        assets.append(
            RuntimeAsset(
                id=asset_id,
                source=copy_source,
                destination=_mirrored_destination(copy_source),
                policy=(
                    policy
                    if required or mode is SandboxMode.PARTIAL_SANDBOX
                    else CopyPolicy.OPTIONAL
                ),
                required=required,
                bind_target=absolute,
                priority=250 if required else 80,
            )
        )
    browser_candidate = (
        Path(browser_path).expanduser()
        if browser_path is not None
        else Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "~/.cache/ms-playwright")).expanduser()
    )
    if browser_candidate.exists():
        browser = browser_candidate.absolute()
        if browser.exists():
            assets.append(
                RuntimeAsset(
                    id="playwright-browsers",
                    source=browser,
                    destination=_mirrored_destination(browser),
                    policy=policy,
                    required=False,
                    bind_target=browser,
                    priority=120,
                )
            )
    result_assets = _minimal_assets(assets)
    logger.info(
        "Runtime assets created",
        extra={
            "origin": __name__,
            "mode": mode.name,
            "count": len(result_assets),
        },
    )
    return result_assets


def _discovered_assets(
    *,
    mode: SandboxMode,
    discovery: RuntimeDiscoveryResult,
) -> tuple[RuntimeAsset, ...]:
    """Convert discovered host roots into deterministic mirrored assets."""
    policy = _copy_policy(mode)
    return tuple(
        RuntimeAsset(
            id=_asset_id(path),
            source=path.expanduser().absolute(),
            destination=_mirrored_destination(path),
            policy=policy,
            required=True,
            bind_target=path.expanduser().absolute(),
            priority=300,
        )
        for path in _minimal_paths(discovery.readonly_binds)
    )


def _copy_policy(mode: SandboxMode) -> CopyPolicy:
    """Return the provisioning policy required by one sandbox mode."""
    if mode is SandboxMode.FULL_SANDBOX:
        return CopyPolicy.COPY_REQUIRED
    if mode is SandboxMode.PARTIAL_SANDBOX:
        return CopyPolicy.BIND_ONLY
    raise ValueError(f"Unsupported sandbox mode: {mode!r}")


def _minimal_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Deduplicate existing paths and remove descendants of directory roots."""
    unique = tuple(
        sorted(
            {
                path.expanduser().absolute()
                for path in paths
                if path.expanduser().exists()
            },
            key=lambda item: (len(item.parts), str(item)),
        )
    )
    return tuple(
        path
        for path in unique
        if not any(
            parent != path and parent.is_dir() and _is_within(parent, path)
            for parent in unique
        )
    )


def _minimal_assets(assets: Iterable[RuntimeAsset]) -> tuple[RuntimeAsset, ...]:
    """Remove duplicate assets and children already covered by a directory."""
    by_source: dict[Path, RuntimeAsset] = {}
    for asset in assets:
        source = asset.source.expanduser().absolute()
        existing = by_source.get(source)
        if existing is None or asset.priority > existing.priority:
            by_source[source] = asset
    ordered = sorted(
        by_source.values(),
        key=lambda item: (len(item.source.parts), str(item.source)),
    )
    return tuple(
        asset
        for asset in ordered
        if not any(
            parent is not asset
            and parent.source.is_dir()
            and _is_within(parent.source, asset.source)
            for parent in ordered
        )
    )


def _containing_asset(
    executable: Path,
    assets: tuple[RuntimeAsset, ...],
) -> RuntimeAsset | None:
    """Return the narrowest asset that contains a command entry point."""
    matches = [
        asset
        for asset in assets
        if asset.source == executable or (
            asset.source.is_dir() and _is_within(asset.source, executable)
        )
    ]
    return max(matches, key=lambda item: len(item.source.parts), default=None)


def _mirrored_destination(path: Path) -> PurePosixPath:
    """Map an absolute host path below the runtime's mirrored rootfs."""
    absolute = path.expanduser().absolute()
    parts = absolute.parts[1:] if absolute.is_absolute() else absolute.parts
    return PurePosixPath("rootfs", *parts)


def _asset_id(path: Path) -> str:
    """Create a stable collision-resistant identifier for a host path."""
    normalized = os.path.normpath(str(path.expanduser().absolute()))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    name = path.name or "root"
    return f"host:{name}:{digest}"


def _is_within(root: Path, path: Path) -> bool:
    """Return whether ``path`` is lexically below ``root``."""
    try:
        path.absolute().relative_to(root.absolute())
        return True
    except ValueError:
        return False


__all__ = [
    "default_runtime_assets",
    "default_tool_definitions",
    "discover_host_runtime",
]
