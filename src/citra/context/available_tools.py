from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from typing import TYPE_CHECKING

from .runtime import CopyPolicy, RuntimeAsset, ToolDefinition


_DEFAULT_COMMANDS = (
    "bash",
    "git",
    "python3",
    "python",
    "node",
    "npm",
    "npx",
)


def default_tool_definitions() -> tuple[ToolDefinition, ...]:
    """Declare the small host command set required by built-in tools.

    Commands are exposed as explicit read-only assets so availability checks
    and the runtime manifest describe the same executable paths the sandbox
    will use. Missing optional commands remain unavailable without preventing
    startup.
    """
    definitions: list[ToolDefinition] = []
    for command in _DEFAULT_COMMANDS:
        discovered = shutil.which(command)
        source = (
            Path(discovered).absolute()
            if discovered is not None
            else Path("/__citra_missing_commands__") / command
        )
        asset_id = f"host-command:{command}"
        asset = RuntimeAsset(
            id=asset_id,
            source=source,
            destination=PurePosixPath("commands") / command,
            policy=CopyPolicy.BIND_ONLY,
            required=False,
            bind_target=source,
            priority=300,
        )
        definitions.append(
            ToolDefinition(
                id=f"command:{command}",
                commands=(command,),
                assets=(asset,),
                command_assets={command: asset_id},
                health_check=("{executable}", "--version"),
            )
        )
    return tuple(definitions)

def default_runtime_assets(
    *,
    browser_path: str | Path | None = None,
) -> tuple[RuntimeAsset, ...]:
    """Declare narrow immutable host compatibility assets.

    These replace the historical blanket read-only bind of ``/``.  They are
    deliberately recorded in the runtime manifest even when their policy is
    bind-only, so the effective filesystem can be diagnosed precisely.
    """
    assets: list[RuntimeAsset] = []
    candidates: list[tuple[str, Path, bool]] = [
        ("os-usr", Path("/usr"), True),
        ("os-bin", Path("/bin"), True),
        ("os-lib", Path("/lib"), False),
        ("os-lib64", Path("/lib64"), False),
        ("etc-ssl", Path("/etc/ssl"), False),
        ("etc-ca-certificates", Path("/etc/ca-certificates"), False),
        ("etc-alternatives", Path("/etc/alternatives"), False),
        ("etc-passwd", Path("/etc/passwd"), False),
        ("etc-group", Path("/etc/group"), False),
        ("etc-nsswitch", Path("/etc/nsswitch.conf"), False),
        ("etc-ld-cache", Path("/etc/ld.so.cache"), False),
        ("etc-ld-conf", Path("/etc/ld.so.conf"), False),
        ("etc-ld-conf-d", Path("/etc/ld.so.conf.d"), False),
        ("etc-hosts", Path("/etc/hosts"), False),
        ("etc-localtime", Path("/etc/localtime"), False),
    ]

    install_root = os.environ.get("CITRA_INSTALL_ROOT")
    inferred_install = Path(__file__).resolve().parents[3]
    candidates.append(
        (
            "citra-install",
            Path(install_root).expanduser() if install_root else inferred_install,
            True,
        )
    )
    candidates.append(("citra-python-prefix", Path(sys.prefix), True))
    if Path(sys.base_prefix) != Path(sys.prefix):
        candidates.append(
            ("citra-python-base-prefix", Path(sys.base_prefix), True)
        )

    for asset_id, source, required in candidates:
        absolute = source.absolute()
        assets.append(
            RuntimeAsset(
                id=asset_id,
                source=absolute,
                destination=PurePosixPath("compat") / asset_id,
                policy=CopyPolicy.BIND_ONLY,
                required=required,
                bind_target=absolute,
                priority=200,
            )
        )

    if browser_path is not None:
        expanded = Path(browser_path).expanduser().absolute()
        assets.append(
            RuntimeAsset(
                id="playwright-browsers",
                source=expanded,
                destination=PurePosixPath("browsers/playwright"),
                policy=CopyPolicy.COPY_OR_BIND,
                required=False,
                bind_target=expanded,
                priority=10,
            )
        )
    return tuple(assets)


def _is_native_executable(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(4)
    except OSError:
        return False
    return header == b"\x7fELF" or header in {
        b"\xcf\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xca\xfe\xba\xbe",
    }


def _is_under_system_root(path: Path) -> bool:
    for root in (Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64")):
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False

