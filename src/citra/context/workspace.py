from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping

from .config_loader import WorkspaceContextConfig

_PATH_ALIAS_PATTERN = re.compile(
    r"^@([a-z_]+)(?:/(.*))?$"
)


class AvailablePathAlias(str, Enum):
    """
    A Citra-defined path alias, just like "~/" or "./".
    """
    WORKSPACE = "workspace"
    AGENT = "agent"
    HOME = "home"
    TMP = "tmp"
    CACHE = "cache"
    CONFIG = "config"
    DATA = "data"
    STATE = "state"
    RUNTIME = "runtime"

    def as_alias(self) -> str:
        return f"@{self.value}"


@dataclass(frozen=True)
class WorkspaceContext:
    """
    Class in charge of, given an agent turn:

        1. Provide read/write command scope (meaning writing/reading at a
           place) error messages (actual constraints are handled by the
           sandbox).
        2. Manage the temporary filesystem for the agent.
        3. Manage workspace.

    All filesystem path operations, regardless of scope, must pass through
    this class first.
    """
    workspace: Path

    root: Path
    home: Path
    tmp: Path
    cache: Path
    config: Path
    data: Path
    state: Path
    runtime: Path

    @classmethod
    def create(
        cls,
        config: WorkspaceContextConfig,
        workspace: str | Path,
    ) -> "WorkspaceContext":
        workspace_path = Path(workspace).resolve()

        if not workspace_path.is_dir():
            raise NotADirectoryError(
                f"Workspace does not exist: {workspace_path}"
            )

        temp_base = config.temporary_workspace

        if temp_base is not None:
            temp_base_path = Path(temp_base).resolve()
            temp_base_path.mkdir(parents=True, exist_ok=True)
            root = Path(
                tempfile.mkdtemp(
                    prefix="citra-agent-",
                    dir=str(temp_base_path),
                )
            ).resolve()
        else:
            root = Path(
                tempfile.mkdtemp(prefix="citra-agent-")
            ).resolve()

        home = root / "home"
        tmp = root / "tmp"
        cache = root / "cache"
        config_dir = root / "config"
        data = root / "data"
        state = root / "state"
        runtime = root / "runtime"

        for directory in (
            home,
            tmp,
            cache,
            config_dir,
            data,
            state,
            runtime,
        ):
            directory.mkdir(parents=True, exist_ok=False)

        return cls(
            workspace=workspace_path,
            root=root,
            home=home,
            tmp=tmp,
            cache=cache,
            config=config_dir,
            data=data,
            state=state,
            runtime=runtime,
        )

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        return (self.workspace, self.root)

    @property
    def writable_roots(self) -> tuple[Path, ...]:
        """
        Writable paths are:
            - Workspace: current workspace
            - Temporary: everything under the temporary folder created for
              the agent run.
        """
        return self.allowed_roots

    def resolve_path(self, path: str | Path) -> Path:
        """
        Resolves path aliases in the form of:
            1. @<path>/* : provided by sandbox
            2. ~/ : resolves to current user root

        If none of these matches, the path is resolved relative to the
        workspace (when not absolute) and returned. In every case the
        resolved path must fall within the allowed roots.
        """
        raw = str(path)

        alias_match = _PATH_ALIAS_PATTERN.fullmatch(raw)
        if alias_match:
            alias = alias_match.group(1)
            remainder = alias_match.group(2)
            base = self._alias_root(alias)
            resolved = (
                base if not remainder else base / remainder
            ).resolve()
        elif raw == "~" or raw.startswith("~/"):
            remainder = "" if raw == "~" else raw[2:]
            resolved = (
                self.home if not remainder else self.home / remainder
            ).resolve()
        else:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = self.workspace / candidate
            resolved = candidate.resolve()

        self.require_allowed_path(resolved)
        return resolved

    def require_allowed_path(self, path: str | Path) -> Path:
        resolved = Path(path).resolve()

        if any(
            self._is_within(root, resolved)
            for root in self.allowed_roots
        ):
            return resolved

        raise ValueError(
            "Path is outside the active workspace "
            "and temporary agent filesystem: "
            f"{resolved}"
        )

    def is_valid_read_path(self, path: str | Path) -> bool:
        """
        Returns True if the path is valid for reading. Everything in the
        PC is readable, so validity here means "within the allowed roots"
        — there is no narrower read scope.
        """
        try:
            self.require_allowed_path(path)
            return True
        except ValueError:
            return False

    def require_writable_path(self, path: str | Path) -> Path:
        """
        Returns the normalized path to a writable folder. If it is not
        contained in the allowed write folders (see writable_roots), it
        raises.
        """
        resolved = self.resolve_path(path)

        if any(
            self._is_within(root, resolved)
            for root in self.writable_roots
        ):
            return resolved

        raise ValueError(f"Path is not writable: {resolved}")

    def display_path(self, path: str | Path) -> str:
        resolved = Path(path).resolve()

        # Workspace paths are shown ./-style (relative), matching the
        # AvailablePathAlias docstring's "./" analogy.
        try:
            relative = resolved.relative_to(self.workspace)
            return "." if not relative.parts else str(relative)
        except ValueError:
            pass

        aliases = (
            (AvailablePathAlias.WORKSPACE.value, self.workspace),
            (AvailablePathAlias.TMP.value, self.tmp),
            (AvailablePathAlias.HOME.value, self.home),
            (AvailablePathAlias.CACHE.value, self.cache),
            (AvailablePathAlias.CONFIG.value, self.config),
            (AvailablePathAlias.DATA.value, self.data),
            (AvailablePathAlias.STATE.value, self.state),
            (AvailablePathAlias.RUNTIME.value, self.runtime),
            (AvailablePathAlias.AGENT.value, self.root),
        )

        for alias, root in aliases:
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue

            if not relative.parts:
                return f"@{alias}"
            return f"@{alias}/{relative.as_posix()}"

        return str(resolved)

    def environment(
        self,
        extra: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        env: dict[str, str] = {}

        for name in (
            "PATH",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TERM",
            "COLORTERM",
            "TZ",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "JAVA_HOME",
            "GOROOT",
        ):
            value = os.environ.get(name)
            if value is not None:
                env[name] = value

        env.update(
            {
                "HOME": str(self.home),
                "TMPDIR": str(self.tmp),
                "TMP": str(self.tmp),
                "TEMP": str(self.tmp),
                "XDG_CACHE_HOME": str(self.cache),
                "XDG_CONFIG_HOME": str(self.config),
                "XDG_DATA_HOME": str(self.data),
                "XDG_STATE_HOME": str(self.state),
                "XDG_RUNTIME_DIR": str(self.runtime),
                "CITRA_WORKSPACE": str(self.workspace),
                "CITRA_AGENT_ROOT": str(self.root),
                "CITRA_TMP": str(self.tmp),
                "CITRA_CACHE": str(self.cache),
            }
        )

        if extra:
            env.update(extra)

        return env

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _alias_root(self, alias: str) -> Path:
        aliases: dict[str, Path] = {
            AvailablePathAlias.WORKSPACE.value: self.workspace,
            AvailablePathAlias.AGENT.value: self.root,
            AvailablePathAlias.HOME.value: self.home,
            AvailablePathAlias.TMP.value: self.tmp,
            AvailablePathAlias.CACHE.value: self.cache,
            AvailablePathAlias.CONFIG.value: self.config,
            AvailablePathAlias.DATA.value: self.data,
            AvailablePathAlias.STATE.value: self.state,
            AvailablePathAlias.RUNTIME.value: self.runtime,
        }

        try:
            return aliases[alias]
        except KeyError as error:
            raise ValueError(
                f"Unknown workspace path alias: @{alias}"
            ) from error

    @staticmethod
    def _is_within(root: Path, path: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True


    def write_text_atomic(
        self,
        path: str | Path,
        text: str,
        *,
        encoding: str = "utf-8",
    ) -> Path:
        destination = self.require_writable_path(
            path
        )

        parent = self.require_writable_path(
            destination.parent
        )

        descriptor, temporary_raw = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=parent,
        )

        temporary = Path(
            temporary_raw
        )

        try:
            with os.fdopen(
                descriptor,
                "w",
                encoding=encoding,
            ) as file:
                file.write(
                    text
                )

            temporary.replace(
                destination
            )

        except Exception:
            temporary.unlink(
                missing_ok=True,
            )
            raise

        return destination