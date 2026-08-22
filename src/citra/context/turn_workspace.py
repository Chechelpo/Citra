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
from .workspace_changes import WorkspaceChanges


_PATH_ALIAS_PATTERN = re.compile(
    r"^@([a-z_]+)(?:/(.*))?$"
)


class AvailablePathAlias(str, Enum):
    """A Citra-defined path alias, similar to ``~/`` or ``./``."""

    WORKSPACE = "workspace"
    SOURCE = "source"
    HOME = "home"
    TMP = "tmp"
    CACHE = "cache"
    CONFIG = "config"
    DATA = "data"
    RUNTIME = "runtime"

    def as_alias(self) -> str:
        return f"@{self.value}"


@dataclass(frozen=True)
class WorkspaceContext:
    """
    Filesystem context for one running Citra process.

    ``source_workspace`` is permanent and read-only to general tools.
    ``workspace`` is an initially empty working directory inside ``root``.
    ``changes`` selectively materializes source files and is the sole
    service allowed to apply staged file updates back to the source. The
    filesystem survives every conversation turn and is removed only when
    the owning application lifecycle closes.
    """

    source_workspace: Path
    workspace: Path

    root: Path
    home: Path
    tmp: Path
    cache: Path
    config: Path
    data: Path
    state: Path
    runtime: Path

    changes: WorkspaceChanges

    @classmethod
    def create(
        cls,
        config: WorkspaceContextConfig,
        workspace: str | Path,
    ) -> WorkspaceContext:
        source_workspace = Path(
            config.permanent_workspace
            or workspace
        ).expanduser().resolve()

        if not source_workspace.is_dir():
            raise NotADirectoryError(
                "Source workspace does not exist: "
                f"{source_workspace}"
            )

        temp_base = config.temporary_workspace

        if temp_base is not None:
            temp_base_path = Path(
                temp_base
            ).expanduser().resolve()
            temp_base_path.mkdir(
                parents=True,
                exist_ok=True,
            )
            root = Path(
                tempfile.mkdtemp(
                    prefix="citra-process-",
                    dir=str(temp_base_path),
                )
            ).resolve()
        else:
            root = Path(
                tempfile.mkdtemp(
                    prefix="citra-process-"
                )
            ).resolve()

        if cls._is_within(
            source_workspace,
            root,
        ):
            shutil.rmtree(
                root,
                ignore_errors=True,
            )
            raise ValueError(
                "The temporary agent root cannot be created inside "
                "the source workspace."
            )

        workspace_path = root / "workspace"
        home = root / "home"
        tmp = root / "tmp"
        cache = root / "cache"
        config_dir = root / "config"
        data = root / "data"
        state = root / "state"
        runtime = root / "runtime"

        try:
            for directory in (
                workspace_path,
                home,
                tmp,
                cache,
                config_dir,
                data,
                state,
                runtime,
            ):
                directory.mkdir(
                    parents=True,
                    exist_ok=False,
                )

            # Bubblewrap overlays this empty path with the permanent source.
            # Scoped tools resolve @source directly instead.
            (workspace_path / "@source").mkdir()

            changes = WorkspaceChanges.create(
                source_workspace=source_workspace,
                workspace=workspace_path,
                state=state,
                home=home,
            )

            return cls(
                source_workspace=source_workspace,
                workspace=workspace_path,
                root=root,
                home=home,
                tmp=tmp,
                cache=cache,
                config=config_dir,
                data=data,
                state=state,
                runtime=runtime,
                changes=changes,
            )

        except Exception:
            shutil.rmtree(
                root,
                ignore_errors=True,
            )
            raise

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        """Roots addressable by model-facing scoped filesystem tools.

        ``root`` and ``state`` are deliberately absent: they contain Citra's
        trusted baseline/index bookkeeping. Merely knowing an absolute path
        to that control plane must not make it readable.
        """
        return (
            self.source_workspace,
            self.workspace,
            self.home,
            self.tmp,
            self.cache,
            self.config,
            self.data,
            self.runtime,
        )

    @property
    def writable_roots(self) -> tuple[Path, ...]:
        """Return the roots writable by ordinary tools."""
        return (
            self.workspace,
            self.home,
            self.tmp,
            self.cache,
            self.config,
            self.data,
            self.runtime,
        )

    def resolve_path(
        self,
        path: str | Path,
    ) -> Path:
        """
        Resolve aliases and relative paths within the allowed roots.

        Relative paths resolve from the isolated workspace. ``~/`` resolves
        from the disposable agent home. Both ``@source/x`` and
        ``./@source/x`` resolve from the permanent source workspace.
        """
        raw = str(path)
        alias_raw = raw

        while alias_raw.startswith("./"):
            alias_raw = alias_raw[2:]

        alias_match = _PATH_ALIAS_PATTERN.fullmatch(
            alias_raw
        )

        if alias_match:
            alias = alias_match.group(1)
            remainder = alias_match.group(2)
            base = self._alias_root(
                alias
            )
            resolved = (
                base
                if not remainder
                else base / remainder
            ).resolve()
        elif raw == "~" or raw.startswith("~/"):
            remainder = "" if raw == "~" else raw[2:]
            resolved = (
                self.home
                if not remainder
                else self.home / remainder
            ).resolve()
        else:
            candidate = Path(raw)

            if not candidate.is_absolute():
                candidate = self.workspace / candidate

            resolved = candidate.resolve()

        self.require_allowed_path(
            resolved
        )
        return resolved

    def require_allowed_path(
        self,
        path: str | Path,
    ) -> Path:
        resolved = Path(path).resolve()

        if any(
            self._is_within(root, resolved)
            for root in self.allowed_roots
        ):
            return resolved

        raise ValueError(
            "Path is outside @source and the lifecycle-scoped agent "
            f"filesystem: {resolved}"
        )

    def is_valid_read_path(
        self,
        path: str | Path,
    ) -> bool:
        try:
            self.require_allowed_path(
                path
            )
            return True
        except ValueError:
            return False

    def require_writable_path(
        self,
        path: str | Path,
    ) -> Path:
        resolved = self.resolve_path(
            path
        )

        if any(
            self._is_within(root, resolved)
            for root in self.writable_roots
        ):
            return resolved

        raise ValueError(
            "Path is read-only: "
            f"{self.display_path(resolved)}"
        )

    def display_path(
        self,
        path: str | Path,
    ) -> str:
        resolved = Path(path).resolve()

        try:
            relative = resolved.relative_to(
                self.workspace
            )
            return (
                "."
                if not relative.parts
                else relative.as_posix()
            )
        except ValueError:
            pass

        aliases = (
            (
                AvailablePathAlias.SOURCE.value,
                self.source_workspace,
            ),
            (
                AvailablePathAlias.TMP.value,
                self.tmp,
            ),
            (
                AvailablePathAlias.HOME.value,
                self.home,
            ),
            (
                AvailablePathAlias.CACHE.value,
                self.cache,
            ),
            (
                AvailablePathAlias.CONFIG.value,
                self.config,
            ),
            (
                AvailablePathAlias.DATA.value,
                self.data,
            ),
            (
                AvailablePathAlias.RUNTIME.value,
                self.runtime,
            ),
        )

        for alias, root in aliases:
            try:
                relative = resolved.relative_to(
                    root
                )
            except ValueError:
                continue

            if not relative.parts:
                return f"@{alias}"

            return (
                f"@{alias}/"
                f"{relative.as_posix()}"
            )

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
            "NODE_PATH",
            "NVM_BIN",
            "NVM_DIR",
            "PYENV_ROOT",
            "PYTHONPATH",
            "RUSTUP_HOME",
            "VIRTUAL_ENV",
        ):
            value = os.environ.get(
                name
            )

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
                "XDG_STATE_HOME": str(self.data / "xdg-state"),
                "XDG_RUNTIME_DIR": str(self.runtime),
                "CITRA_WORKSPACE": str(self.workspace),
                "CITRA_SOURCE": str(self.source_workspace),
                "CITRA_AGENT_ROOT": str(self.root),
                "CITRA_TMP": str(self.tmp),
                "CITRA_CACHE": str(self.cache),
            }
        )

        if extra:
            env.update(
                extra
            )

        return env

    def cleanup(self) -> None:
        shutil.rmtree(
            self.root,
            ignore_errors=True,
        )

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

    def _alias_root(
        self,
        alias: str,
    ) -> Path:
        aliases: dict[str, Path] = {
            AvailablePathAlias.WORKSPACE.value: self.workspace,
            AvailablePathAlias.SOURCE.value: self.source_workspace,
            AvailablePathAlias.HOME.value: self.home,
            AvailablePathAlias.TMP.value: self.tmp,
            AvailablePathAlias.CACHE.value: self.cache,
            AvailablePathAlias.CONFIG.value: self.config,
            AvailablePathAlias.DATA.value: self.data,
            AvailablePathAlias.RUNTIME.value: self.runtime,
        }

        try:
            return aliases[alias]
        except KeyError as error:
            raise ValueError(
                f"Unknown workspace path alias: @{alias}"
            ) from error

    @staticmethod
    def _is_within(
        root: Path,
        path: Path,
    ) -> bool:
        try:
            path.relative_to(
                root
            )
        except ValueError:
            return False

        return True
