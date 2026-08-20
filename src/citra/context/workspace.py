from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping


_PATH_ALIAS_PATTERN = re.compile(
    r"^@([a-z_]+)(?:/(.*))?$"
)


@dataclass(frozen=True)
class WorkspaceContext:
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
        workspace: str | Path,
    ) -> "WorkspaceContext":
        workspace_path = Path(
            workspace
        ).resolve()

        if not workspace_path.is_dir():
            raise NotADirectoryError(
                f"Workspace does not exist: {workspace_path}"
            )

        root = Path(
            tempfile.mkdtemp(
                prefix="citra-agent-",
            )
        ).resolve()

        home = root / "home"
        tmp = root / "tmp"
        cache = root / "cache"
        config = root / "config"
        data = root / "data"
        state = root / "state"
        runtime = root / "runtime"

        for directory in (
            home,
            tmp,
            cache,
            config,
            data,
            state,
            runtime,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=False,
            )

        return cls(
            workspace=workspace_path,
            root=root,
            home=home,
            tmp=tmp,
            cache=cache,
            config=config,
            data=data,
            state=state,
            runtime=runtime,
        )

    @property
    def allowed_roots(
        self,
    ) -> tuple[Path, ...]:
        return (
            self.workspace,
            self.root,
        )

    @property
    def writable_roots(
        self,
    ) -> tuple[Path, ...]:
        return self.allowed_roots

    def resolve_path(
        self,
        path: str | Path,
    ) -> Path:
        raw = str(
            path
        )

        alias_match = _PATH_ALIAS_PATTERN.fullmatch(
            raw
        )

        if alias_match:
            alias = alias_match.group(
                1
            )

            remainder = alias_match.group(
                2
            )

            base = self._alias_root(
                alias
            )

            resolved = (
                base
                if not remainder
                else base / remainder
            ).resolve()

        else:
            candidate = Path(
                raw
            )

            if not candidate.is_absolute():
                candidate = (
                    self.workspace
                    / candidate
                )

            resolved = candidate.resolve()

        self.require_allowed_path(
            resolved
        )

        return resolved

    def require_allowed_path(
        self,
        path: str | Path,
    ) -> Path:
        resolved = Path(
            path
        ).resolve()

        if any(
            self._is_within(
                root,
                resolved,
            )
            for root in self.allowed_roots
        ):
            return resolved

        raise ValueError(
            "Path is outside the active workspace "
            "and temporary agent filesystem: "
            f"{resolved}"
        )

    def require_writable_path(
        self,
        path: str | Path,
    ) -> Path:
        resolved = self.resolve_path(
            path
        )

        if any(
            self._is_within(
                root,
                resolved,
            )
            for root in self.writable_roots
        ):
            return resolved

        raise ValueError(
            f"Path is not writable: {resolved}"
        )

    def display_path(
        self,
        path: str | Path,
    ) -> str:
        resolved = Path(
            path
        ).resolve()

        try:
            relative = resolved.relative_to(
                self.workspace
            )

            return (
                "."
                if not relative.parts
                else str(
                    relative
                )
            )

        except ValueError:
            pass

        aliases = (
            (
                "tmp",
                self.tmp,
            ),
            (
                "home",
                self.home,
            ),
            (
                "cache",
                self.cache,
            ),
            (
                "config",
                self.config,
            ),
            (
                "data",
                self.data,
            ),
            (
                "state",
                self.state,
            ),
            (
                "runtime",
                self.runtime,
            ),
            (
                "agent",
                self.root,
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
                return (
                    f"@{alias}"
                )

            return (
                f"@{alias}/"
                f"{relative.as_posix()}"
            )

        return str(
            resolved
        )

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
            value = os.environ.get(
                name
            )

            if value is not None:
                env[
                    name
                ] = value

        env.update(
            {
                "HOME": str(
                    self.home
                ),

                "TMPDIR": str(
                    self.tmp
                ),
                "TMP": str(
                    self.tmp
                ),
                "TEMP": str(
                    self.tmp
                ),

                "XDG_CACHE_HOME": str(
                    self.cache
                ),
                "XDG_CONFIG_HOME": str(
                    self.config
                ),
                "XDG_DATA_HOME": str(
                    self.data
                ),
                "XDG_STATE_HOME": str(
                    self.state
                ),
                "XDG_RUNTIME_DIR": str(
                    self.runtime
                ),

                "CITRA_WORKSPACE": str(
                    self.workspace
                ),
                "CITRA_AGENT_ROOT": str(
                    self.root
                ),
                "CITRA_TMP": str(
                    self.tmp
                ),
                "CITRA_CACHE": str(
                    self.cache
                ),
            }
        )

        if extra:
            env.update(
                extra
            )

        return env

    def cleanup(
        self,
    ) -> None:
        shutil.rmtree(
            self.root,
            ignore_errors=True,
        )

    def _alias_root(
        self,
        alias: str,
    ) -> Path:
        aliases = {
            "workspace": self.workspace,
            "agent": self.root,
            "home": self.home,
            "tmp": self.tmp,
            "cache": self.cache,
            "config": self.config,
            "data": self.data,
            "state": self.state,
            "runtime": self.runtime,
        }

        try:
            return aliases[
                alias
            ]

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