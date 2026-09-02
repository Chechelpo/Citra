from __future__ import annotations

import os
from pathlib import Path
import tempfile
import re
from typing import Any, cast


_ALIAS = re.compile(r"^@([a-z_]+)(?:/(.*))?$")


class ScopedFilesystem:
    """Path resolver whose authority is limited to sandbox data mounts."""

    def __init__(self) -> None:
        project_raw = os.environ.get("CITRA_PROJECT_ROOT")
        self.workspace = (
            Path(project_raw).resolve()
            if project_raw
            else Path.cwd().resolve()
        )
        self.home = self._required_path("HOME")
        self.tmp = self._required_path("CITRA_TMP")
        self.cache = self._required_path("CITRA_CACHE")

        env_raw = os.environ.get("CITRA_ENV")
        self.env = Path(env_raw).resolve() if env_raw else self.workspace.parent / "env"

        self.config = self._required_path("XDG_CONFIG_HOME")
        self.data = self._required_path("XDG_DATA_HOME")

        runtime_raw = os.environ.get("CITRA_RUNTIME")
        self.runtime = (
            Path(runtime_raw).resolve()
            if runtime_raw
            else self._required_path("XDG_RUNTIME_DIR")
        )

        library_raw = os.environ.get("CITRA_LIBRARY")
        self.library = (
            Path(library_raw).resolve()
            if library_raw
            else self.workspace.parent / "library"
        )

        self._denied_roots = (self.library,)
        self._aliases: dict[str, Path] = {
            "home": self.home,
            "tmp": self.tmp,
            "cache": self.cache,
            "env": self.env,
            "config": self.config,
            "data": self.data,
            "runtime": self.runtime,
        }
        self._read_roots = (self.workspace, *self._aliases.values())
        self._write_roots = tuple(
            root
            for name, root in self._aliases.items()
            if name != "runtime"
        )
        self._write_roots = (self.workspace, *self._write_roots)

    @staticmethod
    def _required_path(name: str) -> Path:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Sandbox environment is missing {name}.")
        return Path(value).resolve()

    @staticmethod
    def _within(root: Path, path: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def resolve_path(self, value: str | Path) -> Path:
        raw = str(value)
        alias_raw = raw
        while alias_raw.startswith("./"):
            alias_raw = alias_raw[2:]

        match = _ALIAS.fullmatch(alias_raw)
        if match:
            alias_name, remainder = match.groups()
            try:
                base = self._aliases[alias_name]
            except KeyError as error:
                raise ValueError(
                    f"Unknown workspace path alias: @{alias_name}"
                ) from error
            candidate = base if not remainder else base / remainder
        elif raw == "~" or raw.startswith("~/"):
            remainder = "" if raw == "~" else raw[2:]
            candidate = self.home if not remainder else self.home / remainder
        else:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = self.workspace / candidate

        return candidate.resolve()

    def require_allowed_path(self, value: str | Path) -> Path:
        resolved = Path(value).resolve()
        if any(self._within(root, resolved) for root in self._denied_roots):
            raise ValueError(
                "The Citra document library is accessible only through the Document tool."
            )
        if any(self._within(root, resolved) for root in self._read_roots):
            return resolved
        raise ValueError(f"Path is outside the model-facing filesystem: {resolved}")

    def require_writable_path(self, value: str | Path) -> Path:
        resolved = self.resolve_path(value)
        if any(self._within(root, resolved) for root in self._write_roots):
            return resolved
        raise ValueError(f"Path is read-only: {self.display_path(resolved)}")

    def display_path(self, value: str | Path) -> str:
        resolved = Path(value).resolve()
        ordered = (
            ("", self.workspace),
            ("tmp", self.tmp),
            ("home", self.home),
            ("cache", self.cache),
            ("env", self.env),
            ("config", self.config),
            ("data", self.data),
            ("runtime", self.runtime),
        )
        for alias, root in ordered:
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            if not alias:
                return "." if not relative.parts else relative.as_posix()
            return f"@{alias}" if not relative.parts else f"@{alias}/{relative.as_posix()}"
        return str(resolved)

    def write_text_atomic(
        self,
        value: str | Path,
        text: str,
        *,
        encoding: str = "utf-8",
    ) -> Path:
        destination = self.require_writable_path(value)
        parent = self.require_writable_path(destination.parent)
        parent.mkdir(parents=True, exist_ok=True)

        descriptor, raw = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary = Path(raw)
        try:
            with os.fdopen(descriptor, "w", encoding=encoding) as stream:
                stream.write(text)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def convert_readable(self, path: Path) -> Path:
        if path.suffix.lower() not in {".pdf", ".ipynb"}:
            return path

        from citra.utils.converters import convert

        return self.require_allowed_path(
            convert(path, workspace=cast(Any, self))
        )
