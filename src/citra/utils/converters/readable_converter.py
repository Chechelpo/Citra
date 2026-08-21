from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from pathlib import Path
import tempfile

from citra.context import WorkspaceContext


class ReadableConverter(ABC):
    """
    Converts non-plain-text files into cached UTF-8 text files suitable
    for LLM consumption.
    """

    VERSION = 1

    @property
    @abstractmethod
    def extensions(
        self,
    ) -> frozenset[str]:
        pass

    def supports(
        self,
        path: Path,
    ) -> bool:
        return (
            path.is_file()
            and path.suffix.lower()
            in self.extensions
        )

    def convert(
        self,
        path: Path,
        *,
        workspace: WorkspaceContext,
    ) -> Path:
        path = workspace.require_allowed_path(
            path
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"File not found: "
                f"{workspace.display_path(path)}"
            )

        output = self._output_path(
            path,
            workspace=workspace,
        )

        output = workspace.require_writable_path(
            output
        )

        if output.is_file():
            return output

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = self._temporary_path(
            output
        )

        temporary = workspace.require_writable_path(
            temporary
        )

        try:
            self._convert(
                source=path,
                destination=temporary,
            )

            temporary.replace(
                output
            )

        finally:
            temporary.unlink(
                missing_ok=True
            )

        return output

    @abstractmethod
    def _convert(
        self,
        *,
        source: Path,
        destination: Path,
    ) -> None:
        pass

    def _output_path(
        self,
        path: Path,
        *,
        workspace: WorkspaceContext,
    ) -> Path:
        stat = path.stat()

        identity = (
            f"{path}\0"
            f"{stat.st_mtime_ns}\0"
            f"{stat.st_size}\0"
            f"{type(self).__name__}\0"
            f"{self.VERSION}"
        )

        digest = hashlib.sha256(
            identity.encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            workspace.cache
            / "readable"
            / f"{digest}.txt"
        )

    @staticmethod
    def _temporary_path(
        output: Path,
    ) -> Path:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{output.stem}-",
            suffix=".tmp",
            dir=output.parent,
        )

        try:
            return Path(
                raw_path
            )
        finally:
            os.close(
                descriptor
            )