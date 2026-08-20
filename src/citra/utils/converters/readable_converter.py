from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import os
from pathlib import Path
import tempfile

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
    ) -> Path:
        path = path.resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        output = self._output_path(
            path
        )

        if output.is_file():
            return output

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = output.with_suffix(
            ".tmp"
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
            self._cache_directory()
            / f"{digest}.txt"
        )

    @staticmethod
    def _cache_directory() -> Path:
        configured = os.environ.get(
            "CITRA_CACHE"
        )

        if configured:
            root = Path(
                configured
            )
        else:
            root = (
                Path(
                    tempfile.gettempdir()
                )
                / "citra"
                / "cache"
            )

        return (
            root
            / "readable"
        )