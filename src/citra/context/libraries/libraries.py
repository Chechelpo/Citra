from pathlib import Path
import os

from .repository_library import RepositoryLibrary


class Libraries:
    """
    Container for Citra's persistent libraries.

    All persistent library paths are derived from ``CITRA_ROOT``.
    They never depend on the process working directory because Citra
    intentionally preserves the caller's directory as the active
    workspace.

    Expected layout:

        <CITRA_ROOT>/
        └── libraries/
            └── repos/
    """

    __repositories: RepositoryLibrary

    def __init__(
        self,
        root: str | Path | None = None,
    ):
        if root is None:
            configured_root = os.environ.get("CITRA_ROOT")
            if configured_root is None:
                config_path = os.environ.get("CITRA_CONFIG_PATH")
                if config_path is None:
                    raise RuntimeError(
                        "CITRA_ROOT or CITRA_CONFIG_PATH is required for libraries."
                    )
                root = Path(config_path).resolve().parent
            else:
                root = configured_root
        libraries_path = (
            Path(root)
            .resolve()
            / "libraries"
        )

        self.__repositories = RepositoryLibrary(
            root=(
                libraries_path
                / "repos"
            )
        )

    @property
    def repositories(
        self,
    ) -> RepositoryLibrary:
        return self.__repositories
