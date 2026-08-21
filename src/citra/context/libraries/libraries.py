from pathlib import Path
from dataclasses import dataclass
from .repository_library import RepositoryLibrary

class Libraries():
    __repositories:RepositoryLibrary = RepositoryLibrary()

    def __init__(
        self,
        config: Path
    ):
        self.__repositories = RepositoryLibrary(config)

    @property
    def repositories(self) -> RepositoryLibrary:
        return self.__repositories 