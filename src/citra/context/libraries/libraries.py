from citra.context import WorkspaceContext
from dataclasses import dataclass
from .repository_library import RepositoryLibrary

class Libraries():
    __repositories:RepositoryLibrary = RepositoryLibrary()

    def __init__(
        self,
        workspace:WorkspaceContext
    ):
        self.__repositories = RepositoryLibrary(workspace.config)

    @property
    def repositories(self) -> RepositoryLibrary:
        return self.__repositories 