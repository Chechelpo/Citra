from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..session_tool import SessionTool


TExtract = TypeVar("TExtract")


class MemoryTool(SessionTool, ABC, Generic[TExtract]):

    @property
    @abstractmethod
    def heading(self) -> str:
        ...

    @abstractmethod
    def get_extracts(self) -> list[TExtract]:
        ...

    @abstractmethod
    def format_extract(
        self,
        extract: TExtract,
    ) -> str:
        ...

    @abstractmethod
    def should_offer_documentation(self) -> bool:
        """
        Return whether this memory type may contain information worth
        persisting into repository documentation at the end of the work.
        """
        ...

    def format_for_llm(self) -> str:
        extracts = self.get_extracts()

        if not extracts:
            return ""

        return "\n".join(
            [
                f"## {self.heading}",
                *(
                    self.format_extract(extract)
                    for extract in extracts
                ),
            ]
        )
