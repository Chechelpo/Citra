from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

from .skill import Skill

if TYPE_CHECKING:
    from citra.context import ExecutionContext


class CodingConventions(Skill):

    """Represent CodingConventions."""
    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(
            "coding",
            "Describes correct patterns and formatting for any code project",
            Path(),
        )

    @override
    def get_md(
        self,
        context: ExecutionContext,
    ) -> str:
        """Return get md."""
        return ""