from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citra.context import ExecutionContext
    from citra.agent import AgentSession

@dataclass(frozen=True, slots=True)
class Skill:
    """
    A skill containing conditionally loaded instructions.

    Directory structure:

        skills/<skill-name>/
        ├── skill.toml
        └── SKILL.md
    """

    name: str
    description: str
    root: Path | None = field(
        default=None
    )

    def get_md(
        self,
        context: ExecutionContext
    ) -> str:
        """Return the complete skill instructions."""
        if self.root is None:
            raise Exception(f"Skill {self.name} does not override get_md() and also doesn't declare a path yet get_md() was called")
        return (self.root / "SKILL.md").read_text(encoding="utf-8")