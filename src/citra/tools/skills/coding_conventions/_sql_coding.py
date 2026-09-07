from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

from ..skill import Skill

if TYPE_CHECKING:
    from citra.context import ExecutionContext


class SQLCodingSkill(Skill):
    """Represent SQL coding conventions."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(
            "sql-coding-conventions",
            "Describes correct patterns and formatting for sql code",
            Path(),
        )

    @override
    def get_md(
        self,
        context: ExecutionContext,
    ) -> str:
        """Return get md."""
        return _SKILL

_SKILL = """
# SQL conventions

## On JSON columns

Avoid storing relational data as JSON.

Use normalized tables when data:
- has its own lifecycle
- needs indexes
- is queried independently
- participates in relationships
- requires constraints

Use JSON only for:
- external payload compatibility
- flexible metadata
- rarely queried attributes
- AI-generated auxiliary data

JSON should be a deliberate escape hatch, not the default schema design.

## On IDs

Prefer stable identifiers over user-facing names.

Names change; references should not.

## On foreign keys

Use foreign keys for relationships.

Do not rely on application code alone to maintain referential integrity.

## On timestamps

Store creation and update timestamps consistently.

Prefer database-managed defaults where possible.

## On migrations

Migrations should be:
- deterministic
- forward-only
- idempotent when possible
- small and focused

Avoid rewriting existing data implicitly in schema migrations.
"""