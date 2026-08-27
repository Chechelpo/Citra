"""Source positions and ranges with LSP conversions.

Internally the LSP subsystem stores positions as **0-based** line and
character offsets, matching the LSP wire protocol.  Public constructors
and helpers also accept **0-based** values, and conversion helpers are
provided so callers can translate between 0-based LSP positions and
1-based human-readable positions.

Standalone functions are provided for converting between LSP wire dicts
and :class:`SourcePosition` / :class:`SourceRange` objects, as well as
for moving between 0-based and 1-based coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class SourcePosition:
    """A 0-based position inside a text document.

    Attributes:
        line:      0-based line number.
        character: 0-based character offset within the line.
    """

    line: int
    character: int

    def __str__(self) -> str:
        return f"{self.line + 1}:{self.character + 1}"

    # ------------------------------------------------------------------
    # 1-based helpers
    # ------------------------------------------------------------------

    @property
    def line_1(self) -> int:
        """1-based line number (suitable for display)."""
        return self.line + 1

    @property
    def character_1(self) -> int:
        """1-based character offset (suitable for display)."""
        return self.character + 1

    @classmethod
    def from_1based(cls, line: int, character: int) -> SourcePosition:
        """Create a position from 1-based values."""
        return cls(line=line - 1, character=character - 1)


@dataclass(frozen=True)
class SourceRange:
    """A half-open range ``[start, end)`` inside a document."""

    start: SourcePosition
    end: SourcePosition

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"

    @property
    def is_single_line(self) -> bool:
        return self.start.line == self.end.line

    @property
    def is_empty(self) -> bool:
        return self.start == self.end

    def contains(self, position: SourcePosition) -> bool:
        if position < self.start:
            return False
        if position >= self.end:
            return False
        return True


# ---------------------------------------------------------------------------
# Standalone conversion functions
# ---------------------------------------------------------------------------


def position_to_lsp(position: SourcePosition) -> dict[str, int]:
    """Convert a :class:`SourcePosition` to an LSP wire dict."""
    return {"line": position.line, "character": position.character}


def position_from_lsp(data: dict[str, int]) -> SourcePosition:
    """Convert an LSP wire position dict to a :class:`SourcePosition`."""
    return SourcePosition(
        line=int(data.get("line", 0)),
        character=int(data.get("character", 0)),
    )


def range_to_lsp(range_: SourceRange) -> dict[str, dict[str, int]]:
    """Convert a :class:`SourceRange` to an LSP wire dict."""
    return {
        "start": position_to_lsp(range_.start),
        "end": position_to_lsp(range_.end),
    }


def range_from_lsp(data: dict[str, dict[str, int]]) -> SourceRange:
    """Convert an LSP wire range dict to a :class:`SourceRange`."""
    return SourceRange(
        start=position_from_lsp(data["start"]),
        end=position_from_lsp(data["end"]),
    )


def lsp_position_to_range(position: dict[str, int]) -> SourceRange:
    """Convert a single LSP position into a zero-width :class:`SourceRange`."""
    pos = position_from_lsp(position)
    return SourceRange(start=pos, end=pos)


def position_to_1based(position: SourcePosition) -> tuple[int, int]:
    """Return ``(line_1, character_1)`` for *position*."""
    return position.line_1, position.character_1


def position_from_1based(line: int, character: int) -> SourcePosition:
    """Create a :class:`SourcePosition` from 1-based values."""
    return SourcePosition.from_1based(line, character)
