from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import FilesystemInput, FilesystemOutput, require_payload_dict, require_string
from .scope import ScopedFilesystem

@dataclass(frozen=True, slots=True)
class EditOutput(FilesystemOutput):
    """Represent EditOutput."""
    status: str

    @classmethod
    def from_payload(cls, payload: Any) -> "EditOutput":
        """Create an instance from payload."""
        raw = require_payload_dict(payload)
        status = raw.get("status")
        if not isinstance(status, str):
            raise ValueError("Edit output 'status' must be a string.")
        return cls(status=status)

    def to_payload(self) -> dict[str, Any]:
        """Convert the value to payload."""
        return {"status": self.status}

    def render(self) -> str:
        """Handle render."""
        return self.status


@dataclass(frozen=True, slots=True)
class EditInput(FilesystemInput[EditOutput]):
    """Represent EditInput."""
    operation = "edit"
    output_type = EditOutput

    path: str
    new: str
    old: str | None = None
    line: int | None = None
    all: bool = False

    def __post_init__(self) -> None:
        """Validate and initialize the instance after construction."""
        if self.old is not None and self.line is not None:
            raise ValueError("Use either 'old' or 'line' for replacement, not both.")
        if self.line is not None:
            if not isinstance(self.line, int) or self.line < 1:
                raise ValueError("'line' must be a positive 1-based line number.")
            return
        if self.old is None:
            raise ValueError("'old' or 'line' is required for replacement.")
        if not self.old:
            raise ValueError("'old' cannot be empty; use 'line' to replace a whole line.")

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> "EditInput":
        """Handle parse."""
        path = require_string(arguments, "path")
        new = require_string(arguments, "new")
        old = arguments.get("old")
        line = arguments.get("line")

        # Some tool-call providers materialize optional string properties as
        # empty strings. In line mode that means "not selected", not a second
        # replacement target.
        if line is not None and old == "":
            old = None

        if old is not None and line is not None:
            raise ValueError("Use either 'old' or 'line' for replacement, not both.")
        if line is not None:
            return cls(path=path, line=line, new=new)
        if old is None:
            raise ValueError("'old' or 'line' is required for replacement.")
        if not isinstance(old, str):
            raise ValueError("'old' must be a string.")
        return cls(
            path=path,
            old=old,
            new=new,
            all=bool(arguments.get("all", False)),
        )

    def to_arguments(self) -> dict[str, Any]:
        """Convert the value to arguments."""
        result: dict[str, Any] = {"path": self.path, "new": self.new}
        if self.line is not None:
            result["line"] = self.line
        else:
            result["old"] = self.old
            if self.all:
                result["all"] = True
        return result


def execute(order: EditInput, fs: ScopedFilesystem) -> EditOutput:
    """Execute the execute operation."""
    path = fs.require_writable_path(order.path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {fs.display_path(path)}")

    with path.open("r", encoding="utf-8", newline="") as stream:
        text = stream.read()

    if order.line is not None:
        lines = text.splitlines(keepends=True)

        if order.line > len(lines):
            raise ValueError(
                f"Replacement line must be between 1 and {len(lines)}, got {order.line}."
            )

        original = lines[order.line - 1]
        if original.endswith("\r\n"):
            line_ending = "\r\n"
        elif original.endswith(("\n", "\r")):
            line_ending = original[-1]
        else:
            line_ending = ""

        replacement = order.new
        if replacement and line_ending and not replacement.endswith(("\n", "\r")):
            replacement += line_ending

        lines[order.line - 1] = replacement
        fs.write_text_atomic(path, "".join(lines))

        return EditOutput(status="ok")

    assert order.old is not None

    count = text.count(order.old)

    if count == 0:
        return EditOutput(status="error: old_string not found")

    if count > 1 and not order.all:
        return EditOutput(
            status=(
                f"error: old_string appears {count} times, "
                "must be unique (use all=true)"
            )
        )

    replacement = text.replace(
        order.old,
        order.new,
        -1 if order.all else 1,
    )

    fs.write_text_atomic(path, replacement)

    return EditOutput(status="ok")
