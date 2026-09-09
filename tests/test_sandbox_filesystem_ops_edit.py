"""End-to-end tests for the sandboxed edit filesystem operation."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from citra.sandbox.filesystem_ops.edit import EditInput, execute
from citra.sandbox.filesystem_ops.scope import ScopedFilesystem


def _make_filesystem(root: Path) -> ScopedFilesystem:
    paths = {
        "HOME": root / "home",
        "CITRA_TMP": root / "tmp",
        "CITRA_CACHE": root / "cache",
        "CITRA_ENV": root / "env",
        "CITRA_RUNTIME": root / "runtime",
        "XDG_CONFIG_HOME": root / "config",
        "XDG_DATA_HOME": root / "data",
        "XDG_RUNTIME_DIR": root / "run",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    environment = {
        "CITRA_PROJECT_ROOT": str(root / "project"),
        **{name: str(path) for name, path in paths.items()},
    }
    (root / "project").mkdir(parents=True, exist_ok=True)
    with mock.patch.dict(os.environ, environment, clear=True):
        return ScopedFilesystem()


def test_line_replaces_content_and_preserves_line_ending(tmp_path: Path) -> None:
    filesystem = _make_filesystem(tmp_path)
    path = filesystem.workspace / "example.txt"
    path.write_bytes(b"first\r\nsecond\r\nthird\r\n")

    result = execute(
        EditInput.parse({"path": "example.txt", "line": 2, "new": "changed"}),
        filesystem,
    )

    assert result.status == "ok"
    assert path.read_bytes() == b"first\r\nchanged\r\nthird\r\n"


def test_empty_line_replacement_deletes_the_line(tmp_path: Path) -> None:
    filesystem = _make_filesystem(tmp_path)
    path = filesystem.workspace / "example.txt"
    path.write_text("first\nsecond\nthird\n", encoding="utf-8")

    execute(
        EditInput.parse({"path": "example.txt", "line": 2, "new": ""}),
        filesystem,
    )

    assert path.read_text(encoding="utf-8") == "first\nthird\n"


def test_line_must_refer_to_an_existing_line(tmp_path: Path) -> None:
    filesystem = _make_filesystem(tmp_path)
    path = filesystem.workspace / "example.txt"
    path.write_text("only\n", encoding="utf-8")

    with pytest.raises(ValueError, match="between 1 and 1"):
        execute(
            EditInput.parse({"path": "example.txt", "line": 2, "new": "extra"}),
            filesystem,
        )


def test_empty_old_points_callers_to_line_replacement() -> None:
    with pytest.raises(ValueError, match="use 'line' to replace a whole line"):
        EditInput.parse({"path": "example.txt", "old": "", "new": "changed"})


def test_line_replacement_tolerates_empty_optional_old() -> None:
    parsed = EditInput.parse(
        {"path": "example.txt", "line": 2, "old": "", "new": "changed"}
    )

    assert parsed.line == 2
    assert parsed.old is None
