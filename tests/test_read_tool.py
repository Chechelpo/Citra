"""Contract tests for literal single and batch reads."""

from pathlib import Path

import pytest

from citra.sandbox.filesystem_ops.read import ReadInput, execute
from citra.sandbox.filesystem_ops.scope import ScopedFilesystem
from citra.tools.explorer import Read


def test_model_schema_only_exposes_path_and_paths() -> None:
    properties = Read.DEFINITION.function.parameters.properties
    assert tuple(property.name for property in properties) == ("path", "paths")


def test_read_input_accepts_one_path_or_paths() -> None:
    single = ReadInput.parse({"path": "README.md"})
    batch = ReadInput.parse({"paths": ["README.md", "pyproject.toml"]})

    assert single.paths == ("README.md",)
    assert single.to_arguments() == {"path": "README.md"}
    assert batch.paths == ("README.md", "pyproject.toml")
    assert batch.to_arguments() == {
        "paths": ["README.md", "pyproject.toml"],
    }


@pytest.mark.parametrize(
    "arguments",
    (
        {},
        {"path": "a.py", "paths": ["b.py"]},
        {"path": "*.py"},
        {"paths": ["src/**/*.py"]},
        {"paths": []},
        {"requests": [{"path": "a.py"}]},
        {"path": "a.py", "offset": 1},
    ),
)
def test_read_input_rejects_ambiguous_glob_and_legacy_shapes(arguments) -> None:
    with pytest.raises(ValueError):
        ReadInput.parse(arguments)


def test_read_executes_literal_batch(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("alpha\n", encoding="utf-8")
    (workspace / "b.txt").write_text("beta\n", encoding="utf-8")

    environment = {
        "CITRA_PROJECT_ROOT": str(workspace),
        "HOME": str(tmp_path / "home"),
        "CITRA_TMP": str(tmp_path / "tmp"),
        "CITRA_CACHE": str(tmp_path / "cache"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
        "CITRA_LIBRARY": str(tmp_path / "library"),
    }
    for value in environment.values():
        Path(value).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CITRA_PROJECT_ROOT", environment["CITRA_PROJECT_ROOT"])
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    result = execute(
        ReadInput.parse({"paths": ["a.txt", "b.txt"]}),
        ScopedFilesystem(),
    )

    assert result.render() == (
        "===== a.txt =====\nalpha\n\n\n===== b.txt =====\nbeta\n"
    )
