"""Contract tests for literal single and batch reads."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from citra.sandbox.filesystem_ops.read import (
    ReadEntry,
    ReadInput,
    ReadOutput,
    ReadSymbol,
    execute,
)
from citra.sandbox.filesystem_ops.scope import ScopedFilesystem
from citra.tools.explorer import Read
from citra.utils.repo_map import RepoMap


def test_model_schema_exposes_paths_ranges_and_budget() -> None:
    properties = Read.DEFINITION.function.parameters.properties
    assert tuple(property.name for property in properties) == (
        "path",
        "paths",
        "from_line",
        "to_line",
        "max_tokens",
    )


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


def test_read_output_renders_compact_symbols_before_content() -> None:
    output = ReadOutput(
        entries=(
            ReadEntry(
                path="sample.py",
                content="class Example:\n    pass\n",
            ),
        ),
        single_path=True,
    ).with_symbols(
        {
            "sample.py": (
                ReadSymbol(name="Example", from_line=1, to_line=2),
            ),
        }
    )

    assert output.render() == (
        "symbols: Example:L1-L2\n\n"
        "class Example:\n    pass\n"
    )


def test_repo_map_indexes_only_functions_and_classes(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "class Example:\n"
        "    def method(self):\n"
        "        return 1\n"
        "\n"
        "def helper():\n"
        "    return 2\n"
        "\n"
        "VALUE = 3\n",
        encoding="utf-8",
    )
    workspace = SimpleNamespace(
        resolve_path=lambda path: tmp_path / path,
        display_path=lambda path: Path(path).relative_to(tmp_path).as_posix(),
    )

    definitions = RepoMap(cast(Any, workspace)).definitions_for_paths(
        ("sample.py",)
    )

    assert [
        (definition.name, definition.from_line, definition.to_line)
        for definition in definitions["sample.py"]
    ] == [
        ("Example", 0, 2),
        ("method", 1, 2),
        ("helper", 4, 5),
    ]


def test_read_tool_adds_repo_map_symbols_before_budgeting() -> None:
    output = ReadOutput(
        entries=(ReadEntry(path="sample.py", content="content\n"),),
        single_path=True,
    )
    filesystem = SimpleNamespace(execute=lambda request: output)
    repo_map = SimpleNamespace(
        definitions_for_paths=lambda paths: {
            "sample.py": (
                SimpleNamespace(
                    name="Example",
                    from_line=2,
                    to_line=8,
                ),
            ),
        }
    )
    context = SimpleNamespace(
        filesystem=filesystem,
        repo_map=repo_map,
        model_config=lambda: SimpleNamespace(id="claude-sonnet"),
    )

    result = Read(cast(Any, context)).execute({"path": "sample.py"})

    assert result == "symbols: Example:L3-L9\n\ncontent\n"
