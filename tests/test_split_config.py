from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from citra.context import CitraConfig
from citra.context.config import ModelConfigStore
from citra.utils.sandbox import WorkspaceSandbox


def _write_split_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, CitraConfig]:
    source = tmp_path / "source"
    source.mkdir()
    config_dir = tmp_path / ".citra" / "config"
    config_dir.mkdir(parents=True)

    (config_dir / "tools.toml").write_text(
        f'''\
[web-search]
host_url = "http://search.invalid"

[workspace]
temporary_workspace = "{tmp_path / 'agent'}"
permanent_workspace = "{source}"
direct_source = true

[memory]
enabled = false

[sandbox]
global_network_disallow = true

[lsp]
enabled = true
''',
        encoding="utf-8",
    )
    (config_dir / "models.toml").write_text(
        '''\
[models]
active = "alpha"

[models.alpha]
host = "https://alpha.invalid/v1"
api_key = "alpha-secret"
id = "alpha-model"
max_input_tokens = 1024
max_output_tokens = 128

[models.beta]
host = "https://beta.invalid/v1"
api_key = "beta-secret"
id = "beta-model"
max_input_tokens = 2048
max_output_tokens = 256
''',
        encoding="utf-8",
    )
    (config_dir / "linting.toml").write_text(
        '''\
[lint]
timeout = 12
max_output_length = 4096

[[lint.rules]]
name = "ruff"
command = ["ruff", "check", "{path}"]
include = ["**/*.py"]
''',
        encoding="utf-8",
    )

    monkeypatch.setenv("CITRA_CONFIG_PATH", str(config_dir))
    return config_dir, CitraConfig.load()


def test_split_config_loads_all_three_domains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, config = _write_split_config(tmp_path, monkeypatch)

    assert config.web_search.host_url == "http://search.invalid"
    assert config.workspace_context.direct_source is True
    assert config.memory.enabled is False
    assert config.sandbox.global_network_disallow is True
    assert config.lsp.enabled is True
    assert config.lint.enabled is True
    assert config.lint.rules[0].name == "ruff"
    assert config.models() == ("alpha", "beta")
    assert config.model().name == "alpha"
    assert config.model().id == "alpha-model"
    assert config.model("beta").decrypt_api_key() == "beta-secret"
    assert config.model_config_store.config_path == (config_dir / "models.toml").resolve()


def test_model_mutations_only_write_models_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, config = _write_split_config(tmp_path, monkeypatch)
    tools_path = config_dir / "tools.toml"
    linting_path = config_dir / "linting.toml"
    models_path = config_dir / "models.toml"
    tools_before = tools_path.read_bytes()
    linting_before = linting_path.read_bytes()

    config.model_config_store.set_active("beta")
    config.model_config_store.set(name="beta", max_output_tokens=512)

    assert tools_path.read_bytes() == tools_before
    assert linting_path.read_bytes() == linting_before
    raw = tomllib.loads(models_path.read_text(encoding="utf-8"))
    assert raw["models"]["active"] == "beta"
    assert raw["models"]["beta"]["max_output_tokens"] == 512


def test_model_store_accepts_split_config_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, _config = _write_split_config(tmp_path, monkeypatch)
    store = ModelConfigStore(config_dir)
    assert store.get().name == "alpha"
    assert store.get("beta").id == "beta-model"


@pytest.mark.parametrize("filename", ["tools.toml", "models.toml"])
def test_split_config_requires_each_file(
    filename: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, _config = _write_split_config(tmp_path, monkeypatch)
    (config_dir / filename).unlink()

    with pytest.raises(FileNotFoundError, match=filename):
        CitraConfig.load()



def test_split_config_allows_missing_global_linting_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, _config = _write_split_config(tmp_path, monkeypatch)
    (config_dir / "linting.toml").unlink()

    config = CitraConfig.load()

    assert config.lint.enabled is True
    assert config.lint.rules == ()

def test_split_config_rejects_misplaced_model_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, _config = _write_split_config(tmp_path, monkeypatch)
    with (config_dir / "tools.toml").open("a", encoding="utf-8") as file:
        file.write("\n[models]\nactive = 'wrong'\n")

    with pytest.raises(ValueError, match="tools.toml.*models"):
        CitraConfig.load()


def test_split_config_rejects_non_lint_sections_in_linting_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, _config = _write_split_config(tmp_path, monkeypatch)
    with (config_dir / "linting.toml").open("a", encoding="utf-8") as file:
        file.write("\n[tool.ruff]\nline-length = 88\n")

    with pytest.raises(ValueError, match="linting.toml must contain only \\[lint\\]"):
        CitraConfig.load()


def test_sandbox_masks_every_split_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, _config = _write_split_config(tmp_path, monkeypatch)
    monkeypatch.setenv("CITRA_CONFIG_PATH", str(config_dir))

    assert WorkspaceSandbox._citra_private_config_files() == tuple(
        sorted(path.resolve() for path in config_dir.glob("*.toml"))
    )


def test_legacy_single_file_config_remains_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    legacy = tmp_path / "config.toml"
    legacy.write_text(
        f'''\
[model]
host = "https://legacy.invalid/v1"
api_key = "legacy-secret"
id = "legacy-model"
max_tokens = 128

[web-search]
host_url = "http://search.invalid"

[workspace]
permanent_workspace = "{source}"
''',
        encoding="utf-8",
    )
    monkeypatch.setenv("CITRA_CONFIG_PATH", str(legacy))

    config = CitraConfig.load()
    assert config.model().name == "default"
    assert config.model().id == "legacy-model"
    assert config.model().decrypt_api_key() == "legacy-secret"
    assert config.workspace_context.direct_source is False
    assert config.memory.enabled is True


@pytest.mark.parametrize(
    ("old", "new", "match"),
    (
        ("direct_source = true", 'direct_source = "yes"', "workspace.direct_source"),
        ("enabled = false", 'enabled = "no"', "memory.enabled"),
        (
            "global_network_disallow = true",
            'global_network_disallow = "yes"',
            "sandbox.global_network_disallow",
        ),
    ),
)
def test_system_toggles_require_booleans(
    old: str,
    new: str,
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, _config = _write_split_config(tmp_path, monkeypatch)
    tools_path = config_dir / "tools.toml"
    tools_path.write_text(
        tools_path.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match):
        CitraConfig.load()
