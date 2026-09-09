from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from citra.commands.model import ModelCommand
from citra.config import CitraConfig, ModelConfigStore


def _config(tmp_path: Path) -> CitraConfig:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "tools.toml").write_text(
        "[web-search]\nhost_url = 'https://search.invalid'\n",
        encoding="utf-8",
    )
    (config_dir / "sandbox.toml").write_text(
        "[sandbox]\nglobal_network_disallow = false\n",
        encoding="utf-8",
    )
    (config_dir / "models.toml").write_text(
        """
[models]
orchestrator = "alpha"
subagent = "beta"

[models.alpha]
host = "https://alpha.invalid"
api_key = "alpha-secret"
id = "alpha-model"
max_input_tokens = 10000
max_output_tokens = 1000

[models.beta]
host = "https://beta.invalid"
api_key = "beta-secret"
id = "beta-model"
max_input_tokens = 8000
max_output_tokens = 800
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return CitraConfig.load(config_dir)


def test_named_orchestrator_and_subagent_profiles(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.model().name == "alpha"
    assert config.model_config_store.subagent_name() == "beta"
    assert config.model("beta").decrypt_api_key() == "beta-secret"


def test_profile_accepts_multiple_plaintext_api_keys(tmp_path: Path) -> None:
    config = _config(tmp_path)
    models_file = config.model_config_store.config_file
    contents = models_file.read_text(encoding="utf-8")
    models_file.write_text(
        contents.replace(
            'api_key = "alpha-secret"',
            'api_keys = ["alpha-secret", "alpha-backup"]',
        ),
        encoding="utf-8",
    )

    model = ModelConfigStore.load(models_file.parent).get("alpha")

    assert model.decrypt_api_key() == "alpha-secret"
    assert model.decrypt_api_keys() == ("alpha-secret", "alpha-backup")


def test_profile_rejects_mixed_single_and_multiple_api_keys(tmp_path: Path) -> None:
    config = _config(tmp_path)
    models_file = config.model_config_store.config_file
    contents = models_file.read_text(encoding="utf-8")
    models_file.write_text(
        contents.replace(
            'api_key = "alpha-secret"',
            'api_key = "alpha-secret"\napi_keys = ["alpha-backup"]',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one"):
        ModelConfigStore.load(models_file.parent)


def test_store_persists_multiple_api_keys_encrypted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    config = _config(tmp_path)

    config.model_config_store.set_api_keys(
        ["alpha-secret", "alpha-backup"],
        name="alpha",
    )

    raw = config.model_config_store.config_file.read_text(encoding="utf-8")
    assert "encrypted_keys = [" in raw
    assert "alpha-secret" not in raw
    reloaded = ModelConfigStore.load(config.model_config_store.config_file.parent)
    assert reloaded.get("alpha").decrypt_api_keys() == (
        "alpha-secret",
        "alpha-backup",
    )


def test_model_store_persists_selectors_and_retry(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = config.model_config_store
    store.set_orchestrator("beta")
    store.set_subagent(None)
    store.set_retry(name="beta", max_attempts=5, max_backoff=8.0)

    reloaded = ModelConfigStore.load(store.config_file.parent)
    assert reloaded.orchestrator_name() == "beta"
    assert reloaded.subagent_name() == "beta"
    assert reloaded.get().retry.max_attempts == 5
    assert reloaded.get().retry.max_backoff == 8.0


def test_model_command_updates_orchestrator_selector(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command = ModelCommand(SimpleNamespace(config=config))
    result = command._run("use beta")

    assert "Orchestrator model profile = beta" in result.output
    assert config.model_config_store.orchestrator_name() == "beta"


def test_model_command_sets_multiple_api_keys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    config = _config(tmp_path)
    command = ModelCommand(SimpleNamespace(config=config))

    result = command.run("set --profile alpha api_keys '[\"first\", \"second\"]'")

    assert "2 model API keys updated for alpha" in result.output
    assert config.model("alpha").decrypt_api_keys() == ("first", "second")


def test_model_command_adds_api_keys_one_at_a_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    config = _config(tmp_path)
    command = ModelCommand(SimpleNamespace(config=config))

    first = command.run("add api_key alpha-backup --profile alpha")
    second = command.run("add api_key alpha-third --profile alpha")

    assert "2 configured" in first.output
    assert "3 configured" in second.output
    assert config.model("alpha").decrypt_api_keys() == (
        "alpha-secret",
        "alpha-backup",
        "alpha-third",
    )
    raw = config.model_config_store.config_file.read_text(encoding="utf-8")
    assert "encrypted_keys = [" in raw
    assert "alpha-backup" not in raw


def test_model_add_accepts_copy_option_in_declared_position(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command = ModelCommand(SimpleNamespace(config=config))

    result = command.run("add --copy beta gamma")

    assert "Added model profile gamma from beta" in result.output
    assert config.model_config_store.get("gamma").id == "beta-model"


def test_model_add_retains_trailing_copy_option(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command = ModelCommand(SimpleNamespace(config=config))

    result = command.run("add gamma --copy beta")

    assert "Added model profile gamma from beta" in result.output
    assert config.model_config_store.get("gamma").id == "beta-model"
