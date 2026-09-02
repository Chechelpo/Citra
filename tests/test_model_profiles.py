from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
