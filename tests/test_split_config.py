from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from citra.config import CitraConfig, ModelConfigStore


def _write_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, CitraConfig]:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    key = Fernet.generate_key()
    encrypted = Fernet(key).encrypt(b"secret").decode("ascii")
    monkeypatch.setenv("CITRA_ENCRYPTION_KEY", key.decode("ascii"))
    monkeypatch.setenv("CITRA_CONFIG_PATH", str(config_dir))

    (config_dir / "tools.toml").write_text(
        """
[web-search]
host_url = "https://search.invalid"

[bash]
permission_timeout = 17

[subprocess]
max_output_length = 4096

[browser]
request_timeout = 12.5

[lsp]
enabled = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "models.toml").write_text(
        f"""
[models]
orchestrator = "alpha"
subagent = "beta"

[models.alpha]
host = "https://alpha.invalid"
encrypted_key = "{encrypted}"
id = "alpha-model"
max_input_tokens = 10000
max_output_tokens = 1000
reasoning_effort = "high"

[models.beta]
host = "https://beta.invalid"
encrypted_key = "{encrypted}"
id = "beta-model"
max_input_tokens = 8000
max_output_tokens = 800
reasoning_effort = "medium"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "sandbox.toml").write_text(
        "[sandbox]\nglobal_network_disallow = true\n",
        encoding="utf-8",
    )
    return config_dir, CitraConfig.load()


def test_canonical_config_aggregates_each_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, config = _write_config(tmp_path, monkeypatch)

    assert config.model().name == "alpha"
    assert config.model_config_store.subagent_name() == "beta"
    assert config.models() == ("alpha", "beta")
    assert config.bash.permission_timeout == 17
    assert config.subprocess.max_output_length == 4096
    assert config.browser.request_timeout == 12.5
    assert config.lsp.enabled is False
    assert config.sandbox_policy.global_disallow_network is True
    assert config.model_config_store.config_path == config_dir / "models.toml"


@pytest.mark.parametrize("filename", ["tools.toml", "models.toml", "sandbox.toml"])
def test_required_config_files(
    filename: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, _ = _write_config(tmp_path, monkeypatch)
    (config_dir / filename).unlink()
    with pytest.raises(FileNotFoundError):
        CitraConfig.load()


def test_curl_section_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, _ = _write_config(tmp_path, monkeypatch)
    with (config_dir / "tools.toml").open("a", encoding="utf-8") as file:
        file.write("\n[curl]\npermission_timeout = 10\n")
    with pytest.raises(ValueError, match=r"\[curl\].*removed"):
        CitraConfig.load()


def test_model_mutation_only_rewrites_models_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, config = _write_config(tmp_path, monkeypatch)
    tools_before = (config_dir / "tools.toml").read_bytes()
    sandbox_before = (config_dir / "sandbox.toml").read_bytes()

    config.model_config_store.set_orchestrator("beta")
    config.model_config_store.set(name="beta", max_output_tokens=1200)

    raw = tomllib.loads((config_dir / "models.toml").read_text(encoding="utf-8"))
    assert raw["models"]["orchestrator"] == "beta"
    assert raw["models"]["beta"]["max_output_tokens"] == 1200
    assert (config_dir / "tools.toml").read_bytes() == tools_before
    assert (config_dir / "sandbox.toml").read_bytes() == sandbox_before


def test_model_store_accepts_config_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, _ = _write_config(tmp_path, monkeypatch)
    assert ModelConfigStore(config_dir).get().name == "alpha"
