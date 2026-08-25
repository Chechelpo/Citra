from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import urllib.error

from cryptography.fernet import Fernet
import pytest
import tomllib

from citra.commands.model import ModelCommand
from citra.context import CitraConfig, ModelConfig
from citra.context.config import ModelConfigStore, RetryConfig
from citra.utils.chat_completions_api import call_api


def _install_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fernet:
    xdg = tmp_path / "xdg"
    key_dir = xdg / "citra"
    key_dir.mkdir(parents=True)
    key = Fernet.generate_key()
    (key_dir / "encryption.key").write_bytes(key + b"\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    return Fernet(key)


def _write_canonical_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, CitraConfig]:
    fernet = _install_key(tmp_path, monkeypatch)
    config_path = tmp_path / "config.toml"
    alpha_key = fernet.encrypt(b"alpha-secret").decode("ascii")
    beta_key = fernet.encrypt(b"beta-secret").decode("ascii")
    config_path.write_text(
        f'''\
[models]
active = "alpha"

[models.alpha]
host = "https://alpha.invalid/v1"
encrypted_key = "{alpha_key}"
id = "alpha-model"
max_input_tokens = 1000
max_output_tokens = 100
reasoning_effort = "high"

[models.alpha.retry]
max_attempts = 3
request_timeout = 11.0
initial_backoff = 1.0
max_backoff = 4.0

[models.beta]
host = "https://beta.invalid/v1"
encrypted_key = "{beta_key}"
id = "beta-model"
max_input_tokens = 2000
max_output_tokens = 200

[web-search]
host_url = "http://search.invalid"

[workspace]
temporary_workspace = "{tmp_path / 'agent'}"
permanent_workspace = "{tmp_path / 'source'}"
''',
        encoding="utf-8",
    )
    monkeypatch.setenv("CITRA_CONFIG_PATH", str(config_path))
    return config_path, CitraConfig.load()


def test_resolves_active_and_named_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config = _write_canonical_config(tmp_path, monkeypatch)

    assert config.models() == ("alpha", "beta")
    assert config.model().name == "alpha"
    assert config.model().id == "alpha-model"
    assert config.model("beta").name == "beta"
    assert config.model("beta").id == "beta-model"
    assert config.model("alpha").decrypt_api_key() == "alpha-secret"

    with pytest.raises(KeyError, match="Unknown model profile"):
        config.model("missing")


def test_switch_persists_and_explicit_mutation_is_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, config = _write_canonical_config(tmp_path, monkeypatch)
    store = config.model_config_store

    store.set_active("beta")
    store.set(name="alpha", id="alpha-model-v2", max_output_tokens=150)
    store.set_retry(name="alpha", request_timeout=22.0)

    reloaded = CitraConfig.load()
    assert reloaded.model().name == "beta"
    assert reloaded.model().id == "beta-model"
    assert reloaded.model("alpha").id == "alpha-model-v2"
    assert reloaded.model("alpha").max_output_tokens == 150
    assert reloaded.model("alpha").retry.request_timeout == 22.0
    assert reloaded.model("beta").max_output_tokens == 200

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert raw["models"]["active"] == "beta"


def test_add_delete_and_active_invariants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config = _write_canonical_config(tmp_path, monkeypatch)
    store = config.model_config_store

    store.add("gamma", copy_from="beta")
    assert store.names() == ("alpha", "beta", "gamma")
    assert store.get("gamma").id == "beta-model"
    assert store.get("gamma").decrypt_api_key() == "beta-secret"

    store.set_active("gamma")
    with pytest.raises(ValueError, match="Cannot delete the active"):
        store.delete("gamma")

    store.delete("beta")
    assert store.names() == ("alpha", "gamma")
    with pytest.raises(KeyError, match="Unknown model profile"):
        store.set_active("beta")


def test_legacy_model_is_implicit_default_and_migrates_on_add(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_key(tmp_path, monkeypatch)
    config_path = tmp_path / "legacy.toml"
    config_path.write_text(
        '''\
[model]
host = "https://legacy.invalid/v1"
api_key = "legacy-secret"
id = "legacy-model"
max_tokens = 128

[web-search]
host_url = "http://search.invalid"

[workspace]
temporary_workspace = "/tmp/citra-test-agent"
permanent_workspace = "/tmp/citra-test-source"
''',
        encoding="utf-8",
    )
    monkeypatch.setenv("CITRA_CONFIG_PATH", str(config_path))

    config = CitraConfig.load()
    model = config.model()
    assert model.name == "default"
    assert model.max_input_tokens == 128
    assert model.max_output_tokens == 128
    assert model.decrypt_api_key() == "legacy-secret"

    config.model_config_store.add("second")
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert "model" not in raw
    assert raw["models"]["active"] == "default"
    assert raw["models"]["default"]["id"] == "legacy-model"
    assert raw["models"]["second"]["id"] == "legacy-model"
    assert "api_key" not in raw["models"]["default"]
    assert "encrypted_key" in raw["models"]["default"]
    assert "max_tokens" not in raw["models"]["default"]
    assert raw["models"]["default"]["max_input_tokens"] == 128
    assert raw["models"]["default"]["max_output_tokens"] == 128
    assert config.model_config_store.get("default").decrypt_api_key() == "legacy-secret"


def test_model_command_switches_and_targets_profiles_without_leaking_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config = _write_canonical_config(tmp_path, monkeypatch)
    command = ModelCommand(SimpleNamespace(config=config))  # type: ignore[arg-type]

    listing = command.run("list").output
    assert "* alpha: alpha-model" in listing
    assert "  beta: beta-model" in listing

    result = command.run("use beta")
    assert "Active model profile = beta" in result.output
    assert config.model().name == "beta"

    result = command.run("set --profile alpha id alpha-command-model")
    assert "models.alpha.id = alpha-command-model" == result.output
    assert config.model("alpha").id == "alpha-command-model"
    assert config.model("beta").id == "beta-model"

    shown = command.run("show alpha").output
    assert "profile: alpha" in shown
    assert "active: no" in shown
    assert "api_key: ********" in shown
    assert "alpha-secret" not in shown

    command.run("add gamma --copy alpha")
    assert "gamma" in config.models()
    delete_active = command.run("delete beta").output
    assert "Cannot delete the active" in delete_active


def test_set_api_key_encrypts_only_target_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, config = _write_canonical_config(tmp_path, monkeypatch)
    store = config.model_config_store

    old_beta_ciphertext = store.get_model_section("beta")["encrypted_key"]
    store.set_api_key("new-alpha-secret", name="alpha")

    assert store.get("alpha").decrypt_api_key() == "new-alpha-secret"
    assert store.get_model_section("beta")["encrypted_key"] == old_beta_ciphertext
    assert "new-alpha-secret" not in config_path.read_text(encoding="utf-8")


def test_invalid_active_profile_fails_at_config_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fernet = _install_key(tmp_path, monkeypatch)
    encrypted = fernet.encrypt(b"secret").decode("ascii")
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        f'''\
[models]
active = "missing"

[models.alpha]
host = "https://alpha.invalid/v1"
encrypted_key = "{encrypted}"
id = "alpha-model"
max_input_tokens = 100
max_output_tokens = 10

[web-search]
host_url = "http://search.invalid"

[workspace]
temporary_workspace = "/tmp/a"
permanent_workspace = "/tmp/b"
''',
        encoding="utf-8",
    )
    monkeypatch.setenv("CITRA_CONFIG_PATH", str(config_path))

    with pytest.raises(RuntimeError, match="Active model profile 'missing'"):
        CitraConfig.load()


def test_call_api_uses_supplied_model_snapshot_for_entire_request() -> None:
    snapshot = ModelConfig(
        name="snapshot",
        host="https://snapshot.invalid/v1",
        encrypted_key="",
        id="snapshot-model",
        max_input_tokens=100,
        max_output_tokens=17,
        reasoning_effort=None,
        retry=RetryConfig(max_attempts=1),
        _plaintext_api_key="snapshot-secret",
    )

    class ConfigThatMustNotBeRead:
        def model(self):
            raise AssertionError("active model was re-resolved")

    context = SimpleNamespace(config=ConfigThatMustNotBeRead())

    class Response:
        status = 200
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"},
                        }
                    ]
                }
            ).encode("utf-8")

    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    with mock.patch(
        "citra.utils.chat_completions_api.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        result = call_api(
            context,  # type: ignore[arg-type]
            [],
            {},
            model_config=snapshot,
            sys_prompt="system",
        )

    request = captured["request"]
    payload = json.loads(request.data.decode("utf-8"))
    assert result["choices"][0]["message"]["content"] == "ok"
    assert request.full_url == "https://snapshot.invalid/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer snapshot-secret"
    assert payload["model"] == "snapshot-model"
    assert payload["max_tokens"] == 17


def test_legacy_encrypted_migration_preserves_ciphertext(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fernet = _install_key(tmp_path, monkeypatch)
    ciphertext = fernet.encrypt(b"keep-me").decode("ascii")
    config_path = tmp_path / "legacy-encrypted.toml"
    config_path.write_text(
        f'''\
[model]
host = "https://legacy.invalid/v1"
encrypted_key = "{ciphertext}"
id = "legacy-model"
max_input_tokens = 256
max_output_tokens = 32
''',
        encoding="utf-8",
    )

    store = ModelConfigStore(config_path)
    store.add("copy")

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert raw["models"]["default"]["encrypted_key"] == ciphertext
    assert raw["models"]["copy"]["encrypted_key"] == ciphertext
    assert store.get("copy").decrypt_api_key() == "keep-me"


def test_invalid_profile_mutations_are_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, config = _write_canonical_config(tmp_path, monkeypatch)
    store = config.model_config_store
    before = config_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="max_output_tokens"):
        store.set(name="alpha", max_output_tokens=0)
    assert config_path.read_text(encoding="utf-8") == before

    with pytest.raises(ValueError, match="cannot exceed"):
        store.set_retry(name="alpha", initial_backoff=9.0, max_backoff=1.0)
    assert config_path.read_text(encoding="utf-8") == before


def test_call_api_resolves_active_model_only_once_across_retries() -> None:
    first = ModelConfig(
        name="first",
        host="https://first.invalid/v1",
        encrypted_key="",
        id="first-model",
        max_input_tokens=100,
        max_output_tokens=11,
        reasoning_effort=None,
        retry=RetryConfig(
            max_attempts=2,
            request_timeout=1.0,
            initial_backoff=0.0,
            max_backoff=0.0,
        ),
        _plaintext_api_key="first-secret",
    )
    second = ModelConfig(
        name="second",
        host="https://second.invalid/v1",
        encrypted_key="",
        id="second-model",
        max_input_tokens=100,
        max_output_tokens=22,
        reasoning_effort=None,
        retry=RetryConfig(max_attempts=1),
        _plaintext_api_key="second-secret",
    )

    class SwitchingConfig:
        def __init__(self) -> None:
            self.calls = 0

        def model(self):
            self.calls += 1
            return first if self.calls == 1 else second

    switching = SwitchingConfig()
    context = SimpleNamespace(config=switching)
    requests = []

    class Response:
        status = 200
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"},
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        if len(requests) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "Service Unavailable",
                {},
                io.BytesIO(b'{"error":{"message":"temporary"}}'),
            )
        return Response()

    with mock.patch(
        "citra.utils.chat_completions_api.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        result = call_api(
            context,  # type: ignore[arg-type]
            [],
            {},
            sys_prompt="system",
        )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert switching.calls == 1
    assert len(requests) == 2
    for request, _timeout in requests:
        payload = json.loads(request.data.decode("utf-8"))
        assert request.full_url == "https://first.invalid/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer first-secret"
        assert payload["model"] == "first-model"
        assert payload["max_tokens"] == 11


def test_model_config_preserves_legacy_positional_constructor() -> None:
    retry = RetryConfig(max_attempts=3)
    config = ModelConfig(
        "https://legacy-constructor.invalid/v1",
        "ciphertext",
        "legacy-constructor-model",
        100,
        10,
        None,
        retry,
    )

    assert config.name == "default"
    assert config.host == "https://legacy-constructor.invalid/v1"
    assert config.retry is retry


def test_invalid_canonical_profile_name_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fernet = _install_key(tmp_path, monkeypatch)
    encrypted = fernet.encrypt(b"secret").decode("ascii")
    config_path = tmp_path / "bad-profile-name.toml"
    config_path.write_text(
        f'''\
[models]
active = "bad.name"

[models."bad.name"]
host = "https://bad.invalid/v1"
encrypted_key = "{encrypted}"
id = "bad-model"
max_input_tokens = 100
max_output_tokens = 10
''',
        encoding="utf-8",
    )

    store = ModelConfigStore(config_path)
    with pytest.raises(ValueError, match="profile names"):
        store.get()
