from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any

import tomllib

from cryptography.fernet import Fernet, InvalidToken

from ._constants import MODELS_CONFIG_FILE


_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_LEGACY_PROFILE_NAME = "default"


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 12
    request_timeout: float = 120.0
    initial_backoff: float = 1.0
    max_backoff: float = 30.0


@dataclass(frozen=True)
class ModelConfig:
    host: str
    encrypted_key: str
    id: str
    max_input_tokens: int
    max_output_tokens: int
    reasoning_effort: str | None
    retry: RetryConfig = RetryConfig()
    name: str = _LEGACY_PROFILE_NAME
    _plaintext_api_key: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def decrypt_api_key(self) -> str:
        if self._plaintext_api_key is not None:
            return self._plaintext_api_key

        return ModelConfigStore.decrypt_secret(
            self.encrypted_key,
        )


class ModelConfigStore:
    """Persistent model configuration repository."""

    def __init__(
        self,
        config_path: str | Path,
    ) -> None:
        path = Path(config_path).expanduser().resolve()
        self.config_file = (
            path / MODELS_CONFIG_FILE
            if path.is_dir() or path.suffix.lower() != ".toml"
            else path
        )
        self.config_path = self.config_file

    @classmethod
    def load(
        cls,
        config_path: str | Path,
    ) -> ModelConfigStore:
        config_dir = Path(config_path).expanduser().resolve()
        config_file = config_dir / MODELS_CONFIG_FILE

        if not config_file.is_file():
            raise FileNotFoundError(
                f"Model config file not found: {config_file}"
            )

        store = cls(config_file)
        store._layout(store._load_document())
        store.get()
        return store

    def _load_document(
        self,
    ) -> dict[str, Any]:
        with self.config_file.open(
            "rb",
        ) as file:
            document = tomllib.load(file)

        if not isinstance(document, dict):
            raise ValueError(
                f"{MODELS_CONFIG_FILE} must contain a TOML table."
            )

        return document

    def _layout(
        self,
        document: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        models = document.get("models")
        if not isinstance(models, dict):
            raise ValueError("models.toml must contain a [models] table.")

        selectors: dict[str, Any] = {}
        profiles: dict[str, dict[str, Any]] = {}
        for name, value in models.items():
            if name in {"orchestrator", "subagent", "active"}:
                selectors[name] = value
            elif isinstance(value, dict):
                self._validate_profile_name(name)
                profiles[name] = value
            else:
                raise ValueError(f"'models.{name}' must be a profile table.")

        if not profiles:
            raise ValueError("models.toml must define at least one model profile.")
        if "orchestrator" in selectors and "active" in selectors:
            raise ValueError(
                "models.toml cannot define both 'models.orchestrator' and "
                "the legacy 'models.active' selector."
            )
        orchestrator = selectors.get("orchestrator", selectors.get("active"))
        if not isinstance(orchestrator, str) or orchestrator not in profiles:
            raise ValueError("models.orchestrator must select an existing profile.")
        subagent = selectors.get("subagent")
        if subagent is not None and (
            not isinstance(subagent, str) or subagent not in profiles
        ):
            raise ValueError("models.subagent must select an existing profile.")
        return selectors, profiles

    def names(self) -> tuple[str, ...]:
        _, profiles = self._layout(self._load_document())
        return tuple(profiles)

    def orchestrator_name(self) -> str:
        selectors, _ = self._layout(self._load_document())
        return str(selectors.get("orchestrator") or selectors.get("active"))

    def active_name(self) -> str:
        return self.orchestrator_name()

    def subagent_name(self) -> str:
        selectors, _ = self._layout(self._load_document())
        return str(selectors.get("subagent") or self.orchestrator_name())

    def get(self, name: str | None = None) -> ModelConfig:
        selectors, profiles = self._layout(self._load_document())
        selected = name or str(
            selectors.get("orchestrator") or selectors.get("active")
        )
        try:
            raw = profiles[selected]
        except KeyError as error:
            raise KeyError(f"Unknown model profile: {selected}") from error

        retry_raw = raw.get("retry", {})
        if not isinstance(retry_raw, dict):
            raise ValueError(f"'models.{selected}.retry' must be a table.")

        plaintext = raw.get("api_key")
        encrypted = raw.get("encrypted_key", "")
        if plaintext is not None and (
            not isinstance(plaintext, str) or not plaintext
        ):
            raise ValueError(f"'models.{selected}.api_key' must be a string.")
        if plaintext is None and (
            not isinstance(encrypted, str) or not encrypted
        ):
            raise ValueError(
                f"'models.{selected}' must define api_key or encrypted_key."
            )

        retry = RetryConfig(
            max_attempts=_positive_number(
                retry_raw,
                "max_attempts",
                12,
                integer=True,
                section=f"models.{selected}.retry",
            ),
            request_timeout=float(
                _positive_number(
                    retry_raw,
                    "request_timeout",
                    120.0,
                    section=f"models.{selected}.retry",
                )
            ),
            initial_backoff=float(
                _nonnegative_number(
                    retry_raw,
                    "initial_backoff",
                    1.0,
                    section=f"models.{selected}.retry",
                )
            ),
            max_backoff=float(
                _nonnegative_number(
                    retry_raw,
                    "max_backoff",
                    30.0,
                    section=f"models.{selected}.retry",
                )
            ),
        )
        if retry.initial_backoff > retry.max_backoff:
            raise ValueError(
                f"'models.{selected}.retry.initial_backoff' cannot exceed "
                "max_backoff."
            )

        return ModelConfig(
            host=_required_profile_string(raw, "host", selected),
            encrypted_key=str(encrypted),
            id=_required_profile_string(raw, "id", selected),
            max_input_tokens=_positive_profile_int(
                raw,
                "max_input_tokens",
                selected,
            ),
            max_output_tokens=_positive_profile_int(
                raw,
                "max_output_tokens",
                selected,
            ),
            reasoning_effort=_optional_profile_string(
                raw,
                "reasoning_effort",
                selected,
            ),
            retry=retry,
            name=selected,
            _plaintext_api_key=(
                str(plaintext) if plaintext is not None else None
            ),
        )

    def set_orchestrator(self, name: str) -> None:
        self._set_selector("orchestrator", name)

    def set_active(self, name: str) -> None:
        self.set_orchestrator(name)

    def set_subagent(self, name: str | None) -> None:
        if name is None:
            document = self._load_document()
            models = document["models"]
            assert isinstance(models, dict)
            models.pop("subagent", None)
            self._save_document(document)
            return
        self._set_selector("subagent", name)

    def _set_selector(self, selector: str, name: str) -> None:
        document = self._load_document()
        _, profiles = self._layout(document)
        if name not in profiles:
            raise KeyError(f"Unknown model profile: {name}")
        models = document["models"]
        assert isinstance(models, dict)
        models[selector] = name
        if selector == "orchestrator":
            models.pop("active", None)
        self._save_document(document)

    def add(self, name: str, *, copy_from: str | None = None) -> None:
        self._validate_profile_name(name)
        document = self._load_document()
        _, profiles = self._layout(document)
        if name in profiles:
            raise ValueError(f"Model profile already exists: {name}")
        source = copy_from or self.orchestrator_name()
        if source not in profiles:
            raise KeyError(f"Unknown model profile: {source}")
        models = document["models"]
        assert isinstance(models, dict)
        models[name] = deepcopy(profiles[source])
        self._save_document(document)

    def delete(self, name: str) -> None:
        document = self._load_document()
        selectors, profiles = self._layout(document)
        if name not in profiles:
            raise KeyError(f"Unknown model profile: {name}")
        selected = {
            str(selectors.get("orchestrator", selectors.get("active"))),
            str(selectors.get("subagent")) if selectors.get("subagent") else "",
        }
        if name in selected:
            raise ValueError("Cannot delete a selected model profile.")
        models = document["models"]
        assert isinstance(models, dict)
        del models[name]
        self._save_document(document)

    def set(self, *, name: str | None = None, **values: Any) -> None:
        allowed = {
            "host",
            "id",
            "max_input_tokens",
            "max_output_tokens",
            "reasoning_effort",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError("Unknown model settings: " + ", ".join(unknown))
        document, profile_name, profile = self._mutable_profile(name)
        for key, value in values.items():
            if key in {"host", "id"} and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"models.{profile_name}.{key} must be a string.")
            if key in {"max_input_tokens", "max_output_tokens"} and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"models.{profile_name}.{key} must be positive.")
            if key == "reasoning_effort" and value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError("reasoning_effort must be a string or None.")
            if value is None:
                profile.pop(key, None)
            else:
                profile[key] = value
        self._save_document(document)

    def set_retry(self, *, name: str | None = None, **values: Any) -> None:
        allowed = {
            "max_attempts",
            "request_timeout",
            "initial_backoff",
            "max_backoff",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError("Unknown retry settings: " + ", ".join(unknown))
        document, profile_name, profile = self._mutable_profile(name)
        retry = profile.setdefault("retry", {})
        if not isinstance(retry, dict):
            raise ValueError(f"models.{profile_name}.retry must be a table.")
        for key, value in values.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"retry.{key} must be numeric.")
            if key == "max_attempts" and not isinstance(value, int):
                raise ValueError("retry.max_attempts must be an integer.")
            minimum = 1 if key == "max_attempts" else 0
            if value < minimum or (key == "request_timeout" and value == 0):
                raise ValueError(f"retry.{key} is outside its valid range.")
            retry[key] = value
        if float(retry.get("initial_backoff", 1.0)) > float(
            retry.get("max_backoff", 30.0)
        ):
            raise ValueError("retry.initial_backoff cannot exceed max_backoff.")
        self._save_document(document)

    def set_api_key(self, value: str, *, name: str | None = None) -> None:
        if not value:
            raise ValueError("API key cannot be empty.")
        document, _, profile = self._mutable_profile(name)
        profile["encrypted_key"] = self.encrypt_secret(value)
        profile.pop("api_key", None)
        self._save_document(document)

    def set_host(self, value: str, *, name: str | None = None) -> None:
        self.set(name=name, host=value)

    def set_model_id(self, value: str, *, name: str | None = None) -> None:
        self.set(name=name, id=value)

    def _mutable_profile(
        self,
        name: str | None,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        document = self._load_document()
        selectors, profiles = self._layout(document)
        selected = name or str(
            selectors.get("orchestrator") or selectors.get("active")
        )
        try:
            profile = profiles[selected]
        except KeyError as error:
            raise KeyError(f"Unknown model profile: {selected}") from error
        return document, selected, profile

    def _save_document(self, document: dict[str, Any]) -> None:
        self._layout(document)
        models = document["models"]
        assert isinstance(models, dict)
        lines = ["[models]"]
        for selector in ("orchestrator", "subagent", "active"):
            value = models.get(selector)
            if value is not None:
                lines.append(f"{selector} = {json.dumps(value)}")
        for name, profile in models.items():
            if name in {"orchestrator", "subagent", "active"}:
                continue
            assert isinstance(profile, dict)
            lines.extend(("", f"[models.{name}]"))
            retry = profile.get("retry")
            for key, value in profile.items():
                if key == "retry" or value is None:
                    continue
                lines.append(f"{key} = {_toml_value(value)}")
            if isinstance(retry, dict) and retry:
                lines.extend(("", f"[models.{name}.retry]"))
                for key, value in retry.items():
                    lines.append(f"{key} = {_toml_value(value)}")
        temporary = self.config_file.with_suffix(".toml.tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, self.config_file)

    @staticmethod
    def _validate_profile_name(name: str) -> None:
        if not isinstance(name, str) or _PROFILE_NAME.fullmatch(name) is None:
            raise ValueError(f"Invalid model profile name: {name!r}")

    @staticmethod
    def encrypt_secret(value: str) -> str:
        key = load_encryption_key().encode("ascii")
        return Fernet(key).encrypt(value.encode("utf-8")).decode("ascii")

    @staticmethod
    def decrypt_secret(
        encrypted_value: str,
    ) -> str:
        encryption_key: str = load_encryption_key()

        if not encryption_key:
            raise RuntimeError(
                "Encryption key is not defined."
            )

        try:
            fernet = Fernet(
                encryption_key.encode("ascii")
            )
        except (
            ValueError,
            TypeError,
        ) as error:
            raise RuntimeError(
                "Configured Citra encryption key is invalid."
            ) from error

        try:
            plaintext = fernet.decrypt(
                encrypted_value.encode("ascii")
            )
        except InvalidToken as error:
            raise RuntimeError(
                "Unable to decrypt configured API key."
            ) from error

        try:
            return plaintext.decode(
                "utf-8",
            )
        except UnicodeDecodeError as error:
            raise RuntimeError(
                "Decrypted API key is not valid UTF-8."
            ) from error


def _required_profile_string(
    raw: dict[str, Any],
    key: str,
    profile: str,
) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'models.{profile}.{key}' must be a non-empty string.")
    return value.strip()


def _optional_profile_string(
    raw: dict[str, Any],
    key: str,
    profile: str,
) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'models.{profile}.{key}' must be a non-empty string.")
    return value.strip()


def _positive_profile_int(raw: dict[str, Any], key: str, profile: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"'models.{profile}.{key}' must be a positive integer.")
    return value


def _positive_number(
    raw: dict[str, Any],
    key: str,
    default: int | float,
    *,
    section: str,
    integer: bool = False,
) -> int | float:
    value = raw.get(key, default)
    valid_type = isinstance(value, int) if integer else isinstance(value, (int, float))
    if not valid_type or isinstance(value, bool) or value <= 0:
        raise ValueError(f"'{section}.{key}' must be positive.")
    return value


def _nonnegative_number(
    raw: dict[str, Any],
    key: str,
    default: float,
    *,
    section: str,
) -> int | float:
    value = raw.get(key, default)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError(f"'{section}.{key}' cannot be negative.")
    return value


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise ValueError(f"Unsupported model configuration value: {value!r}")

def _config_home() -> Path:
    xdg = os.environ.get(
        "XDG_CONFIG_HOME"
    )

    if xdg:
        return Path(xdg).expanduser()

    return Path.home() / ".config"


def _encryption_key_path() -> Path:
    return (
        _config_home()
        / "citra"
        / "encryption.key"
    )


def load_encryption_key() -> str:
    """
    Load Citra's Fernet key.

    Creates the key on first use and restricts permissions where supported.
    """

    key_path = _encryption_key_path()

    key_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not key_path.exists():
        key = Fernet.generate_key()

        key_path.write_bytes(
            key,
        )

        try:
            key_path.chmod(
                0o600,
            )
        except OSError:
            pass

    return key_path.read_text(
        encoding="ascii",
    ).strip()
