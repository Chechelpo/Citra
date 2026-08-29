from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
import tomlkit


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
    # Preserve the historical constructor order; ``name`` is additive metadata.
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
        """Return the configured API credential.

        ``_plaintext_api_key`` exists only to read the oldest legacy config
        shape. All writes performed by ``ModelConfigStore`` persist encrypted
        credentials.
        """
        if self._plaintext_api_key is not None:
            return self._plaintext_api_key
        return decrypt_secret(self.encrypted_key)


class ModelConfigStore:
    """Load and persist Citra model profiles.

    Canonical configuration lives in ``.citra/config/models.toml`` and uses
    one persisted active selector plus any number of named profiles::

        [models]
        active = "primary"

        [models.primary]
        ...

    The historical singular ``[model]`` table remains readable and mutable as
    the implicit ``default`` profile. Operations that require multiple
    profiles migrate it atomically to the canonical ``[models]`` layout.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
    ) -> None:
        if config_path is None:
            config_path = os.environ.get("CITRA_CONFIG_PATH")

        if not config_path:
            raise RuntimeError(
                "CITRA_CONFIG_PATH is not defined and no config path "
                "was supplied."
            )

        resolved = Path(config_path).resolve()
        if resolved.is_dir():
            resolved = resolved / "models.toml"
        self.config_path = resolved

        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Citra model config file not found: {self.config_path}"
            )

    def _load(self):
        with self.config_path.open("r", encoding="utf-8") as file:
            return tomlkit.load(file)

    def _save(self, document) -> None:
        """Atomically replace the model configuration file."""
        temp_path = self.config_path.with_suffix(
            self.config_path.suffix + ".tmp"
        )

        try:
            with temp_path.open("w", encoding="utf-8") as file:
                tomlkit.dump(document, file)
            os.replace(temp_path, self.config_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _is_table(value: Any) -> bool:
        return hasattr(value, "get") and hasattr(value, "items")

    @staticmethod
    def _validate_profile_name(name: str) -> str:
        normalized = name.strip()
        if not normalized:
            raise ValueError("Model profile name cannot be empty.")
        if _PROFILE_NAME.fullmatch(normalized) is None:
            raise ValueError(
                "Model profile names may contain only letters, numbers, "
                "underscores, and hyphens, and must start with a letter or "
                "number."
            )
        return normalized

    def _layout(
        self,
        document,
    ) -> tuple[str, str, str, tuple[str, ...]]:
        models = document.get("models")
        legacy = document.get("model")

        if models is not None and legacy is not None:
            raise RuntimeError(
                "Citra config cannot contain both [model] and [models]."
            )

        if models is not None:
            if not self._is_table(models):
                raise RuntimeError("[models] must be a TOML table.")

            names: list[str] = []
            for key, value in models.items():
                if key in {"active", "orchestrator", "subagent"}:
                    continue
                raw_name = str(key)
                profile_name = self._validate_profile_name(raw_name)
                if profile_name != raw_name:
                    raise ValueError(
                        "Model profile names cannot contain surrounding "
                        "whitespace."
                    )
                if not self._is_table(value):
                    raise RuntimeError(
                        f"'models.{key}' must be a TOML table."
                    )
                names.append(profile_name)

            if not names:
                raise RuntimeError(
                    "[models] must contain at least one model profile."
                )

            orchestrator, subagent = self._resolve_role_selectors(
                models, tuple(names)
            )
            return "models", orchestrator, subagent, tuple(names)

        if legacy is not None:
            if not self._is_table(legacy):
                raise RuntimeError("[model] must be a TOML table.")
            return (
                "model",
                _LEGACY_PROFILE_NAME,
                _LEGACY_PROFILE_NAME,
                (_LEGACY_PROFILE_NAME,),
            )

        raise RuntimeError(
            "Missing model configuration. Expected [models] or legacy [model]."
        )

    @staticmethod
    def _resolve_role_selectors(
        models: Any,
        names: tuple[str, ...],
    ) -> tuple[str, str]:
        """Resolve the orchestrator and subagent profile names.

        ``orchestrator`` is the modern selector. ``active`` remains a
        deprecated alias that is still accepted for backwards
        compatibility. ``subagent`` defaults to the orchestrator profile
        when omitted, which preserves the historical single-model
        behavior.
        """
        active_raw = models.get("active")
        orchestrator_raw = models.get("orchestrator", active_raw)
        if orchestrator_raw is None:
            raise RuntimeError(
                "Missing required model selector: 'models.orchestrator' "
                "(or legacy 'models.active')."
            )
        if not isinstance(orchestrator_raw, str) or not orchestrator_raw.strip():
            raise RuntimeError(
                "Missing required model selector: 'models.orchestrator' "
                "(or legacy 'models.active')."
            )
        if orchestrator_raw is active_raw:
            orchestrator = ModelConfigStore._validate_profile_name(
                orchestrator_raw
            )
        else:
            orchestrator = ModelConfigStore._validate_profile_name(
                orchestrator_raw
            )

        subagent_raw = models.get("subagent")
        if subagent_raw is None:
            subagent = orchestrator
        else:
            if not isinstance(subagent_raw, str) or not subagent_raw.strip():
                raise RuntimeError(
                    "Missing required model selector: 'models.subagent'."
                )
            subagent = ModelConfigStore._validate_profile_name(subagent_raw)

        if orchestrator not in names:
            raise RuntimeError(
                f"Orchestrator model profile '{orchestrator}' does not exist."
            )
        if subagent not in names:
            raise RuntimeError(
                f"Subagent model profile '{subagent}' does not exist."
            )

        if (
            active_raw is not None
            and orchestrator_raw is not active_raw
            and str(active_raw).strip() != orchestrator
        ):
            raise RuntimeError(
                "Citra config cannot declare both 'models.active' and "
                "'models.orchestrator'."
            )

        return orchestrator, subagent

    def _profile_table(
        self,
        document,
        name: str | None = None,
    ) -> tuple[str, Any]:
        layout, orchestrator, _, names = self._layout(document)
        selected = (
            orchestrator
            if name is None
            else self._validate_profile_name(name)
        )

        if selected not in names:
            raise KeyError(f"Unknown model profile: {selected}")

        if layout == "model":
            return _LEGACY_PROFILE_NAME, document["model"]
        return selected, document["models"][selected]

    def _migrate_legacy(self, document):
        layout, _, _, _ = self._layout(document)
        if layout == "models":
            return document["models"]

        legacy = deepcopy(document["model"])
        self._normalize_profile_for_canonical(legacy)
        self._config_from_table(_LEGACY_PROFILE_NAME, legacy)

        models = tomlkit.table()
        models["active"] = _LEGACY_PROFILE_NAME
        models[_LEGACY_PROFILE_NAME] = legacy
        document.pop("model")
        document["models"] = models
        return models

    def _normalize_profile_for_canonical(self, model: Any) -> None:
        """Canonicalize legacy aliases before writing a named profile."""
        if model.get("encrypted_key") is None:
            plaintext = model.get("api_key")
            if plaintext is not None:
                model["encrypted_key"] = self._encrypt_api_key(str(plaintext))
        model.pop("api_key", None)

        legacy_max_tokens = model.get("max_tokens")
        if legacy_max_tokens is not None:
            if model.get("max_input_tokens") is None:
                model["max_input_tokens"] = legacy_max_tokens
            if model.get("max_output_tokens") is None:
                model["max_output_tokens"] = legacy_max_tokens
            model.pop("max_tokens", None)

    def _fernet(self) -> Fernet:
        try:
            return Fernet(_env_key().encode("ascii"))
        except (ValueError, TypeError) as error:
            raise RuntimeError("Citra's encryption key is invalid.") from error

    def _encrypt_api_key(self, api_key: str) -> str:
        if not api_key:
            raise ValueError("API key cannot be empty.")
        return (
            self._fernet()
            .encrypt(api_key.encode("utf-8"))
            .decode("ascii")
        )

    def active_name(self) -> str:
        """Return the persisted orchestrator profile name.

        The orchestrator name is the modern selector; the legacy
        ``active`` key is normalized to the orchestrator name on load.
        """
        _, orchestrator, _, _ = self._layout(self._load())
        return orchestrator

    def orchestrator_name(self) -> str:
        """Return the orchestrator profile name."""
        _, orchestrator, _, _ = self._layout(self._load())
        return orchestrator

    def subagent_name(self) -> str:
        """Return the subagent profile name.

        When ``models.subagent`` is omitted, the subagent profile is
        the orchestrator profile. Operators can therefore opt into a
        dedicated subagent model only when they want one.
        """
        _, _, subagent, _ = self._layout(self._load())
        return subagent

    def names(self) -> tuple[str, ...]:
        """Return model profile names in configuration order."""
        _, _, _, names = self._layout(self._load())
        return names

    def get(self, name: str | None = None) -> ModelConfig:
        """Resolve a named profile, or the active profile when omitted."""
        document = self._load()
        selected, model = self._profile_table(document, name)

        try:
            encrypted_key_raw = model.get("encrypted_key")
            plaintext_key_raw = model.get("api_key")
            if encrypted_key_raw is None and plaintext_key_raw is None:
                raise KeyError("encrypted_key")

            encrypted_key = (
                str(encrypted_key_raw)
                if encrypted_key_raw is not None
                else ""
            )
            plaintext_key = (
                str(plaintext_key_raw)
                if encrypted_key_raw is None and plaintext_key_raw is not None
                else None
            )

            legacy_max_tokens = model.get("max_tokens")
            max_input_raw = model.get(
                "max_input_tokens",
                legacy_max_tokens,
            )
            max_output_raw = model.get(
                "max_output_tokens",
                legacy_max_tokens,
            )
            if max_input_raw is None:
                raise KeyError("max_input_tokens")
            if max_output_raw is None:
                raise KeyError("max_output_tokens")

            retry_raw = model.get("retry", {})
            if not self._is_table(retry_raw):
                raise ValueError(
                    f"'models.{selected}.retry' must be a TOML table."
                )

            config = ModelConfig(
                name=selected,
                host=str(model["host"]),
                encrypted_key=encrypted_key,
                id=str(model["id"]),
                max_input_tokens=int(max_input_raw),
                max_output_tokens=int(max_output_raw),
                reasoning_effort=(
                    str(model["reasoning_effort"])
                    if model.get("reasoning_effort") is not None
                    else None
                ),
                retry=RetryConfig(
                    max_attempts=int(retry_raw.get("max_attempts", 12)),
                    request_timeout=float(
                        retry_raw.get("request_timeout", 120.0)
                    ),
                    initial_backoff=float(
                        retry_raw.get("initial_backoff", 1.0)
                    ),
                    max_backoff=float(retry_raw.get("max_backoff", 30.0)),
                ),
                _plaintext_api_key=plaintext_key,
            )
            self._validate(config)
            return config
        except KeyError as error:
            raise RuntimeError(
                f"Missing required model config value for profile "
                f"'{selected}': {error.args[0]}"
            ) from error
        except (TypeError, ValueError) as error:
            if isinstance(error, ValueError) and str(error).startswith("'"):
                raise
            raise ValueError(
                f"Invalid model configuration for profile '{selected}': {error}"
            ) from error

    def _validate(self, model: ModelConfig) -> None:
        prefix = f"models.{model.name}"
        if not model.host.strip():
            raise ValueError(f"'{prefix}.host' cannot be empty.")
        if not model.id.strip():
            raise ValueError(f"'{prefix}.id' cannot be empty.")
        if not model.encrypted_key and not model._plaintext_api_key:
            raise ValueError(f"'{prefix}.encrypted_key' cannot be empty.")
        if model.max_input_tokens <= 0:
            raise ValueError(
                f"'{prefix}.max_input_tokens' must be greater than zero."
            )
        if model.max_output_tokens <= 0:
            raise ValueError(
                f"'{prefix}.max_output_tokens' must be greater than zero."
            )
        if model.retry.max_attempts < 1:
            raise ValueError(
                f"'{prefix}.retry.max_attempts' must be at least 1."
            )
        if model.retry.request_timeout <= 0:
            raise ValueError(
                f"'{prefix}.retry.request_timeout' must be greater than zero."
            )
        if model.retry.initial_backoff < 0 or model.retry.max_backoff < 0:
            raise ValueError("Model retry backoff values cannot be negative.")
        if model.retry.initial_backoff > model.retry.max_backoff:
            raise ValueError(
                f"'{prefix}.retry.initial_backoff' cannot exceed "
                f"'{prefix}.retry.max_backoff'."
            )

    def set_active(self, name: str) -> None:
        """Set the orchestrator profile.

        This is the legacy API; prefer :meth:`set_orchestrator` for new
        callers. The persisted key remains ``orchestrator`` while the
        deprecated ``active`` alias is also updated to keep existing
        tools that read the legacy selector working.
        """
        selected = self._validate_profile_name(name)
        document = self._load()
        layout, orchestrator, _, names = self._layout(document)
        if selected not in names:
            raise KeyError(f"Unknown model profile: {selected}")
        if selected == orchestrator:
            return
        if layout == "model":
            # A legacy file has only the implicit default profile.
            return
        models = document["models"]
        models["orchestrator"] = selected
        if "active" in models:
            models["active"] = selected
        self._save(document)

    def set_orchestrator(self, name: str) -> None:
        """Persist a new orchestrator profile."""
        self.set_active(name)

    def set_subagent(self, name: str) -> None:
        """Persist a new subagent profile.

        Passing the orchestrator profile name clears the dedicated
        ``subagent`` selector so the configuration reflects that
        subagents reuse the orchestrator's profile.
        """
        selected = self._validate_profile_name(name)
        document = self._load()
        layout, orchestrator, _, names = self._layout(document)
        if layout == "model":
            raise ValueError(
                "Cannot set a dedicated subagent profile on a legacy "
                "[model] config; migrate to [models] first."
            )
        if selected not in names:
            raise KeyError(f"Unknown model profile: {selected}")
        models = document["models"]
        if selected == orchestrator:
            models.pop("subagent", None)
        else:
            models["subagent"] = selected
        self._save(document)

    def add(
        self,
        name: str,
        *,
        copy_from: str | None = None,
    ) -> None:
        """Add a valid profile by cloning an existing profile.

        When ``copy_from`` is omitted, the orchestrator profile is
        cloned. This avoids persisting an unusable half-configured
        profile.
        """
        selected = self._validate_profile_name(name)
        document = self._load()
        _, orchestrator, _, names = self._layout(document)
        if selected in names:
            raise ValueError(f"Model profile already exists: {selected}")

        source_name = (
            orchestrator
            if copy_from is None
            else self._validate_profile_name(copy_from)
        )
        if source_name not in names:
            raise KeyError(f"Unknown model profile: {source_name}")

        models = self._migrate_legacy(document)
        source = models[source_name]
        clone = deepcopy(source)
        self._normalize_profile_for_canonical(clone)
        self._config_from_table(selected, clone)
        models[selected] = clone
        self._save(document)

    def delete(self, name: str) -> None:
        selected = self._validate_profile_name(name)
        document = self._load()
        layout, orchestrator, subagent, names = self._layout(document)
        if selected not in names:
            raise KeyError(f"Unknown model profile: {selected}")
        if selected == orchestrator or selected == subagent:
            raise ValueError(
                "Cannot delete the active or subagent model profile. "
                "Activate or assign another profile first."
            )
        if layout == "model":
            raise ValueError("Cannot delete the only model profile.")
        document["models"].pop(selected)
        # If the subagent selector pointed at the deleted profile, fall
        # back to the orchestrator profile so the layout remains valid.
        if "subagent" in document["models"]:
            del document["models"]["subagent"]
        self._save(document)

    def set_api_key(
        self,
        api_key: str,
        *,
        name: str | None = None,
    ) -> None:
        """Encrypt and persist a new API key for one profile."""
        document = self._load()
        _, model = self._profile_table(document, name)
        model["encrypted_key"] = self._encrypt_api_key(api_key)
        model.pop("api_key", None)
        self._save(document)

    def set_model_id(
        self,
        model_id: str,
        *,
        name: str | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("Model ID cannot be empty.")
        self.set(name=name, id=model_id)

    def set_host(
        self,
        host: str,
        *,
        name: str | None = None,
    ) -> None:
        if not host.strip():
            raise ValueError("Model host cannot be empty.")
        self.set(name=name, host=host)

    def set(
        self,
        *,
        name: str | None = None,
        **values: Any,
    ) -> None:
        """Update one or more non-secret fields on one model profile."""
        if not values:
            return
        if "api_key" in values:
            raise ValueError("Use set_api_key() for API keys.")
        if "encrypted_key" in values:
            raise ValueError(
                "encrypted_key cannot be assigned directly. Use set_api_key()."
            )

        allowed = {
            "host",
            "id",
            "max_input_tokens",
            "max_output_tokens",
            "reasoning_effort",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                "Unknown model setting(s): " + ", ".join(sorted(unknown))
            )

        document = self._load()
        selected, model = self._profile_table(document, name)
        candidate = deepcopy(model)
        for key, value in values.items():
            if value is None:
                candidate.pop(key, None)
            else:
                candidate[key] = value

        # Validate the complete effective profile before writing anything.
        self._config_from_table(selected, candidate)

        for key, value in values.items():
            if value is None:
                model.pop(key, None)
            else:
                model[key] = value
        self._save(document)

    def _config_from_table(self, name: str, model: Any) -> ModelConfig:
        """Validate a modified table using the same parser as ``get``."""
        # Avoid a temporary file by parsing through a small in-memory document.
        document = tomlkit.document()
        models = tomlkit.table()
        models["active"] = name
        models[name] = deepcopy(model)
        document["models"] = models

        selected, table = self._profile_table(document, name)
        encrypted_key_raw = table.get("encrypted_key")
        plaintext_key_raw = table.get("api_key")
        if encrypted_key_raw is None and plaintext_key_raw is None:
            raise RuntimeError(
                f"Missing required model config value for profile "
                f"'{selected}': encrypted_key"
            )
        legacy_max_tokens = table.get("max_tokens")
        max_input_raw = table.get("max_input_tokens", legacy_max_tokens)
        max_output_raw = table.get("max_output_tokens", legacy_max_tokens)
        if max_input_raw is None or max_output_raw is None:
            missing = (
                "max_input_tokens"
                if max_input_raw is None
                else "max_output_tokens"
            )
            raise RuntimeError(
                f"Missing required model config value for profile "
                f"'{selected}': {missing}"
            )
        retry_raw = table.get("retry", {})
        if not self._is_table(retry_raw):
            raise ValueError(f"'models.{selected}.retry' must be a TOML table.")
        config = ModelConfig(
            name=selected,
            host=str(table["host"]),
            encrypted_key=(
                str(encrypted_key_raw)
                if encrypted_key_raw is not None
                else ""
            ),
            id=str(table["id"]),
            max_input_tokens=int(max_input_raw),
            max_output_tokens=int(max_output_raw),
            reasoning_effort=(
                str(table["reasoning_effort"])
                if table.get("reasoning_effort") is not None
                else None
            ),
            retry=RetryConfig(
                max_attempts=int(retry_raw.get("max_attempts", 12)),
                request_timeout=float(retry_raw.get("request_timeout", 120.0)),
                initial_backoff=float(retry_raw.get("initial_backoff", 1.0)),
                max_backoff=float(retry_raw.get("max_backoff", 30.0)),
            ),
            _plaintext_api_key=(
                str(plaintext_key_raw)
                if encrypted_key_raw is None and plaintext_key_raw is not None
                else None
            ),
        )
        self._validate(config)
        return config

    def set_retry(
        self,
        *,
        name: str | None = None,
        **values: Any,
    ) -> None:
        if not values:
            return

        allowed = {
            "max_attempts",
            "request_timeout",
            "initial_backoff",
            "max_backoff",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                "Unknown retry setting(s): " + ", ".join(sorted(unknown))
            )

        document = self._load()
        selected, model = self._profile_table(document, name)
        candidate = deepcopy(model)
        retry = candidate.get("retry")
        if retry is None:
            retry = tomlkit.table()
            candidate["retry"] = retry
        if not self._is_table(retry):
            raise ValueError(f"'models.{selected}.retry' must be a TOML table.")
        for key, value in values.items():
            retry[key] = value

        self._config_from_table(selected, candidate)

        actual_retry = model.get("retry")
        if actual_retry is None:
            actual_retry = tomlkit.table()
            model["retry"] = actual_retry
        for key, value in values.items():
            actual_retry[key] = value
        self._save(document)

    def get_model_section(
        self,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Return one raw profile section without decrypting secrets."""
        document = self._load()
        _, model = self._profile_table(document, name)
        return dict(model)


def decrypt_secret(encrypted_value: str) -> str:
    encryption_key = _env_key()
    if not encryption_key:
        raise RuntimeError(
            "Encryption key is not defined. It is required to decrypt secrets "
            "from the Citra config."
        )

    try:
        fernet = Fernet(encryption_key.encode("ascii"))
    except (ValueError, TypeError) as error:
        raise RuntimeError(
            "Configured Citra encryption key is not a valid Fernet key."
        ) from error

    try:
        plaintext = fernet.decrypt(encrypted_value.encode("ascii"))
    except InvalidToken as error:
        raise RuntimeError(
            "Unable to decrypt the configured API key. Check that Citra is "
            "using the encryption key that created it."
        ) from error

    try:
        return plaintext.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(
            "The decrypted API key is not valid UTF-8."
        ) from error


def _env_key() -> str:
    """Load or create Citra's Fernet key from the user config directory."""
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        config_dir = Path(xdg_config_home).expanduser() / "citra"
    else:
        config_dir = Path.home() / ".config" / "citra"

    config_dir.mkdir(parents=True, exist_ok=True)
    key_path = config_dir / "encryption.key"

    if not key_path.exists():
        key_path.write_bytes(Fernet.generate_key())
        try:
            key_path.chmod(0o600)
        except OSError:
            pass

    try:
        return key_path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise RuntimeError(
            f"Unable to read Citra encryption key: {key_path}"
        ) from error
