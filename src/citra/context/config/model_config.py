from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
import tomlkit


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
    max_input_tokens:int
    max_output_tokens: int
    reasoning_effort: str | None
    retry: RetryConfig = RetryConfig()


    def decrypt_api_key(self) -> str:
        return decrypt_secret(
            self.encrypted_key
        )



class ModelConfigStore:
    """
    Updates the [model] section of Citra's config.toml.

    API keys are encrypted before being written to disk.
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

        self.config_path = Path(config_path).resolve()

        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Citra config file not found: {self.config_path}"
            )

    def _load(self):
        with self.config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return tomlkit.load(file)

    def _save(self, document) -> None:
        """
        Atomically replace config.toml so an interrupted write
        does not leave the configuration corrupted.
        """
        temp_path = self.config_path.with_suffix(
            self.config_path.suffix + ".tmp"
        )

        try:
            with temp_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                tomlkit.dump(
                    document,
                    file,
                )

            os.replace(
                temp_path,
                self.config_path,
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _fernet(self) -> Fernet:
        try:
            return Fernet(
                _env_key().encode("ascii")
            )
        except (ValueError, TypeError) as error:
            raise RuntimeError(
                "Citra's encryption key is invalid."
            ) from error

    def _encrypt_api_key(
        self,
        api_key: str,
    ) -> str:
        if not api_key:
            raise ValueError(
                "API key cannot be empty."
            )

        return (
            self._fernet()
            .encrypt(
                api_key.encode("utf-8")
            )
            .decode("ascii")
        )

    def get(self) -> ModelConfig:
        document = self._load()

        model = document.get("model")

        if model is None:
            raise RuntimeError(
                "Missing [model] section in Citra config."
            )

        try:
            encrypted_key = str(
                model["encrypted_key"]
            )

            retry_raw = model.get(
                "retry",
                {},
            )

            config = ModelConfig(
                host=str(
                    model["host"]
                ),
                encrypted_key=encrypted_key,
                id=str(
                    model["id"]
                ),
                max_input_tokens=int(
                    model["max_input_tokens"]
                ),
                max_output_tokens=int(
                    model["max_output_tokens"]
                ),
                reasoning_effort=(
                    str(model["reasoning_effort"])
                    if model.get("reasoning_effort") is not None
                    else None
                ),
                retry=RetryConfig(
                    max_attempts=int(
                        retry_raw.get(
                            "max_attempts",
                            12,
                        )
                    ),
                    request_timeout=float(
                        retry_raw.get(
                            "request_timeout",
                            120.0,
                        )
                    ),
                    initial_backoff=float(
                        retry_raw.get(
                            "initial_backoff",
                            1.0,
                        )
                    ),
                    max_backoff=float(
                        retry_raw.get(
                            "max_backoff",
                            30.0,
                        )
                    ),
                ),
            )

            self._validate(config)

            return config

        except KeyError as error:
            raise RuntimeError(
                f"Missing required model config value: {error.args[0]}"
            ) from error

    def _validate(
        self,
        model: ModelConfig,
    ) -> None:
        if not model.host.strip():
            raise ValueError(
                "'model.host' cannot be empty."
            )

        if not model.id.strip():
            raise ValueError(
                "'model.id' cannot be empty."
            )

        if model.max_input_tokens <= 0:
            raise ValueError(
                "'model.max_input_tokens' must be greater than zero."
            )

        if model.max_output_tokens <= 0:
            raise ValueError(
                "'model.max_output_tokens' must be greater than zero."
            )

        if model.retry.max_attempts < 1:
            raise ValueError(
                "'model.retry.max_attempts' must be at least 1."
            )

        if model.retry.request_timeout <= 0:
            raise ValueError(
                "'model.retry.request_timeout' must be greater than zero."
            )

        if (
            model.retry.initial_backoff < 0
            or model.retry.max_backoff < 0
        ):
            raise ValueError(
                "Model retry backoff values cannot be negative."
            )

        if (
            model.retry.initial_backoff
            > model.retry.max_backoff
        ):
            raise ValueError(
                "'model.retry.initial_backoff' cannot exceed "
                "'model.retry.max_backoff'."
            )

    def set_api_key(
        self,
        api_key: str,
    ) -> None:
        """
        Encrypt and persist a new model API key.
        """
        document = self._load()

        model = document.get("model")

        if model is None:
            model = tomlkit.table()
            document["model"] = model

        model["encrypted_key"] = (
            self._encrypt_api_key(api_key)
        )

        # Remove old plaintext config if it exists.
        model.pop(
            "api_key",
            None,
        )

        self._save(document)

    def set_model_id(
        self,
        model_id: str,
    ) -> None:
        if not model_id.strip():
            raise ValueError(
                "Model ID cannot be empty."
            )

        self.set(
            id=model_id,
        )

    def set_host(
        self,
        host: str,
    ) -> None:
        if not host.strip():
            raise ValueError(
                "Model host cannot be empty."
            )

        self.set(
            host=host,
        )

    def set(
        self,
        **values: Any,
    ) -> None:
        """
        Update one or more non-secret [model] fields.

        Example:
            store.set(
                id="z-ai/glm-5.2:free",
                max_input_tokens=131072,
                max_output_tokens=8192,
            )
        """
        if not values:
            return

        if "api_key" in values:
            raise ValueError(
                "Use set_api_key() for API keys."
            )

        if "encrypted_key" in values:
            raise ValueError(
                "encrypted_key cannot be assigned directly. "
                "Use set_api_key()."
            )

        document = self._load()

        model = document.get("model")

        if model is None:
            model = tomlkit.table()
            document["model"] = model

        for key, value in values.items():
            if value is None:
                model.pop(
                    key,
                    None,
                )
            else:
                model[key] = value

        self._save(document)

    def set_retry(
        self,
        **values: Any,
    ) -> None:
        allowed = {
            "max_attempts",
            "request_timeout",
            "initial_backoff",
            "max_backoff",
        }

        unknown = set(values) - allowed

        if unknown:
            raise ValueError(
                "Unknown retry setting(s): "
                + ", ".join(sorted(unknown))
            )

        document = self._load()

        model = document.get("model")

        if model is None:
            model = tomlkit.table()
            document["model"] = model

        retry = model.get("retry")

        if retry is None:
            retry = tomlkit.table()
            model["retry"] = retry

        # Work with effective values so cross-field validation
        # also works when changing only one setting.
        max_attempts = int(
            values.get(
                "max_attempts",
                retry.get("max_attempts", 12),
            )
        )

        request_timeout = float(
            values.get(
                "request_timeout",
                retry.get("request_timeout", 120.0),
            )
        )

        initial_backoff = float(
            values.get(
                "initial_backoff",
                retry.get("initial_backoff", 1.0),
            )
        )

        max_backoff = float(
            values.get(
                "max_backoff",
                retry.get("max_backoff", 30.0),
            )
        )

        if max_attempts < 1:
            raise ValueError(
                "retry.max_attempts must be at least 1."
            )

        if request_timeout <= 0:
            raise ValueError(
                "retry.request_timeout must be greater than zero."
            )

        if initial_backoff < 0:
            raise ValueError(
                "retry.initial_backoff cannot be negative."
            )

        if max_backoff < 0:
            raise ValueError(
                "retry.max_backoff cannot be negative."
            )

        if initial_backoff > max_backoff:
            raise ValueError(
                "retry.initial_backoff cannot exceed retry.max_backoff."
            )

        for key, value in values.items():
            retry[key] = value

        self._save(document)

    def get_model_section(
        self,
    ) -> dict[str, Any]:
        """
        Return the model section without decrypting secrets.
        """
        document = self._load()

        model = document.get(
            "model",
            {},
        )

        return dict(model)

def decrypt_secret(
    encrypted_value: str
) -> str:
    encryption_key = _env_key()

    if not encryption_key:
        raise RuntimeError(
            f"{encryption_key} is not defined. "
            "It is required to decrypt secrets from the Citra config."
        )

    try:
        fernet = Fernet(
            encryption_key.encode("ascii")
        )
    except (ValueError, TypeError) as error:
        raise RuntimeError(
            f"{encryption_key} is not a valid Fernet encryption key."
        ) from error

    try:
        plaintext = fernet.decrypt(
            encrypted_value.encode("ascii")
        )
    except InvalidToken as error:
        raise RuntimeError(
            "Unable to decrypt the configured API key. "
            f"Check that {encryption_key} contains the correct encryption key."
        ) from error

    try:
        return plaintext.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(
            "The decrypted API key is not valid UTF-8."
        ) from error

def _env_key() -> str:
    """
    Load Citra's Fernet encryption key from the user's config directory.

    Creates the key if it does not already exist.

    Linux paths:
        $XDG_CONFIG_HOME/citra/encryption.key

    or, if XDG_CONFIG_HOME is unset:
        ~/.config/citra/encryption.key
    """
    xdg_config_home = os.environ.get(
        "XDG_CONFIG_HOME"
    )

    if xdg_config_home:
        config_dir = Path(
            xdg_config_home
        ).expanduser() / "citra"
    else:
        config_dir = (
            Path.home()
            / ".config"
            / "citra"
        )

    config_dir.mkdir(
        mode=0o700,
        parents=True,
        exist_ok=True,
    )

    # Tighten permissions even if the directory already existed.
    os.chmod(
        config_dir,
        0o700,
    )

    key_path = (
        config_dir
        / "encryption.key"
    )

    if key_path.exists():
        if not key_path.is_file():
            raise RuntimeError(
                f"Citra encryption key path is not a file: {key_path}"
            )

        # Ensure only the current user can read/write it.
        os.chmod(
            key_path,
            0o600,
        )

        key = key_path.read_text(
            encoding="ascii",
        ).strip()

        if not key:
            raise RuntimeError(
                f"Citra encryption key file is empty: {key_path}"
            )

        try:
            Fernet(
                key.encode("ascii")
            )
        except (ValueError, TypeError) as error:
            raise RuntimeError(
                f"Citra encryption key file contains an invalid "
                f"Fernet key: {key_path}"
            ) from error

        return key

    key = Fernet.generate_key().decode(
        "ascii"
    )

    # O_EXCL avoids accidentally overwriting a key if another process
    # creates it between our existence check and creation.
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
    )

    fd = os.open(
        key_path,
        flags,
        0o600,
    )

    try:
        os.write(
            fd,
            (
                key + "\n"
            ).encode("ascii"),
        )
    finally:
        os.close(fd)

    os.chmod(
        key_path,
        0o600,
    )

    return key