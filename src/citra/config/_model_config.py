from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
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
        config_file: Path,
    ) -> None:
        self.config_file = config_file

    @classmethod
    def load(
        cls,
        config_path: str | Path,
    ) -> ModelConfigStore:
        config_dir = (
            Path(config_path)
            .expanduser()
            .resolve()
        )

        config_file = (
            config_dir / MODELS_CONFIG_FILE
        )

        if not config_file.is_file():
            raise FileNotFoundError(
                f"Model config file not found: {config_file}"
            )

        return cls(
            config_file,
        )

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