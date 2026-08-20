from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ModelConfig:
    host: str
    api_key: str
    id: str
    max_tokens: int


@dataclass(frozen=True)
class WebSearchConfig:
    host_url: str


@dataclass(frozen=True)
class MessageContextConfig:
    uncompressed_messages: int


@dataclass(frozen=True)
class CitraConfig:
    model: ModelConfig
    web_search: WebSearchConfig
    message_context: MessageContextConfig

    @classmethod
    def load(
        cls,
        path: str | Path = ".citra/config.toml",
    ) -> CitraConfig:
        config_path = Path(path)

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Citra config file not found: {config_path}"
            )

        with config_path.open("rb") as file:
            raw = tomllib.load(file)

        try:
            model_raw = raw["model"]
            web_search_raw = raw["web-search"]
            message_context_raw = raw["message-context"]

            model = ModelConfig(
                host=model_raw["host"],
                api_key=model_raw["api_key"],
                id=model_raw["id"],
                max_tokens=model_raw["max_tokens"],
            )

            web_search = WebSearchConfig(
                host_url=web_search_raw["host_url"],
            )

            message_context = MessageContextConfig(
                uncompressed_messages=message_context_raw[
                    "uncompressed_messages"
                ],
            )

        except KeyError as error:
            raise ValueError(
                f"Missing required config value: {error.args[0]}"
            ) from error

        if model.max_tokens <= 0:
            raise ValueError(
                "'model.max_tokens' must be greater than zero."
            )

        if message_context.uncompressed_messages < 0:
            raise ValueError(
                "'message-context.uncompressed_messages' "
                "must be zero or greater."
            )

        return cls(
            model=model,
            web_search=web_search,
            message_context=message_context,
        )