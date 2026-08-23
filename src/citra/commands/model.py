from __future__ import annotations

import shlex
from typing import override

from citra.commands import CommandResult
from .command import Command


class ModelCommand(Command):
    """
    Inspect and modify Citra's model configuration.
    """

    id = "model"
    description = "change citra's model config"

    _TOP_LEVEL_FIELDS = {
        "host",
        "id",
        "api_key",
        "max_tokens",
        "reasoning_effort",
    }

    _RETRY_FIELDS = {
        "retry.max_attempts",
        "retry.request_timeout",
        "retry.initial_backoff",
        "retry.max_backoff",
    }

    @override
    def _run(self, args: str) -> CommandResult:
        try:
            parts = shlex.split(args)
        except ValueError as error:
            return CommandResult(
                f"Invalid arguments: {error}"
            )

        if not parts:
            return self._show()

        action = parts[0].lower()

        if action in {
            "show",
            "get",
            "status",
        }:
            return self._show()

        if action in {
            "help",
            "-h",
            "--help",
        }:
            return self._help()

        if action == "set":
            return self._set(parts[1:])

        return CommandResult(
            f"Unknown model command: {action}\n\n"
            f"{self._usage()}"
        )

    def _set(
        self,
        args: list[str],
    ) -> CommandResult:
        if len(args) < 2:
            return CommandResult(
                "Expected a setting and value.\n\n"
                f"{self._usage()}"
            )

        field = (
            args[0]
            .strip()
            .lower()
            .replace("-", "_")
        )

        # Allows values containing spaces without forcing callers
        # to know anything about our parser after shlex processing.
        value = " ".join(
            args[1:]
        )

        all_fields = (
            self._TOP_LEVEL_FIELDS
            | self._RETRY_FIELDS
        )

        if field not in all_fields:
            return CommandResult(
                f"Unknown model setting: {field}\n\n"
                "Available settings:\n"
                + "\n".join(
                    f"  {name}"
                    for name in sorted(all_fields)
                )
            )

        store = self.context.config.model_config_store

        try:
            if field == "api_key":
                store.set_api_key(
                    value
                )

                # Never print the supplied API key.
                return CommandResult(
                    "Model API key updated."
                )

            if field == "host":
                if not value.strip():
                    raise ValueError(
                        "host cannot be empty."
                    )

                store.set_host(
                    value
                )

                return CommandResult(
                    f"model.host = {value}"
                )

            if field == "id":
                if not value.strip():
                    raise ValueError(
                        "id cannot be empty."
                    )

                store.set_model_id(
                    value
                )

                return CommandResult(
                    f"model.id = {value}"
                )

            if field == "max_input_tokens":
                parsed = self._positive_int(
                    value,
                    field,
                )

                store.set(
                    max_input_tokens=parsed
                )

                return CommandResult(
                    f"model.max_input_tokens = {parsed}"
                )

            if field == "max_output_tokens":
                parsed = self._positive_int(
                    value,
                    field,
                )

                store.set(
                    max_output_tokens=parsed
                )

                return CommandResult(
                    f"model.max_output_tokens = {parsed}"
                )

            if field == "reasoning_effort":
                parsed = self._optional_string(
                    value
                )

                store.set(
                    reasoning_effort=parsed
                )

                rendered = (
                    parsed
                    if parsed is not None
                    else "none"
                )

                return CommandResult(
                    f"model.reasoning_effort = {rendered}"
                )

            if field == "retry.max_attempts":
                parsed = self._positive_int(
                    value,
                    field,
                )

                store.set_retry(
                    max_attempts=parsed
                )

                return CommandResult(
                    f"model.retry.max_attempts = {parsed}"
                )

            if field == "retry.request_timeout":
                parsed = self._positive_float(
                    value,
                    field,
                )

                store.set_retry(
                    request_timeout=parsed
                )

                return CommandResult(
                    f"model.retry.request_timeout = {parsed}"
                )

            if field == "retry.initial_backoff":
                parsed = self._nonnegative_float(
                    value,
                    field,
                )

                store.set_retry(
                    initial_backoff=parsed
                )

                return CommandResult(
                    f"model.retry.initial_backoff = {parsed}"
                )

            if field == "retry.max_backoff":
                parsed = self._nonnegative_float(
                    value,
                    field,
                )

                store.set_retry(
                    max_backoff=parsed
                )

                return CommandResult(
                    f"model.retry.max_backoff = {parsed}"
                )

        except (
            ValueError,
            RuntimeError,
            OSError,
        ) as error:
            return CommandResult(
                f"Unable to update model config: {error}"
            )

        # Defensive fallback. Every allowed field should have
        # returned above.
        return CommandResult(
            f"Unsupported model setting: {field}"
        )

    def _show(self) -> CommandResult:
        config = self.context.config.model()

        reasoning = (
            config.reasoning_effort
            if config.reasoning_effort is not None
            else "none"
        )

        return CommandResult(
            "\n".join(
                (
                    f"host: {config.host}",
                    f"id: {config.id}",
                    "api_key: ********",
                    f"input_max_tokens: {config.max_input_tokens}",
                    f"output_max_tokens: {config.max_output_tokens}",
                    f"reasoning_effort: {reasoning}",
                    (
                        "retry.max_attempts: "
                        f"{config.retry.max_attempts}"
                    ),
                    (
                        "retry.request_timeout: "
                        f"{config.retry.request_timeout}"
                    ),
                    (
                        "retry.initial_backoff: "
                        f"{config.retry.initial_backoff}"
                    ),
                    (
                        "retry.max_backoff: "
                        f"{config.retry.max_backoff}"
                    ),
                )
            )
        )

    def _help(self) -> CommandResult:
        return CommandResult(
            self._usage()
        )

    @staticmethod
    def _positive_int(
        value: str,
        field: str,
    ) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise ValueError(
                f"{field} must be an integer."
            ) from error

        if parsed <= 0:
            raise ValueError(
                f"{field} must be greater than zero."
            )

        return parsed

    @staticmethod
    def _positive_float(
        value: str,
        field: str,
    ) -> float:
        try:
            parsed = float(value)
        except ValueError as error:
            raise ValueError(
                f"{field} must be a number."
            ) from error

        if parsed <= 0:
            raise ValueError(
                f"{field} must be greater than zero."
            )

        return parsed

    @staticmethod
    def _nonnegative_float(
        value: str,
        field: str,
    ) -> float:
        try:
            parsed = float(value)
        except ValueError as error:
            raise ValueError(
                f"{field} must be a number."
            ) from error

        if parsed < 0:
            raise ValueError(
                f"{field} cannot be negative."
            )

        return parsed

    @staticmethod
    def _optional_string(
        value: str,
    ) -> str | None:
        normalized = value.strip()

        if normalized.lower() in {
            "none",
            "null",
            "unset",
            "off",
        }:
            return None

        if not normalized:
            return None

        return normalized

    @staticmethod
    def _usage() -> str:
        return (
            "Usage:\n"
            "  model show\n"
            "  model set host <url>\n"
            "  model set id <model-id>\n"
            "  model set api_key <key>\n"
            "  model set max_input_tokens <integer>\n"
            "  model set max_output_tokens <integer>\n"
            "  model set reasoning_effort <value|none>\n"
            "  model set retry.max_attempts <integer>\n"
            "  model set retry.request_timeout <seconds>\n"
            "  model set retry.initial_backoff <seconds>\n"
            "  model set retry.max_backoff <seconds>"
        )