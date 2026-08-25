from __future__ import annotations

import shlex
from typing import override

from citra.commands import CommandResult
from .command import Command


class ModelCommand(Command):
    """Inspect and modify Citra's named model profiles."""

    id = "model"
    description = "inspect and switch model profiles"

    _TOP_LEVEL_FIELDS = {
        "host",
        "id",
        "api_key",
        "max_input_tokens",
        "max_output_tokens",
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
            return CommandResult(f"Invalid arguments: {error}")

        if not parts:
            return self._show([])

        action = parts[0].lower()
        rest = parts[1:]

        if action in {"show", "get", "status"}:
            return self._show(rest)
        if action == "list":
            return self._list(rest)
        if action in {"use", "activate"}:
            return self._use(rest)
        if action in {"add", "create"}:
            return self._add(rest)
        if action in {"delete", "remove", "rm"}:
            return self._delete(rest)
        if action == "set":
            return self._set(rest)
        if action in {"help", "-h", "--help"}:
            return self._help()

        return CommandResult(
            f"Unknown model command: {action}\n\n{self._usage()}"
        )

    def _show(self, args: list[str]) -> CommandResult:
        if len(args) > 1:
            return CommandResult(
                "Expected zero or one profile name.\n\n" + self._usage()
            )

        name = args[0] if args else None
        store = self.context.config.model_config_store
        try:
            config = store.get(name)
            active = store.active_name()
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            return self._error("read model config", error)

        reasoning = config.reasoning_effort or "none"
        return CommandResult(
            "\n".join(
                (
                    f"profile: {config.name}",
                    f"active: {'yes' if config.name == active else 'no'}",
                    f"host: {config.host}",
                    f"id: {config.id}",
                    "api_key: ********",
                    f"max_input_tokens: {config.max_input_tokens}",
                    f"max_output_tokens: {config.max_output_tokens}",
                    f"reasoning_effort: {reasoning}",
                    f"retry.max_attempts: {config.retry.max_attempts}",
                    f"retry.request_timeout: {config.retry.request_timeout}",
                    f"retry.initial_backoff: {config.retry.initial_backoff}",
                    f"retry.max_backoff: {config.retry.max_backoff}",
                )
            )
        )

    def _list(self, args: list[str]) -> CommandResult:
        if args:
            return CommandResult("model list takes no arguments.\n\n" + self._usage())

        store = self.context.config.model_config_store
        try:
            active = store.active_name()
            lines = []
            for name in store.names():
                config = store.get(name)
                marker = "*" if name == active else " "
                lines.append(f"{marker} {name}: {config.id}")
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            return self._error("list model profiles", error)

        return CommandResult("\n".join(lines))

    def _use(self, args: list[str]) -> CommandResult:
        if len(args) != 1:
            return CommandResult("Expected a profile name.\n\n" + self._usage())

        store = self.context.config.model_config_store
        name = args[0]
        try:
            store.set_active(name)
            config = store.get(name)
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            return self._error("activate model profile", error)

        return CommandResult(
            f"Active model profile = {config.name} ({config.id})"
        )

    def _add(self, args: list[str]) -> CommandResult:
        if not args:
            return CommandResult("Expected a new profile name.\n\n" + self._usage())

        name = args[0]
        copy_from: str | None = None
        remaining = args[1:]
        if remaining:
            if len(remaining) != 2 or remaining[0] != "--copy":
                return CommandResult(
                    "Expected '--copy <profile>' after the new profile name.\n\n"
                    + self._usage()
                )
            copy_from = remaining[1]

        store = self.context.config.model_config_store
        try:
            source = copy_from or store.active_name()
            store.add(name, copy_from=copy_from)
            config = store.get(name)
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            return self._error("add model profile", error)

        return CommandResult(
            f"Added model profile {config.name} from {source}."
        )

    def _delete(self, args: list[str]) -> CommandResult:
        if len(args) != 1:
            return CommandResult("Expected a profile name.\n\n" + self._usage())

        store = self.context.config.model_config_store
        try:
            store.delete(args[0])
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            return self._error("delete model profile", error)

        return CommandResult(f"Deleted model profile {args[0]}.")

    def _set(self, args: list[str]) -> CommandResult:
        profile: str | None = None
        if args[:1] == ["--profile"]:
            if len(args) < 4:
                return CommandResult(
                    "Expected '--profile <name> <setting> <value>'.\n\n"
                    + self._usage()
                )
            profile = args[1]
            args = args[2:]

        if len(args) < 2:
            return CommandResult(
                "Expected a setting and value.\n\n" + self._usage()
            )

        field = args[0].strip().lower().replace("-", "_")
        value = " ".join(args[1:])
        all_fields = self._TOP_LEVEL_FIELDS | self._RETRY_FIELDS

        if field not in all_fields:
            return CommandResult(
                f"Unknown model setting: {field}\n\nAvailable settings:\n"
                + "\n".join(f"  {name}" for name in sorted(all_fields))
            )

        store = self.context.config.model_config_store
        try:
            target = profile or store.active_name()

            if field == "api_key":
                store.set_api_key(value, name=profile)
                return CommandResult(f"Model API key updated for {target}.")

            if field == "host":
                store.set_host(value, name=profile)
                rendered = value
            elif field == "id":
                store.set_model_id(value, name=profile)
                rendered = value
            elif field == "max_input_tokens":
                rendered = self._positive_int(value, field)
                store.set(name=profile, max_input_tokens=rendered)
            elif field == "max_output_tokens":
                rendered = self._positive_int(value, field)
                store.set(name=profile, max_output_tokens=rendered)
            elif field == "reasoning_effort":
                parsed = self._optional_string(value)
                store.set(name=profile, reasoning_effort=parsed)
                rendered = parsed if parsed is not None else "none"
            elif field == "retry.max_attempts":
                rendered = self._positive_int(value, field)
                store.set_retry(name=profile, max_attempts=rendered)
            elif field == "retry.request_timeout":
                rendered = self._positive_float(value, field)
                store.set_retry(name=profile, request_timeout=rendered)
            elif field == "retry.initial_backoff":
                rendered = self._nonnegative_float(value, field)
                store.set_retry(name=profile, initial_backoff=rendered)
            elif field == "retry.max_backoff":
                rendered = self._nonnegative_float(value, field)
                store.set_retry(name=profile, max_backoff=rendered)
            else:  # pragma: no cover - guarded by all_fields above
                return CommandResult(f"Unsupported model setting: {field}")
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            return self._error("update model config", error)

        return CommandResult(f"models.{target}.{field} = {rendered}")

    def _help(self) -> CommandResult:
        return CommandResult(self._usage())

    @staticmethod
    def _error(operation: str, error: Exception) -> CommandResult:
        return CommandResult(f"Unable to {operation}: {error}")

    @staticmethod
    def _positive_int(value: str, field: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise ValueError(f"{field} must be an integer.") from error
        if parsed <= 0:
            raise ValueError(f"{field} must be greater than zero.")
        return parsed

    @staticmethod
    def _positive_float(value: str, field: str) -> float:
        try:
            parsed = float(value)
        except ValueError as error:
            raise ValueError(f"{field} must be a number.") from error
        if parsed <= 0:
            raise ValueError(f"{field} must be greater than zero.")
        return parsed

    @staticmethod
    def _nonnegative_float(value: str, field: str) -> float:
        try:
            parsed = float(value)
        except ValueError as error:
            raise ValueError(f"{field} must be a number.") from error
        if parsed < 0:
            raise ValueError(f"{field} cannot be negative.")
        return parsed

    @staticmethod
    def _optional_string(value: str) -> str | None:
        normalized = value.strip()
        if normalized.lower() in {"none", "null", "unset", "off"}:
            return None
        return normalized or None

    @staticmethod
    def _usage() -> str:
        return (
            "Usage:\n"
            "  model show [profile]\n"
            "  model list\n"
            "  model use <profile>\n"
            "  model add <profile> [--copy <profile>]\n"
            "  model delete <profile>\n"
            "  model set [--profile <profile>] host <url>\n"
            "  model set [--profile <profile>] id <model-id>\n"
            "  model set [--profile <profile>] api_key <key>\n"
            "  model set [--profile <profile>] max_input_tokens <integer>\n"
            "  model set [--profile <profile>] max_output_tokens <integer>\n"
            "  model set [--profile <profile>] reasoning_effort <value|none>\n"
            "  model set [--profile <profile>] retry.max_attempts <integer>\n"
            "  model set [--profile <profile>] retry.request_timeout <seconds>\n"
            "  model set [--profile <profile>] retry.initial_backoff <seconds>\n"
            "  model set [--profile <profile>] retry.max_backoff <seconds>"
        )
