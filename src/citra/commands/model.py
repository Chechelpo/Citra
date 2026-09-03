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
        """Execute the run operation."""
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
        """Handle show."""
        if len(args) > 1:
            return CommandResult(
                "Expected zero or one profile name.\n\n" + self._usage()
            )

        name = args[0] if args else None
        store = self.context.config.model_config_store
        try:
            config = store.get(name)
            orchestrator = store.orchestrator_name()
            subagent = store.subagent_name()
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            return self._error("read model config", error)

        roles: list[str] = []
        if config.name == orchestrator:
            roles.append("orchestrator")
        if config.name == subagent:
            if config.name == orchestrator:
                roles.append("subagent (inherits orchestrator)")
            else:
                roles.append("subagent")
        role_text = ", ".join(roles) if roles else "no"

        reasoning = config.reasoning_effort or "none"
        return CommandResult(
            "\n".join(
                (
                    f"profile: {config.name}",
                    f"orchestrator: {'yes' if config.name == orchestrator else 'no'}",
                    f"subagent: {'yes' if config.name == subagent else 'no'}",
                    f"roles: {role_text}",
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
        """Handle list."""
        if args:
            return CommandResult("model list takes no arguments.\n\n" + self._usage())

        store = self.context.config.model_config_store
        try:
            orchestrator = store.orchestrator_name()
            subagent = store.subagent_name()
            subagent_shares = subagent == orchestrator
            lines = []
            for name in store.names():
                config = store.get(name)
                markers: list[str] = []
                if name == orchestrator:
                    markers.append("*")
                else:
                    markers.append(" ")
                if name == subagent:
                    if subagent_shares:
                        markers[-1] = "*"
                    else:
                        markers.append("S")
                else:
                    markers.append(" ")
                lines.append(
                    f"{''.join(markers)} {name}: {config.id}"
                )
            if subagent_shares:
                lines.append("(* = orchestrator, subagent inherits)")
            else:
                lines.append(
                    f"(* = orchestrator [{orchestrator}], "
                    f"S = subagent [{subagent}])"
                )
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            return self._error("list model profiles", error)

        return CommandResult("\n".join(lines))

    def _use(self, args: list[str]) -> CommandResult:
        """Handle use."""
        if not args:
            return CommandResult("Expected a role and profile name.\n\n" + self._usage())

        store = self.context.config.model_config_store
        first = args[0].lower()
        if first in {"orchestrator", "subagent"}:
            if len(args) != 2:
                return CommandResult(
                    f"Expected 'model use {first} <profile>'.\n\n" + self._usage()
                )
            profile_name = args[1]
            try:
                if first == "orchestrator":
                    store.set_orchestrator(profile_name)
                else:
                    store.set_subagent(profile_name)
                config = store.get(profile_name)
            except (KeyError, ValueError, RuntimeError, OSError) as error:
                return self._error(f"set {first} model profile", error)
            return CommandResult(
                f"{first.capitalize()} model profile = {config.name} ({config.id})"
            )

        if len(args) != 1:
            return CommandResult("Expected a profile name.\n\n" + self._usage())

        profile_name = args[0]
        try:
            store.set_orchestrator(profile_name)
            config = store.get(profile_name)
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            return self._error("activate model profile", error)

        return CommandResult(
            f"Orchestrator model profile = {config.name} ({config.id})"
        )

    def _add(self, args: list[str]) -> CommandResult:
        """Handle add."""
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
        """Handle delete."""
        if len(args) != 1:
            return CommandResult("Expected a profile name.\n\n" + self._usage())

        store = self.context.config.model_config_store
        try:
            store.delete(args[0])
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            return self._error("delete model profile", error)

        return CommandResult(f"Deleted model profile {args[0]}.")

    def _set(self, args: list[str]) -> CommandResult:
        """Handle set."""
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
        """Handle help."""
        return CommandResult(self._usage())

    @staticmethod
    def _error(operation: str, error: Exception) -> CommandResult:
        """Handle error."""
        return CommandResult(f"Unable to {operation}: {error}")

    @staticmethod
    def _positive_int(value: str, field: str) -> int:
        """Handle positive int."""
        try:
            parsed = int(value)
        except ValueError as error:
            raise ValueError(f"{field} must be an integer.") from error
        if parsed <= 0:
            raise ValueError(f"{field} must be greater than zero.")
        return parsed

    @staticmethod
    def _positive_float(value: str, field: str) -> float:
        """Handle positive float."""
        try:
            parsed = float(value)
        except ValueError as error:
            raise ValueError(f"{field} must be a number.") from error
        if parsed <= 0:
            raise ValueError(f"{field} must be greater than zero.")
        return parsed

    @staticmethod
    def _nonnegative_float(value: str, field: str) -> float:
        """Handle nonnegative float."""
        try:
            parsed = float(value)
        except ValueError as error:
            raise ValueError(f"{field} must be a number.") from error
        if parsed < 0:
            raise ValueError(f"{field} cannot be negative.")
        return parsed

    @staticmethod
    def _optional_string(value: str) -> str | None:
        """Handle optional string."""
        normalized = value.strip()
        if normalized.lower() in {"none", "null", "unset", "off"}:
            return None
        return normalized or None

    @staticmethod
    def _usage() -> str:
        """Handle usage."""
        return (
            "Usage:\n"
            "  model show [profile]\n"
            "  model list\n"
            "  model use <profile>            (sets the orchestrator profile)\n"
            "  model use orchestrator <profile>\n"
            "  model use subagent <profile>    (omit to inherit the orchestrator)\n"
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
