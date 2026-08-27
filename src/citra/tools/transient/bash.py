from __future__ import annotations

import re
from typing import Any, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..tool import Tool, ToolDefinition
from .prompt_user import PromptUser


def _bash_request_schema(
    *,
    command_name: str,
    cwd_name: str,
    timeout_name: str,
    timeout_milliseconds: bool,
    include_description: bool,
) -> JsonSchema:
    properties: list[JsonProperty] = [
        JsonProperty(
            name=command_name,
            schema=JsonSchema.string(
                description="Shell command to execute.",
            ),
        ),
        JsonProperty(
            name=cwd_name,
            schema=JsonSchema.string(
                description=(
                    "Working directory for the command. "
                    "Relative paths resolve from the active workspace."
                ),
            ),
            required=False,
        ),
        JsonProperty(
            name=timeout_name,
            schema=JsonSchema.integer(
                description=(
                    "Maximum execution time in milliseconds."
                    if timeout_milliseconds
                    else "Maximum execution time in seconds."
                ),
            ),
            required=False,
        ),
        JsonProperty(
            name="network",
            schema=JsonSchema.boolean(
                description=(
                    "Request network access for this command. "
                    "Defaults to false."
                ),
            ),
            required=False,
        ),
        JsonProperty(
            name="reason",
            schema=JsonSchema.string(
                description=(
                    "Required when network is true. Explain why "
                    "network access is needed."
                ),
            ),
            required=False,
        ),
    ]

    if include_description:
        properties.insert(
            1,
            JsonProperty(
                name="description",
                schema=JsonSchema.string(
                    description=(
                        "Brief description of what the command does."
                    ),
                ),
                required=False,
            ),
        )

    return JsonSchema.object(
        properties=tuple(properties),
        additional_properties=False,
    )


def _bash_definition(
    *,
    name: str,
    command_name: str,
    cwd_name: str,
    timeout_name: str,
    timeout_milliseconds: bool,
    include_description: bool,
    description: str,
) -> ChatCompletionTool:
    request_schema = _bash_request_schema(
        command_name=command_name,
        cwd_name=cwd_name,
        timeout_name=timeout_name,
        timeout_milliseconds=timeout_milliseconds,
        include_description=include_description,
    )

    properties: list[JsonProperty] = [
        JsonProperty(
            name=command_name,
            schema=JsonSchema.string(
                description=(
                    "Single shell command to execute. "
                    "Use 'requests' for multiple independent commands."
                ),
            ),
            required=False,
        ),
        JsonProperty(
            name=cwd_name,
            schema=JsonSchema.string(
                description=(
                    "Working directory for the single command. "
                    "Relative paths resolve from the active workspace."
                ),
            ),
            required=False,
        ),
        JsonProperty(
            name=timeout_name,
            schema=JsonSchema.integer(
                description=(
                    "Maximum execution time for the single command "
                    + (
                        "in milliseconds."
                        if timeout_milliseconds
                        else "in seconds."
                    )
                ),
            ),
            required=False,
        ),
        JsonProperty(
            name="network",
            schema=JsonSchema.boolean(
                description=(
                    "Request network access for the command. "
                    "Defaults to false."
                ),
            ),
            required=False,
        ),
        JsonProperty(
            name="reason",
            schema=JsonSchema.string(
                description=(
                    "Required when network is true. Explain why "
                    "network access is needed."
                ),
            ),
            required=False,
        ),
        JsonProperty(
            name="requests",
            schema=JsonSchema.array(
                request_schema,
                description=(
                    "Independent shell commands to execute as a batch. "
                    "Each request may specify its own working directory, "
                    "timeout, network flag, and reason."
                ),
            ),
            required=False,
        ),
    ]

    if include_description:
        properties.insert(
            1,
            JsonProperty(
                name="description",
                schema=JsonSchema.string(
                    description=(
                        "Brief description of what the command does. "
                        "Used only for model/tool compatibility."
                    ),
                ),
                required=False,
            ),
        )

    return ChatCompletionTool(
        function=FunctionDefinition(
            name=name,
            description=description,
            parameters=JsonSchema.object(
                properties=tuple(properties),
                additional_properties=False,
            ),
        ),
    )


class Bash(Tool):
    """
    Execute one or more foreground shell commands inside Citra's sandbox.
    """

    TOOL_ID = "bash"

    DEFAULT_TIMEOUT_SECONDS = 30
    MAX_BATCH_SIZE = 20

    # ------------------------------------------------------------------
    # Citra
    #
    # bash(
    #     cmd?,
    #     cwd?,
    #     timeout?,       # seconds
    #     network?,
    #     reason?,
    #     requests?,
    # )
    # ------------------------------------------------------------------

    CITRA_DEFINITION = _bash_definition(
        name="bash",
        command_name="cmd",
        cwd_name="cwd",
        timeout_name="timeout",
        timeout_milliseconds=False,
        include_description=False,
        description=(
            "Execute one or more Bash commands inside the local sandbox. "
            "For a single command use cmd. For multiple independent commands "
            "use requests. Prefer specialized tools when available."
        ),
    )

    # ------------------------------------------------------------------
    # Claude Code
    #
    # Bash(
    #     command?,
    #     timeout?,       # milliseconds
    #     description?,
    #
    #     # Citra extensions:
    #     cwd?,
    #     network?,
    #     reason?,
    #     requests?,
    # )
    #
    # run_in_background deliberately omitted.
    # ------------------------------------------------------------------

    CLAUDE_CODE_DEFINITION = _bash_definition(
        name="Bash",
        command_name="command",
        cwd_name="cwd",
        timeout_name="timeout",
        timeout_milliseconds=True,
        include_description=True,
        description=(
            "Execute shell commands in the sandbox. Prefer specialized "
            "tools over shell commands when possible. Commands run in the "
            "foreground; use Citra's subprocess tool for persistent or "
            "background processes."
        ),
    )

    # ------------------------------------------------------------------
    # Gemini CLI
    #
    # run_shell_command(
    #     command?,
    #     description?,
    #     dir_path?,
    #
    #     # Citra extensions:
    #     timeout_seconds?,
    #     network?,
    #     reason?,
    #     requests?,
    # )
    #
    # is_background deliberately omitted.
    # ------------------------------------------------------------------

    GEMINI_CLI_DEFINITION = _bash_definition(
        name="run_shell_command",
        command_name="command",
        cwd_name="dir_path",
        timeout_name="timeout_seconds",
        timeout_milliseconds=False,
        include_description=True,
        description=(
            "Execute shell commands in the workspace. Use this for command-line "
            "operations that are not better handled by a specialized tool. "
            "Commands run in the foreground."
        ),
    )

    # ------------------------------------------------------------------
    # Qwen Code
    #
    # run_shell_command(
    #     command?,
    #     description?,
    #     directory?,
    #
    #     # Citra extensions:
    #     timeout_seconds?,
    #     network?,
    #     reason?,
    #     requests?,
    # )
    #
    # Native Qwen currently requires is_background. We intentionally omit
    # it because Citra's Bash tool is foreground-only.
    # ------------------------------------------------------------------

    QWEN_CODE_DEFINITION = _bash_definition(
        name="run_shell_command",
        command_name="command",
        cwd_name="directory",
        timeout_name="timeout_seconds",
        timeout_milliseconds=False,
        include_description=True,
        description=(
            "Execute a shell command for command-line operations in the "
            "workspace. Commands run in the foreground. Use the subprocess "
            "tool when a persistent background process is required."
        ),
    )

    # ------------------------------------------------------------------
    # Kimi Code
    #
    # Bash(
    #     command?,
    #     cwd?,
    #     timeout?,       # milliseconds
    #     description?,
    #     ...
    # )
    #
    # run_in_background / disable_timeout deliberately omitted.
    # ------------------------------------------------------------------

    KIMI_CODE_DEFINITION = _bash_definition(
        name="Bash",
        command_name="command",
        cwd_name="cwd",
        timeout_name="timeout",
        timeout_milliseconds=True,
        include_description=True,
        description=(
            "Execute a shell command in the workspace. Commands run in the "
            "foreground. Use Citra's subprocess tool for persistent processes."
        ),
    )

    # ------------------------------------------------------------------
    # ZCode / GLM
    #
    # ZCode has a capitalized Bash tool and is strongly Claude-compatible
    # in its Bash environment/tool conventions. Keep the compatible shape.
    # ------------------------------------------------------------------

    ZCODE_DEFINITION = _bash_definition(
        name="Bash",
        command_name="command",
        cwd_name="cwd",
        timeout_name="timeout",
        timeout_milliseconds=True,
        include_description=True,
        description=(
            "Execute a shell command in the sandbox. Commands run in the "
            "foreground; use the subprocess tool for persistent processes."
        ),
    )

    # ------------------------------------------------------------------
    # Codex
    #
    # Current unified execution:
    #
    # exec_command(
    #     cmd,
    #     workdir?,
    #     ...
    # )
    #
    # Codex's yield/process-continuation parameters are deliberately not
    # reproduced because this Tool is synchronous. Citra extensions remain
    # available.
    # ------------------------------------------------------------------

    CODEX_DEFINITION = _bash_definition(
        name="exec_command",
        command_name="cmd",
        cwd_name="workdir",
        timeout_name="timeout_seconds",
        timeout_milliseconds=False,
        include_description=False,
        description=(
            "Execute a shell command in the workspace and return its output. "
            "Commands run synchronously in the foreground."
        ),
    )

    # ------------------------------------------------------------------
    # OpenCode reference profile
    # ------------------------------------------------------------------

    OPENCODE_DEFINITION = _bash_definition(
        name="bash",
        command_name="command",
        cwd_name="workdir",
        timeout_name="timeout",
        timeout_milliseconds=True,
        include_description=False,
        description=(
            "Execute one shell command string in the working directory."
        ),
    )

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        del context

        return (
            ToolDefinition(
                definition=cls.CLAUDE_CODE_DEFINITION,
                model_family_matchers=("claude",),
            ),
            ToolDefinition(
                definition=cls.GEMINI_CLI_DEFINITION,
                model_family_matchers=("gemini",),
            ),
            ToolDefinition(
                definition=cls.QWEN_CODE_DEFINITION,
                model_family_matchers=("qwen",),
            ),
            ToolDefinition(
                definition=cls.KIMI_CODE_DEFINITION,
                model_family_matchers=(
                    "kimi",
                    "moonshot",
                ),
            ),
            ToolDefinition(
                definition=cls.ZCODE_DEFINITION,
                model_family_matchers=("glm",),
            ),
            ToolDefinition(
                definition=cls.CODEX_DEFINITION,
                model_family_matchers=(
                    "gpt",
                    "codex",
                ),
            ),
            ToolDefinition(
                definition=cls.CITRA_DEFINITION,
            ),
        )

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        super().__init__(
            context=context,
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _first(
        arguments: dict[str, Any],
        *names: str,
    ) -> Any:
        for name in names:
            if name in arguments:
                return arguments[name]

        return None

    def _normalize_request(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        command = self._first(
            arguments,
            "cmd",
            "command",
        )

        cwd = self._first(
            arguments,
            "cwd",
            "dir_path",
            "directory",
            "workdir",
        )

        normalized: dict[str, Any] = {}

        if command is not None:
            normalized["cmd"] = command

        if cwd is not None:
            normalized["cwd"] = cwd

        # Explicit Citra-style seconds field used on harnesses whose
        # native shell tool has no compatible per-call timeout field.
        if "timeout_seconds" in arguments:
            normalized["timeout"] = int(
                arguments["timeout_seconds"]
            )

        elif "timeout" in arguments:
            timeout = int(
                arguments["timeout"]
            )

            # Claude, Kimi, ZCode and OpenCode train on Bash/bash timeout
            # values expressed in milliseconds. Citra internally uses
            # whole seconds.
            timeout_is_milliseconds = (
                "command" in arguments
                and self.model_name in {
                    "Bash",
                    "bash",
                }
            )

            if timeout_is_milliseconds:
                # Round up rather than accidentally shortening the model's
                # requested deadline.
                timeout = max(
                    1,
                    (timeout + 999) // 1000,
                )

            normalized["timeout"] = timeout

        if "network" in arguments:
            normalized["network"] = bool(
                arguments["network"]
            )

        if "reason" in arguments:
            normalized["reason"] = arguments["reason"]

        # `description` is deliberately accepted by native-compatible
        # schemas but has no execution semantics in Citra.
        #
        # is_background/run_in_background are not exposed at all.

        return normalized

    def _normalize_arguments(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        requests = arguments.get(
            "requests"
        )

        if requests is None:
            return self._normalize_request(
                arguments
            )

        if not isinstance(
            requests,
            list,
        ):
            raise ValueError(
                "'requests' must be an array."
            )

        normalized = self._normalize_request(
            arguments
        )

        normalized["requests"] = [
            self._normalize_request(
                request
            )
            for request in requests
        ]

        return normalized

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        arguments = self._normalize_arguments(
            arguments
        )
        if not self.context.has_command(
            "bash"
        ):
            raise RuntimeError(
                "Bash is not available in the current execution context."
            )

        cmd = arguments.get(
            "cmd"
        )
        requests = arguments.get(
            "requests"
        )

        if (
            cmd is not None
            and requests is not None
        ):
            raise ValueError(
                "Use either 'cmd' or 'requests', not both."
            )

        if cmd is not None:
            return self._run(
                cmd=cmd,
                cwd=arguments.get(
                    "cwd"
                ),
                timeout=arguments.get(
                    "timeout",
                    self.DEFAULT_TIMEOUT_SECONDS,
                ),
                network=bool(arguments.get("network", False)),
                reason=arguments.get("reason"),
            )

        if not requests:
            raise ValueError(
                "'cmd' or 'requests' is required."
            )

        if (
            arguments.get("cwd") is not None
            or arguments.get("timeout") is not None
            or arguments.get("network") is not None
            or arguments.get("reason") is not None
        ):
            raise ValueError(
                "'cwd', 'timeout', 'network', and 'reason' are only valid "
                "with single-command 'cmd'. Batch requests specify their own."
            )

        if len(requests) > self.MAX_BATCH_SIZE:
            raise ValueError(
                f"At most {self.MAX_BATCH_SIZE} commands may be "
                "executed in one batch."
            )

        results: list[str] = []

        for index, request in enumerate(
            requests,
            1,
        ):
            cmd = request["cmd"]

            try:
                output = self._run(
                    cmd=cmd,
                    cwd=request.get(
                        "cwd"
                    ),
                    timeout=request.get(
                        "timeout",
                        self.DEFAULT_TIMEOUT_SECONDS,
                    ),
                    network=bool(request.get("network", False)),
                    reason=request.get("reason"),
                )
            except Exception as error:
                output = (
                    f"error: {error}"
                )

            results.append(
                f"===== command {index} =====\n"
                f"$ {cmd}\n"
                f"{output}"
            )

        return "\n\n".join(
            results
        )

    def _run(
        self,
        *,
        cmd: str,
        cwd: str | None,
        timeout: int,
        network: bool,
        reason: str | None,
    ) -> str:
        if not cmd.strip():
            raise ValueError(
                "'cmd' cannot be empty."
            )

        if timeout <= 0:
            raise ValueError(
                "'timeout' must be greater than zero."
            )

        cleaned_reason = "" if reason is None else reason.strip()
        if network and not cleaned_reason:
            raise ValueError(
                "'reason' is required when Bash requests network access."
            )
        if not network and reason is not None:
            raise ValueError(
                "'reason' is only valid when 'network' is true."
            )

        working_directory = (
            self.context.workspace.workspace
            if cwd is None
            else self.context.workspace.resolve_path(
                cwd
            )
        )

        if not working_directory.is_dir():
            raise NotADirectoryError(
                "Working directory does not exist: "
                f"{self.context.workspace.display_path(working_directory)}"
            )

        if network and not self.context.config.bash.always_allow_network:
            shown_cwd = self.context.workspace.display_path(working_directory)
            shown_command = self._safe_terminal_text(cmd)
            shown_reason = self._safe_terminal_text(cleaned_reason)
            permission = PromptUser(self.context)._execute(
                {
                    "question": (
                        "Allow this Bash command to access the network?\n\n"
                        f"Command:\n{shown_command}\n\n"
                        f"Working directory: {shown_cwd}\n"
                        f"Reason: {shown_reason}"
                    ),
                    "options": ["Allow once", "Deny"],
                    "timeout": self.context.config.bash.permission_timeout,
                }
            )
            if permission != "Allow once":
                return (
                    "permission-denied: Bash network access was not granted; "
                    "the command was not executed."
                )

        result = self.context.sandbox.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                cmd,
            ],
            cwd=working_directory,
            timeout=timeout,
            network=network,
        )

        output = result.output.strip()

        if result.timed_out:
            marker = (
                f"(timed out after {timeout}s)"
            )

            if output:
                return (
                    f"{output}\n{marker}"
                )

            return marker

        if result.returncode != 0:
            marker = (
                f"(exit code {result.returncode})"
            )

            if output:
                return (
                    f"{output}\n{marker}"
                )

            return marker

        return output or "(empty)"

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        cmd = arguments.get("cmd")
        requests = arguments.get("requests")

        if cmd is not None:
            cwd = arguments.get("cwd")
            timeout = arguments.get(
                "timeout",
                self.DEFAULT_TIMEOUT_SECONDS,
            )
            network = bool(
                arguments.get(
                    "network",
                    False,
                )
            )

            parts = [
                f"$ {self._truncate_command(cmd)}",
            ]

            if cwd is not None:
                parts.append(
                    f"cwd={cwd}"
                )

            if timeout != self.DEFAULT_TIMEOUT_SECONDS:
                parts.append(
                    f"timeout={timeout}s"
                )

            if network:
                parts.append(
                    "network=true"
                )

            return " | ".join(parts)

        if requests:
            commands = [
                self._truncate_command(
                    str(request.get("cmd", ""))
                )
                for request in requests
            ]

            preview_limit = 3

            preview = "; ".join(
                f"$ {command}"
                for command in commands[:preview_limit]
            )

            remaining = (
                len(commands) - preview_limit
            )

            if remaining > 0:
                preview += (
                    f"; +{remaining} more"
                )

            return (
                f"batch={len(requests)} | "
                f"{preview}"
            )

        return "no command"


    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        text = str(result)

        if not text:
            return "empty output"

        lines = text.splitlines()
        chars = len(text)

        timed_out = (
            "(timed out after "
            in text
        )

        exit_codes = re.findall(
            r"\(exit code (\d+)\)",
            text,
        )

        batch_count = text.count(
            "===== command "
        )

        parts: list[str] = []

        if batch_count:
            parts.append(
                f"{batch_count} commands"
            )

        parts.append(
            f"{len(lines)} lines"
        )

        parts.append(
            f"{chars} chars"
        )

        if timed_out:
            parts.append(
                "timed-out"
            )

        if exit_codes:
            unique_codes = sorted(
                set(exit_codes)
            )

            parts.append(
                "exit="
                + ",".join(unique_codes)
            )

        return " | ".join(parts)


    @staticmethod
    def _truncate_command(
        command: str,
        limit: int = 200,
    ) -> str:
        command = (
            command
            .replace("\n", " ")
            .strip()
        )

        if len(command) <= limit:
            return command

        return (
            command[:limit]
            + "..."
        )

    @staticmethod
    def _safe_terminal_text(value: str) -> str:
        """Render model text without allowing terminal control sequences."""
        return value.encode("unicode_escape").decode("ascii")
