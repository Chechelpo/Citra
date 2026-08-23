from typing import Any, override

from ...context import ExecutionContext
from ..tool import Tool
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from .prompt_user import PromptUser
import re

class Bash(Tool):
    """
    Executes one or more Bash commands inside the workspace sandbox.

    The sandbox controls filesystem access, environment isolation,
    process isolation, and network availability.
    """

    DEFAULT_TIMEOUT_SECONDS = 30
    MAX_BATCH_SIZE = 20

    REQUEST_SCHEMA = JsonSchema.object(
        properties=(
            JsonProperty(
                name="cmd",
                schema=JsonSchema.string(
                    description="Bash command to execute.",
                ),
            ),
            JsonProperty(
                name="cwd",
                schema=JsonSchema.string(
                    description=(
                        "Working directory for this command. "
                        "Relative paths resolve from the active workspace. "
                        "Filesystem aliases such as @tmp are supported."
                    ),
                ),
                required=False,
            ),
            JsonProperty(
                name="timeout",
                schema=JsonSchema.integer(
                    description=(
                        "Maximum execution time in seconds for this command. "
                        "Defaults to 30 seconds."
                    ),
                ),
                required=False,
            ),
            JsonProperty(
                name="network",
                schema=JsonSchema.boolean(
                    description=(
                        "Request network access for this command. Defaults "
                        "to false. Unless globally allowed, the exact command "
                        "and reason are shown to the user for approval."
                    ),
                ),
                required=False,
            ),
            JsonProperty(
                name="reason",
                schema=JsonSchema.string(
                    description=(
                        "Required when network is true. Explain why this "
                        "command needs network access."
                    ),
                ),
                required=False,
            ),
        ),
        additional_properties=False,
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="bash",
            description=(
                "Execute one or more Bash commands inside the local sandbox. "
                "For a single command use cmd, with optional cwd and timeout. "
                "For multiple independent commands use requests. Batch commands "
                "are best-effort: a failed command does not prevent later "
                "commands from running. Network access is disabled by "
                "default; a command may request it with network=true and a "
                "required reason. Unless globally allowed, Citra shows the "
                "exact command and asks the user for permission. "
                "The active workspace and lifecycle agent filesystem are "
                "writable; the rest of the host filesystem is read-only. "
                "Filesystem aliases such as @tmp are supported in cwd. "
                "Inside Bash commands, use environment variables such as "
                "$CITRA_WORKSPACE, $CITRA_TMP, and $CITRA_CACHE."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="cmd",
                        schema=JsonSchema.string(
                            description=(
                                "Single Bash command to execute. "
                                "Use 'requests' for multiple commands."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="cwd",
                        schema=JsonSchema.string(
                            description=(
                                "Working directory for a single command. "
                                "Relative paths resolve from the active "
                                "workspace. Defaults to the workspace."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="timeout",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum execution time for a single command "
                                "in seconds. Defaults to 30 seconds."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="network",
                        schema=JsonSchema.boolean(
                            description=(
                                "Request network access for the single command. "
                                "Defaults to false."
                            )
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="reason",
                        schema=JsonSchema.string(
                            description=(
                                "Required when network is true. Explain why "
                                "the command requires network access."
                            )
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="requests",
                        schema=JsonSchema.array(
                            REQUEST_SCHEMA,
                            description=(
                                "Independent Bash commands to execute as a "
                                "batch. Each request may specify its own cwd, "
                                "timeout, network flag, and reason. At most "
                                "20 commands may be run per batch."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        super().__init__(
            context=context,
            definition=self.DEFINITION,
        )

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
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
