from typing import Any, override

from ...context import ExecutionContext
from ..tool import Tool
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)


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
                "commands from running. Commands have no network access. "
                "The active workspace and temporary agent filesystem are "
                "writable; the rest of the host filesystem is read-only. "
                "Use $CITRA_TMP or @tmp for disposable exploration."
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
                        name="requests",
                        schema=JsonSchema.array(
                            REQUEST_SCHEMA,
                            description=(
                                "Independent Bash commands to execute as a "
                                "batch. Each request may specify its own cwd "
                                "and timeout. At most 20 commands may be run "
                                "per batch."
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
            )

        if not requests:
            raise ValueError(
                "'cmd' or 'requests' is required."
            )

        if (
            arguments.get("cwd") is not None
            or arguments.get("timeout") is not None
        ):
            raise ValueError(
                "'cwd' and 'timeout' are only valid with single-command "
                "'cmd'. Batch requests specify their own cwd and timeout."
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
    ) -> str:
        if not cmd.strip():
            raise ValueError(
                "'cmd' cannot be empty."
            )

        if timeout <= 0:
            raise ValueError(
                "'timeout' must be greater than zero."
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
            network=False,
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