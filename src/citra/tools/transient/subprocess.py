"""Lifecycle-scoped background subprocess tool."""

from __future__ import annotations

import time
import json
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

class Subprocess(Tool):
    ACTIONS = frozenset({"start", "poll", "write", "stop", "list"})
    MAX_SLEEP_AFTER_SECONDS = 60

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="subprocess",
            description=(
                "Manage lifecycle-scoped sandboxed background processes. "
                "Actions: start, poll, write, stop, and list. Processes survive "
                "agent turns and are terminated when Citra exits. A start may "
                "optionally wait with 'sleep_after' and then inspect currently "
                "buffered process output with 'poll_after'. Network is off by "
                "default; networked starts require a reason and user approval "
                "unless globally allowed."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(name="action", schema=JsonSchema.string()),
                    JsonProperty(name="cmd", schema=JsonSchema.string(), required=False),
                    JsonProperty(name="cwd", schema=JsonSchema.string(), required=False),
                    JsonProperty(name="process_id", schema=JsonSchema.integer(), required=False),
                    JsonProperty(name="input", schema=JsonSchema.string(), required=False),
                    JsonProperty(name="clear", schema=JsonSchema.boolean(), required=False),
                    JsonProperty(name="network", schema=JsonSchema.boolean(), required=False),
                    JsonProperty(
                        name="reason",
                        schema=JsonSchema.string(
                            description="Required for a networked start."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="sleep_after",
                        schema=JsonSchema.integer(
                            description=(
                                "Seconds to wait after starting the subprocess before returning "
                                "or performing 'poll_after'. Useful for servers and other "
                                "processes that need time to initialize. Only valid for action "
                                "'start'. Maximum 60 seconds."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="poll_after",
                        schema=JsonSchema.boolean(
                            description=(
                                "After starting and optionally waiting with 'sleep_after', "
                                "poll the subprocess once and return its currently buffered "
                                "output and status. Only valid for action 'start'. "
                                "Use 'clear' to control whether returned buffered output is "
                                "consumed."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        )
    )

    def __init__(self, context: ExecutionContext) -> None:
        super().__init__(context=context, definition=self.DEFINITION)

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        action = str(arguments["action"])

        if action not in self.ACTIONS:
            raise ValueError(
                f"Unsupported subprocess action: {action}"
            )

        if action != "start":
            if "sleep_after" in arguments:
                raise ValueError(
                    "'sleep_after' is only valid for action 'start'."
                )

            if "poll_after" in arguments:
                raise ValueError(
                    "'poll_after' is only valid for action 'start'."
                )

        if action == "list":
            return json.dumps(
                self.context.subprocesses.list(),
                indent=2,
            )

        if action == "start":
            return self._start(
                arguments
            )

        process_id = arguments.get(
            "process_id"
        )

        if process_id is None:
            raise ValueError(
                f"'process_id' is required for action '{action}'."
            )

        process_id = int(
            process_id
        )

        if action == "poll":
            result = self.context.subprocesses.poll(
                process_id,
                clear=bool(
                    arguments.get("clear", True)
                ),
            )

            self._truncate_output(
                result
            )

            return json.dumps(
                result,
                indent=2,
            )

        if action == "write":
            if "input" not in arguments:
                raise ValueError(
                    "'input' is required for action 'write'."
                )

            self.context.subprocesses.write(
                process_id,
                str(arguments["input"]),
            )

            return "ok"

        return json.dumps(
            self.context.subprocesses.stop(
                process_id
            ),
            indent=2,
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        action = str(arguments.get("action", "?"))
        parts = [f"action={action}"]

        if action == "start":
            cmd = str(arguments.get("cmd", ""))
            parts.append(f"$ {self._truncate_cmd(cmd)}")

            if arguments.get("network"):
                parts.append("network=true")

            sleep_after = arguments.get("sleep_after")
            if sleep_after:
                parts.append(f"sleep={sleep_after}s")

            if arguments.get("poll_after"):
                parts.append("poll_after=true")

            return " | ".join(parts)

        process_id = arguments.get("process_id")
        if process_id is not None:
            parts.append(f"pid={process_id}")

        if action == "write":
            input_text = str(arguments.get("input", ""))
            parts.append(f"input={len(input_text)} chars")

        return " | ".join(parts)

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        text = str(result)

        if text == "ok":
            return "ok"

        if text.startswith("permission-denied"):
            return "permission-denied"

        if text.startswith("Started subprocess"):
            return text

        lines = text.splitlines()
        return f"{len(lines)} lines"

    @staticmethod
    def _truncate_cmd(cmd: str) -> str:
        cmd = cmd.replace("\n", " ").strip()
        if len(cmd) <= 200:
            return cmd
        return cmd[:200] + "..."

    def _truncate_output(
        self,
        result: dict[str, Any],
    ) -> None:
        output = str(
            result.get("output", "")
        )

        limit = (
            self.context.config.subprocess.max_output_length
        )

        if len(output) <= limit:
            return

        omitted = len(output) - limit

        result["output"] = (
            output[:limit]
            + f"\n... <truncated {omitted} characters>"
        )
        
    def _start(self, arguments: dict[str, Any]) -> str:
        cmd = str(
            arguments.get("cmd", "")
        )

        if not cmd.strip():
            raise ValueError(
                "'cmd' is required for action 'start'."
            )

        network = bool(
            arguments.get("network", False)
        )

        reason = str(
            arguments.get("reason", "")
        ).strip()

        sleep_after = int(
            arguments.get("sleep_after", 0)
        )

        poll_after = bool(
            arguments.get("poll_after", False)
        )

        if not 0 <= sleep_after <= self.MAX_SLEEP_AFTER_SECONDS:
            raise ValueError(
                f"'sleep_after' must be between 0 and "
                f"{self.MAX_SLEEP_AFTER_SECONDS} seconds."
            )

        if "clear" in arguments and not poll_after:
            raise ValueError(
                "'clear' on action 'start' requires 'poll_after=true'."
            )

        if network and not reason:
            raise ValueError(
                "'reason' is required for a networked subprocess."
            )

        if not network and "reason" in arguments:
            raise ValueError(
                "'reason' is only valid when 'network' is true."
            )

        workspace = self.context.workspace

        cwd = (
            workspace.workspace
            if arguments.get("cwd") is None
            else workspace.resolve_path(
                str(arguments["cwd"])
            )
        )

        if not cwd.is_dir():
            raise NotADirectoryError(
                "Subprocess working directory does not exist: "
                f"{workspace.display_path(cwd)}"
            )

        config = self.context.config.subprocess

        if network and not config.always_allow_network:
            permission = PromptUser(self.context)._execute(
                {
                    "question": (
                        "Allow this persistent subprocess to access the network?\n\n"
                        f"Command:\n{self._safe(cmd)}\n\n"
                        f"Working directory: {workspace.display_path(cwd)}\n"
                        f"Reason: {self._safe(reason)}"
                    ),
                    "options": [
                        "Allow once",
                        "Deny",
                    ],
                    "timeout": config.permission_timeout,
                }
            )

            if permission != "Allow once":
                return (
                    "permission-denied: subprocess was not started."
                )

        process_id = self.context.subprocesses.start(
            cmd,
            cwd=cwd,
            network=network,
        )

        if sleep_after:
            time.sleep(
                sleep_after
            )

        if not poll_after:
            return f"Started subprocess {process_id}."

        result = self.context.subprocesses.poll(
            process_id,
            clear=bool(
                arguments.get("clear", False)
            ),
        )

        self._truncate_output(
            result
        )

        # Ensure the ID is available even if the subprocess manager's
        # poll result does not include it.
        result["process_id"] = process_id

        return json.dumps(
            result,
            indent=2,
        )

    @staticmethod
    def _safe(value: str) -> str:
        return value.encode("unicode_escape").decode("ascii")
