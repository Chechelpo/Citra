"""Lifecycle-scoped background subprocess tool."""

from __future__ import annotations

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

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="subprocess",
            description=(
                "Manage lifecycle-scoped sandboxed background processes. "
                "Actions: start, poll, write, stop, and list. Processes survive "
                "agent turns and are terminated when Citra exits. Network is "
                "off by default; networked starts require a reason and user "
                "approval unless globally allowed."
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
            raise ValueError(f"Unsupported subprocess action: {action}")
        if action == "list":
            return json.dumps(self.context.subprocesses.list(), indent=2)
        if action == "start":
            return self._start(arguments)

        process_id = arguments.get("process_id")
        if process_id is None:
            raise ValueError(f"'process_id' is required for action '{action}'.")
        process_id = int(process_id)
        if action == "poll":
            result = self.context.subprocesses.poll(
                process_id,
                clear=bool(arguments.get("clear", True)),
            )
            output = str(result.get("output", ""))
            limit = self.context.config.subprocess.max_output_length
            if len(output) > limit:
                omitted = len(output) - limit
                result["output"] = (
                    output[:limit]
                    + f"\n... <truncated {omitted} characters>"
                )
            return json.dumps(
                result,
                indent=2,
            )
        if action == "write":
            if "input" not in arguments:
                raise ValueError("'input' is required for action 'write'.")
            self.context.subprocesses.write(process_id, str(arguments["input"]))
            return "ok"
        return json.dumps(self.context.subprocesses.stop(process_id), indent=2)

    def _start(self, arguments: dict[str, Any]) -> str:
        cmd = str(arguments.get("cmd", ""))
        if not cmd.strip():
            raise ValueError("'cmd' is required for action 'start'.")
        network = bool(arguments.get("network", False))
        reason = str(arguments.get("reason", "")).strip()
        if network and not reason:
            raise ValueError("'reason' is required for a networked subprocess.")
        if not network and "reason" in arguments:
            raise ValueError("'reason' is only valid when 'network' is true.")

        workspace = self.context.workspace
        cwd = (
            workspace.workspace
            if arguments.get("cwd") is None
            else workspace.resolve_path(str(arguments["cwd"]))
        )
        if not cwd.is_dir():
            raise NotADirectoryError(
                f"Subprocess working directory does not exist: {workspace.display_path(cwd)}"
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
                    "options": ["Allow once", "Deny"],
                    "timeout": config.permission_timeout,
                }
            )
            if permission != "Allow once":
                return "permission-denied: subprocess was not started."

        process_id = self.context.subprocesses.start(cmd, cwd=cwd, network=network)
        return f"Started subprocess {process_id}."

    @staticmethod
    def _safe(value: str) -> str:
        return value.encode("unicode_escape").decode("ascii")
