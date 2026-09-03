"""Lifecycle-scoped background subprocess tool."""

from __future__ import annotations

import json
import time
from typing import Any, override

from ...context import ExecutionContext
from ..tool import Tool, ToolDefinition
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from .prompt_user import PromptUser
from .bash import ensure_no_git_command


class Subprocess(Tool):
    """Represent Subprocess."""
    TOOL_ID = "subprocess"

    ACTIONS = frozenset(
        {
            "start",
            "poll",
            "write",
            "stop",
            "list",
        }
    )

    MAX_SLEEP_AFTER_SECONDS = 60

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="subprocess",
            description=(
                "Manage lifecycle-scoped sandboxed background processes. "
                "Use action='start' to launch a persistent process, 'poll' "
                "to read its buffered output and status, 'write' to send "
                "input to it, 'stop' to terminate it, and 'list' to inspect "
                "running processes. Processes survive agent turns and are "
                "terminated when Citra exits. A start may optionally wait "
                "with 'sleep_after' and inspect buffered output with "
                "'poll_after'. Network access is disabled by default; "
                "networked starts require a reason and may require user "
                "approval."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description=(
                                "Process operation to perform."
                            ),
                            enum=(
                                "start",
                                "poll",
                                "write",
                                "stop",
                                "list",
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="cmd",
                        schema=JsonSchema.string(
                            description=(
                                "Shell command to execute. Required for "
                                "action='start'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="cwd",
                        schema=JsonSchema.string(
                            description=(
                                "Working directory for a new process. "
                                "Defaults to the current project."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="process_id",
                        schema=JsonSchema.integer(
                            description=(
                                "Identifier of an existing subprocess. "
                                "Required for poll, write, and stop."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="input",
                        schema=JsonSchema.string(
                            description=(
                                "Text to send to the subprocess stdin. "
                                "Required for action='write'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="clear",
                        schema=JsonSchema.boolean(
                            description=(
                                "Whether output returned by a poll should "
                                "be consumed from the process buffer."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="network",
                        schema=JsonSchema.boolean(
                            description=(
                                "Allow network access for a newly started "
                                "process. Defaults to false."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="reason",
                        schema=JsonSchema.string(
                            description=(
                                "Reason network access is needed. Required "
                                "when network=true."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="sleep_after",
                        schema=JsonSchema.integer(
                            description=(
                                "Seconds to wait after starting before "
                                "returning or polling. Only valid for "
                                "action='start'. Maximum 60 seconds."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="poll_after",
                        schema=JsonSchema.boolean(
                            description=(
                                "Poll once immediately after starting and "
                                "any sleep_after delay. Only valid for "
                                "action='start'."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        """Handle definitions for context."""
        del context

        return (
            ToolDefinition(
                definition=cls.DEFINITION,
            ),
        )

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        """Initialize the instance."""
        super().__init__(
            context=context,
        )

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        """Execute the execute operation."""
        action = arguments.get("action")
        if action not in self.ACTIONS:
            raise ValueError(
                "'action' must be one of: " + ", ".join(sorted(self.ACTIONS))
            )

        if action == "start":
            return self._start(arguments)
        if action == "poll":
            process_id = self._process_id(arguments)
            result = self.context.subprocesses.poll(
                process_id,
                clear=bool(arguments.get("clear", True)),
            )
            return self._render(result)
        if action == "write":
            process_id = self._process_id(arguments)
            value = arguments.get("input")
            if not isinstance(value, str):
                raise ValueError("'input' is required for action='write'.")
            self.context.subprocesses.write(process_id, value)
            return f"Wrote {len(value)} characters to subprocess {process_id}."
        if action == "stop":
            return self._render(
                self.context.subprocesses.stop(self._process_id(arguments))
            )

        return self._render(self.context.subprocesses.list())

    def _start(self, arguments: dict[str, Any]) -> str:
        """Handle start."""
        cmd = arguments.get("cmd")
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("'cmd' is required for action='start'.")
        ensure_no_git_command(cmd)

        cwd_raw = arguments.get("cwd")
        cwd = (
            self.context.workspace.workspace
            if cwd_raw is None
            else self.context.workspace.resolve_path(str(cwd_raw))
        )
        if not cwd.is_dir():
            raise NotADirectoryError(
                "Working directory does not exist: "
                f"{self.context.workspace.display_path(cwd)}"
            )

        network = bool(arguments.get("network", False))
        reason_raw = arguments.get("reason")
        reason = "" if reason_raw is None else str(reason_raw).strip()
        if network and not reason:
            raise ValueError("'reason' is required when network=true.")
        if not network and reason_raw is not None:
            raise ValueError("'reason' is only valid when network=true.")

        if network and not self.context.config.subprocess.always_allow_network:
            permission = PromptUser(self.context)._execute(
                {
                    "question": (
                        "Allow this background process to access the network?\n\n"
                        f"Command:\n{self._safe_text(cmd)}\n\n"
                        "Working directory: "
                        f"{self.context.workspace.display_path(cwd)}\n"
                        f"Reason: {self._safe_text(reason)}"
                    ),
                    "options": ["Allow once", "Deny"],
                    "timeout": self.context.config.subprocess.permission_timeout,
                }
            )
            if permission != "Allow once":
                return (
                    "permission-denied: subprocess network access was not "
                    "granted; the process was not started."
                )

        sleep_after = arguments.get("sleep_after", 0)
        if (
            not isinstance(sleep_after, int)
            or isinstance(sleep_after, bool)
            or not 0 <= sleep_after <= self.MAX_SLEEP_AFTER_SECONDS
        ):
            raise ValueError(
                f"'sleep_after' must be between 0 and {self.MAX_SLEEP_AFTER_SECONDS}."
            )

        process_id = self.context.subprocesses.start(
            cmd.strip(),
            cwd=cwd,
            network=network,
        )
        if sleep_after:
            time.sleep(sleep_after)
        if arguments.get("poll_after", False):
            return self._render(
                self.context.subprocesses.poll(process_id, clear=True)
            )
        return f"Started subprocess {process_id}."

    @staticmethod
    def _process_id(arguments: dict[str, Any]) -> int:
        """Handle process id."""
        value = arguments.get("process_id")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("'process_id' must be a positive integer.")
        return value

    def _render(self, value: object) -> str:
        """Handle render."""
        rendered = json.dumps(value, ensure_ascii=False, indent=2)
        limit = self.context.config.subprocess.max_output_length
        if len(rendered) <= limit:
            return rendered
        omitted = len(rendered) - limit
        return rendered[:limit] + f"\n... <truncated {omitted} characters>"

    @staticmethod
    def _safe_text(value: str) -> str:
        """Handle safe text."""
        return value.replace("\x00", "").replace("\r", "\\r")[:4_000]

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        """Handle format call log."""
        action = str(arguments.get("action", "?"))
        parts = [f"action={action}"]
        if "process_id" in arguments:
            parts.append(f"process={arguments['process_id']}")
        if action == "start" and isinstance(arguments.get("cmd"), str):
            command = str(arguments["cmd"])
            parts.append(f"$ {command[:120]}")
        return " | ".join(parts)

    @override
    def format_result_log(self, result: Any) -> str:
        """Handle format result log."""
        text = str(result)
        return f"{len(text.splitlines())} lines | {len(text)} chars"
