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


class Subprocess(Tool):
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
                                "Defaults to the active workspace."
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
        super().__init__(
            context=context,
        )