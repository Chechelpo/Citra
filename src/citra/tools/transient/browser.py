"""Constrained lifecycle-scoped browser automation tool."""

from __future__ import annotations

import json
from typing import Any, override
from urllib.parse import urlparse

from ...context import ExecutionContext
from ..tool import Tool
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from .prompt_user import PromptUser


class Browser(Tool):
    ACTIONS = frozenset(
        {
            "open", "snapshot", "click", "fill", "press", "wait_for",
            "screenshot", "console", "errors", "reload", "back", "forward",
            "close",
        }
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="browser",
            description=(
                "Test web applications with a lifecycle-scoped headless "
                "Chromium browser. Open requires a network reason and user "
                "approval unless browser.always_allow_network is enabled. "
                "Use snapshot to obtain stable element refs for click/fill/press."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(name="action", schema=JsonSchema.string()),
                    JsonProperty(name="url", schema=JsonSchema.string(), required=False),
                    JsonProperty(name="reason", schema=JsonSchema.string(), required=False),
                    JsonProperty(name="ref", schema=JsonSchema.string(), required=False),
                    JsonProperty(name="selector", schema=JsonSchema.string(), required=False),
                    JsonProperty(name="value", schema=JsonSchema.string(), required=False),
                    JsonProperty(name="key", schema=JsonSchema.string(), required=False),
                    JsonProperty(name="path", schema=JsonSchema.string(), required=False),
                    JsonProperty(name="full_page", schema=JsonSchema.boolean(), required=False),
                    JsonProperty(name="state", schema=JsonSchema.string(), required=False),
                    JsonProperty(name="timeout", schema=JsonSchema.integer(), required=False),
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
            raise ValueError(f"Unsupported browser action: {action}")

        if action == "close":
            if len(arguments) != 1:
                raise ValueError("Action 'close' accepts no additional arguments.")
            self.context.browser.close()
            return "Browser session closed."

        request = dict(arguments)
        request.pop("action", None)
        request.pop("reason", None)
        timeout = int(request.pop("timeout", 30))
        if timeout <= 0:
            raise ValueError("'timeout' must be greater than zero.")
        request["timeout_ms"] = timeout * 1000

        if action == "open":
            url = str(arguments.get("url", "")).strip()
            self._validate_url(url)
            reason = str(arguments.get("reason", "")).strip()
            if not reason:
                raise ValueError("'reason' is required for browser navigation.")
            config = self.context.config.browser
            if not config.always_allow_network:
                permission = PromptUser(self.context)._execute(
                    {
                        "question": (
                            "Allow the browser to access this origin?\n\n"
                            f"URL: {self._safe(url)}\n"
                            f"Reason: {self._safe(reason)}"
                        ),
                        "options": ["Allow once", "Deny"],
                        "timeout": config.permission_timeout,
                    }
                )
                if permission != "Allow once":
                    return "permission-denied: browser navigation was not executed."
        elif "reason" in arguments:
            raise ValueError("'reason' is only valid for action 'open'.")

        if action == "screenshot":
            raw_path = arguments.get("path")
            if raw_path is None:
                raw_path = "@tmp/browser/screenshot.png"
            destination = self.context.workspace.require_writable_path(str(raw_path))
            request["path"] = str(destination)

        response = self.context.browser.request(action, **request)
        result = response.get("result")
        if action == "screenshot" and isinstance(result, dict):
            result["path"] = self.context.workspace.display_path(result["path"])
        return json.dumps(result, indent=2, ensure_ascii=False)

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Browser URLs must be absolute HTTP or HTTPS URLs.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Credentials must not be embedded in browser URLs.")
        if any(character in url for character in ("\x00", "\n", "\r")):
            raise ValueError("Browser URL contains invalid characters.")

    @staticmethod
    def _safe(value: str) -> str:
        return value.encode("unicode_escape").decode("ascii")
