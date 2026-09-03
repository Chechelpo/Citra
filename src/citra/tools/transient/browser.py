"""Constrained lifecycle-scoped browser automation tool."""

from __future__ import annotations

import json
from typing import Any, override
from urllib.parse import urlparse

from ...context import ExecutionContext
from ..tool import Tool, ToolDefinition
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from .prompt_user import PromptUser


class Browser(Tool):
    """Represent Browser."""
    TOOL_ID = "browser"
    SAFE_ACTIONS = frozenset(
        {
            "open",
            "snapshot",
            "click",
            "dblclick",
            "hover",
            "fill",
            "press",
            "check",
            "uncheck",
            "select",
            "scroll_into_view",
            "wait_for",
            "download",
            "screenshot",
            "console",
            "errors",
            "reload",
            "back",
            "forward",
            "close",
        }
    )

    SUPPORTED_UNSAFE_ACTIONS = frozenset(
        {
            "evaluate",
            "upload",
        }
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="browser",
            description=(
                "Test web applications with a lifecycle-scoped headless "
                "Chromium browser. Use snapshot to obtain stable element "
                "references for interaction. Supported safe actions include "
                f"{SAFE_ACTIONS}. Open requires a network reason and "
                "user approval unless configured otherwise. Capabilities that "
                "can expose local data or execute arbitrary page JavaScript, "
                "such as upload and evaluate, are disabled by default and must "
                "be explicitly enabled in browser configuration. Unsafe actions "
                "also require a reason and user approval unless configured as "
                "always allowed."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description=(
                                "Browser action."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="values",
                        schema=JsonSchema.array(
                            items=JsonSchema.string(),
                            description=(
                                "Option values for action 'select'. "
                                "Use 'value' for a single option."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="url",
                        schema=JsonSchema.string(),
                        required=False,
                    ),
                    JsonProperty(
                        name="reason",
                        schema=JsonSchema.string(
                            description=(
                                "Required for navigation and unsafe actions. "
                                "Explain why the operation is necessary."
                            ),
                        ),
                        required=False,
                    ),
                                        JsonProperty(
                        name="ref",
                        schema=JsonSchema.string(
                            description=(
                                "Stable element reference returned by "
                                "'snapshot'. Use either 'ref' or 'selector'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="selector",
                        schema=JsonSchema.string(
                            description=(
                                "Playwright selector used to locate an element. "
                                "Use either 'selector' or a snapshot 'ref'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="value",
                        schema=JsonSchema.string(
                            description=(
                                "Text for 'fill' or a single option value "
                                "for 'select'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="key",
                        schema=JsonSchema.string(),
                        required=False,
                    ),
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(),
                        required=False,
                    ),
                    JsonProperty(
                        name="full_page",
                        schema=JsonSchema.boolean(),
                        required=False,
                    ),
                    JsonProperty(
                        name="state",
                        schema=JsonSchema.string(),
                        required=False,
                    ),
                    JsonProperty(
                        name="timeout",
                        schema=JsonSchema.integer(),
                        required=False,
                    ),
                    JsonProperty(
                        name="expression",
                        schema=JsonSchema.string(
                            description=(
                                "JavaScript expression or function body for "
                                "the unsafe evaluate action."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        )
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
    
    def __init__(self, context: ExecutionContext) -> None:
        """Initialize the instance."""
        super().__init__(context)

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Execute the execute operation."""
        action = str(arguments["action"]).strip()
        config = self.context.config.browser

        enabled_unsafe_actions = self._enabled_unsafe_actions(
            config.enabled_unsafe_actions
        )

        if (
            action not in self.SAFE_ACTIONS
            and action not in enabled_unsafe_actions
        ):
            if action in self.SUPPORTED_UNSAFE_ACTIONS:
                raise PermissionError(
                    f"Unsafe browser action {action!r} is disabled. "
                    "Enable it through "
                    "'browser.enabled_unsafe_actions'."
                )

            raise ValueError(
                f"Unsupported browser action: {action!r}"
            )

        if action == "close":
            if len(arguments) != 1:
                raise ValueError(
                    "Action 'close' accepts no additional arguments."
                )

            self.context.browser.close()
            return "Browser session closed."

        request = dict(arguments)
        request.pop("action", None)
        reason = str(
            request.pop("reason", "")
        ).strip()

        timeout = int(
            request.pop("timeout", 30)
        )

        if timeout <= 0:
            raise ValueError(
                "'timeout' must be greater than zero."
            )

        request["timeout_ms"] = timeout * 1000

        if action == "evaluate":
            expression = str(
                arguments.get("expression", "")
            ).strip()

            if not expression:
                raise ValueError(
                    "'expression' is required for action 'evaluate'."
                )

            request["expression"] = expression

        if action == "upload":
            raw_path = arguments.get(
                "path"
            )

            if raw_path is None:
                raise ValueError(
                    "'path' is required for action 'upload'."
                )

            source = self.context.workspace.resolve_path(
                str(raw_path)
            )

            if not source.is_file():
                raise FileNotFoundError(
                    "Upload source is not a file: "
                    f"{self.context.workspace.display_path(source)}"
                )

            request["path"] = str(
                source
            )

        if action == "screenshot":
            raw_path = arguments.get(
                "path",
                "@tmp/browser/screenshot.png",
            )

            destination = (
                self.context.workspace.require_writable_path(
                    str(raw_path)
                )
            )

            request["path"] = str(
                destination
            )

        if action == "download":
            raw_path = arguments.get(
                "path"
            )

            if raw_path is None:
                raise ValueError(
                    "'path' is required for action 'download'."
                )

            destination = (
                self.context.workspace.require_writable_path(
                    str(raw_path)
                )
            )

            request["path"] = str(
                destination
            )

        if action == "open":
            self._authorize_navigation(
                arguments=arguments,
                reason=reason,
            )
        elif action in enabled_unsafe_actions:
            self._authorize_unsafe_action(
                action=action,
                reason=reason,
                request=request,
            )
        elif reason:
            raise ValueError(
                "'reason' is only valid for action 'open' "
                "or an enabled unsafe action."
            )

        if action == "evaluate":
            expression = str(
                arguments.get("expression", "")
            ).strip()

            if not expression:
                raise ValueError(
                    "'expression' is required for action 'evaluate'."
                )

            request["expression"] = expression

        if action == "screenshot":
            raw_path = arguments.get(
                "path",
                "@tmp/browser/screenshot.png",
            )
            destination = (
                self.context.workspace.require_writable_path(
                    str(raw_path)
                )
            )
            request["path"] = str(destination)

        response = self.context.browser.request(
            action,
            **request,
        )
        result = response.get("result")

        if (
            action in {"screenshot", "download"}
            and isinstance(result, dict)
            and "path" in result
        ):
            result["path"] = (
                self.context.workspace.display_path(
                    result["path"]
                )
            )

        return json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Handle format call log."""
        action = str(arguments.get("action", "?"))
        parts = [f"action={action}"]

        url = arguments.get("url")
        if url is not None:
            parts.append(f"url={url}")

        ref = arguments.get("ref")
        if ref is not None:
            parts.append(f"ref={ref}")

        selector = arguments.get("selector")
        if selector is not None:
            parts.append(f"selector={selector}")

        value = arguments.get("value")
        if value is not None:
            parts.append(f"value={self._truncate(str(value))}")

        key = arguments.get("key")
        if key is not None:
            parts.append(f"key={key}")

        return " | ".join(parts)

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        """Handle format result log."""
        text = str(result)

        if text == "Browser session closed.":
            return "closed"

        lines = text.splitlines()
        return f"{len(lines)} lines"

    @staticmethod
    def _truncate(value: str) -> str:
        """Handle truncate."""
        if len(value) <= 120:
            return value
        return value[:120] + "..."

    def _authorize_navigation(
        self,
        *,
        arguments: dict[str, Any],
        reason: str,
    ) -> None:
        """Handle authorize navigation."""
        url = str(
            arguments.get("url", "")
        ).strip()

        self._validate_url(url)

        if not reason:
            raise ValueError(
                "'reason' is required for browser navigation."
            )

        config = self.context.config.browser

        if config.always_allow_network:
            return

        permission = PromptUser(self.context)._execute(
            {
                "question": (
                    "Allow the browser to access this origin?\n\n"
                    f"URL: {self._safe(url)}\n"
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
            raise PermissionError(
                "permission-denied: browser navigation "
                "was not executed."
            )

    def _authorize_unsafe_action(
        self,
        *,
        action: str,
        reason: str,
        request: dict[str, Any],
    ) -> None:
        """Handle authorize unsafe action."""
        if not reason:
            raise ValueError(
                f"'reason' is required for unsafe browser "
                f"action {action!r}."
            )

        config = self.context.config.browser

        if config.always_allow_unsafe_actions:
            return

        details = self._unsafe_action_details(
            action,
            request,
        )

        permission = PromptUser(self.context)._execute(
            {
                "question": (
                    "Allow this unsafe browser action?\n\n"
                    f"Action: {self._safe(action)}\n"
                    f"Reason: {self._safe(reason)}\n"
                    f"{details}"
                ),
                "options": [
                    "Allow once",
                    "Deny",
                ],
                "timeout": config.permission_timeout,
            }
        )

        if permission != "Allow once":
            raise PermissionError(
                "permission-denied: unsafe browser "
                "action was not executed."
            )

    @classmethod
    def _enabled_unsafe_actions(
        cls,
        configured: object,
    ) -> frozenset[str]:
        """Handle enabled unsafe actions."""
        if not isinstance(
            configured,
            (list, tuple, frozenset),
        ):
            raise TypeError(
                "'browser.enabled_unsafe_actions' "
                "must be a list of strings."
            )

        if not all(
            isinstance(action, str)
            for action in configured
        ):
            raise TypeError(
                "'browser.enabled_unsafe_actions' "
                "must contain only strings."
            )

        enabled = frozenset(
            action.strip()
            for action in configured
        )

        unknown = enabled - cls.SUPPORTED_UNSAFE_ACTIONS

        if unknown:
            formatted = ", ".join(
                sorted(unknown)
            )
            raise ValueError(
                "Unsupported configured unsafe browser "
                f"actions: {formatted}"
            )

        return enabled

    @classmethod
    def _unsafe_action_details(
        cls,
        action: str,
        request: dict[str, Any],
    ) -> str:
        """Handle unsafe action details."""
        if action == "evaluate":
            expression = str(
                request.get("expression", "")
            )

            maximum_display_length = 4000

            if len(expression) > maximum_display_length:
                expression = (
                    expression[:maximum_display_length]
                    + "\n... (truncated)"
                )

            return (
                "JavaScript:\n"
                f"{cls._safe(expression)}"
            )

        if action == "upload":
            path = str(
                request.get("path", "")
            )

            reference = request.get(
                "ref"
            )
            selector = request.get(
                "selector"
            )

            target = (
                f"ref={reference}"
                if reference is not None
                else f"selector={selector}"
            )

            return (
                f"Local file: {cls._safe(path)}\n"
                f"Target: {cls._safe(target)}"
            )

        return ""

    @staticmethod
    def _validate_url(url: str) -> None:
        """Handle validate url."""
        parsed = urlparse(url)

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
        ):
            raise ValueError(
                "Browser URLs must be absolute HTTP "
                "or HTTPS URLs."
            )

        if (
            parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                "Credentials must not be embedded "
                "in browser URLs."
            )

        if any(
            character in url
            for character in ("\x00", "\n", "\r")
        ):
            raise ValueError(
                "Browser URL contains invalid characters."
            )

    @staticmethod
    def _safe(value: str) -> str:
        """Handle safe."""
        return value.encode(
            "unicode_escape"
        ).decode(
            "ascii"
        )