"""Permission-gated HTTP requests through sandboxed curl."""

from __future__ import annotations

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


class Curl(Tool):
    """Perform constrained HTTP requests with explicit user authorization."""

    METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"})
    ALLOW_OPTION = "Allow once"
    DENY_OPTION = "Deny"
    MAX_HEADERS = 50

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="curl",
            description=(
                "Perform a sandboxed HTTP or HTTPS request using curl. "
                "Unless the user enabled curl.always_allow_network, every "
                "request asks the terminal user for permission before network "
                "access is granted. Arbitrary curl flags and file uploads are "
                "not supported."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="url",
                        schema=JsonSchema.string(
                            description="Absolute HTTP or HTTPS URL."
                        ),
                    ),
                    JsonProperty(
                        name="method",
                        schema=JsonSchema.string(
                            description=(
                                "HTTP method: GET, HEAD, POST, PUT, PATCH, or "
                                "DELETE. Defaults to GET."
                            )
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="headers",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Optional HTTP headers in 'Name: value' form."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="data",
                        schema=JsonSchema.string(
                            description=(
                                "Optional literal request body. Local file "
                                "references and file uploads are not expanded."
                            )
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="follow_redirects",
                        schema=JsonSchema.boolean(
                            description="Follow HTTP redirects. Defaults to false."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="include_response_headers",
                        schema=JsonSchema.boolean(
                            description=(
                                "Include response headers in output. Defaults "
                                "to false."
                            )
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="download_to",
                        schema=JsonSchema.string(
                            description=(
                                "Optional writable destination file. Relative "
                                "paths resolve from the agent workspace; "
                                "aliases such as @tmp are supported. Parent "
                                "directories are created automatically. The "
                                "destination is not overwritten unless "
                                "overwrite is true."
                            )
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="overwrite",
                        schema=JsonSchema.boolean(
                            description=(
                                "Allow replacing an existing download_to "
                                "file. Defaults to false."
                            )
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="timeout",
                        schema=JsonSchema.integer(
                            description=(
                                "Request timeout in seconds. Defaults to the "
                                "configured curl.default_timeout."
                            )
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
        if not self.context.has_command("curl"):
            raise RuntimeError("curl is not available in this execution context.")

        url = str(arguments["url"]).strip()
        self._validate_url(url)

        method = str(arguments.get("method", "GET")).upper()
        if method not in self.METHODS:
            raise ValueError(f"Unsupported HTTP method: {method}")

        headers = [str(value).strip() for value in arguments.get("headers", ())]
        self._validate_headers(headers)

        data = arguments.get("data")
        if data is not None:
            data = str(data)
            if method in {"GET", "HEAD"}:
                raise ValueError(f"'{method}' requests cannot include 'data'.")

        destination = None
        destination_raw = arguments.get("download_to")
        if destination_raw is not None:
            if method == "HEAD":
                raise ValueError("HEAD responses cannot be downloaded to a file.")
            destination = self.context.workspace.require_writable_path(
                str(destination_raw)
            )
            if destination.exists() and not arguments.get("overwrite", False):
                raise FileExistsError(
                    "Download destination already exists: "
                    f"{self.context.workspace.display_path(destination)}"
                )
            if destination.exists() and destination.is_dir():
                raise IsADirectoryError(
                    f"Download destination is a directory: {destination}"
                )

        config = self.context.config.curl
        timeout = int(arguments.get("timeout", config.default_timeout))
        if timeout <= 0 or timeout > config.max_timeout:
            raise ValueError(
                f"'timeout' must be between 1 and {config.max_timeout}."
            )

        if not config.always_allow_network:
            permission = PromptUser(self.context)._execute(
                {
                    "question": self._permission_question(
                        method,
                        url,
                        headers,
                        data,
                        destination=(
                            self.context.workspace.display_path(destination)
                            if destination is not None
                            else None
                        ),
                    ),
                    "options": [self.ALLOW_OPTION, self.DENY_OPTION],
                    "timeout": config.permission_timeout,
                }
            )
            if permission != self.ALLOW_OPTION:
                return "permission-denied: curl request was not executed."

        command = [
            "curl",
            "--disable",
            "--silent",
            "--show-error",
            "--fail-with-body",
            "--proto",
            "=http,https",
            "--proto-redir",
            "=http,https",
            "--connect-timeout",
            str(min(timeout, 10)),
            "--max-time",
            str(timeout),
            "--request",
            method,
        ]

        if arguments.get("follow_redirects", False):
            command.extend(("--location", "--max-redirs", "5"))
        if arguments.get("include_response_headers", False):
            command.append("--include")
        if method == "HEAD":
            command.append("--head")
        for header in headers:
            command.extend(("--header", header))
        if data is not None:
            command.extend(("--data-raw", data))
        if destination is not None:
            command.extend(
                (
                    "--create-dirs",
                    "--remove-on-error",
                    "--output",
                    str(destination),
                )
            )
        command.extend(("--url", url))

        result = self.context.sandbox.run(
            command,
            timeout=timeout + 5,
            network=True,
            environment={
                "CURL_HOME": str(self.context.workspace.home),
            },
        )
        formatted = self._format_result(
            result.output,
            result.returncode,
            result.timed_out,
            timeout,
            config.max_output_length,
        )
        if result.returncode == 0 and not result.timed_out and destination is not None:
            shown = self.context.workspace.display_path(destination)
            return f"Downloaded to {shown}"
        return formatted

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("curl URLs must use HTTP or HTTPS.")
        if not parsed.hostname:
            raise ValueError("curl URL has no hostname.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Credentials must not be embedded in curl URLs.")
        if any(character in url for character in ("\x00", "\n", "\r")):
            raise ValueError("curl URL contains invalid characters.")

    @classmethod
    def _validate_headers(cls, headers: list[str]) -> None:
        if len(headers) > cls.MAX_HEADERS:
            raise ValueError(f"At most {cls.MAX_HEADERS} headers are allowed.")
        for header in headers:
            if not header or ":" not in header:
                raise ValueError("Each header must use 'Name: value' form.")
            if any(character in header for character in ("\x00", "\n", "\r")):
                raise ValueError("Headers cannot contain control newlines.")

    @staticmethod
    def _permission_question(
        method: str,
        url: str,
        headers: list[str],
        data: str | None,
        destination: str | None,
    ) -> str:
        details = [f"Allow curl network request?\n{method} {url}"]
        if headers:
            details.append(f"Headers: {len(headers)} supplied")
        if data is not None:
            details.append(f"Request body: {len(data.encode('utf-8'))} bytes")
        if destination is not None:
            details.append(f"Download destination: {destination}")
        return "\n".join(details)

    @staticmethod
    def _format_result(
        output: str,
        returncode: int,
        timed_out: bool,
        timeout: int,
        max_output_length: int,
    ) -> str:
        output = output.strip()
        if len(output) > max_output_length:
            omitted = len(output) - max_output_length
            output = output[:max_output_length] + (
                f"\n... <truncated {omitted} characters>"
            )
        if timed_out:
            marker = f"(timed out after {timeout}s)"
            return f"{output}\n{marker}" if output else marker
        if returncode != 0:
            prefix = f"error: curl exited with code {returncode}"
            return f"{prefix}\n{output}" if output else prefix
        return output or "(empty)"
