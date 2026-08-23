from __future__ import annotations

from pathlib import Path
import re
import tempfile
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


class Git(Tool):
    """
    Provides constrained Git inspection and repository cloning.

    This tool intentionally exposes no arbitrary Git argument passthrough.
    """

    DEFAULT_TIMEOUT_SECONDS = 30

    DEFAULT_LOG_LIMIT = 30
    MAX_LOG_LIMIT = 200

    DEFAULT_CLONE_DEPTH = 1
    MAX_CLONE_DEPTH = 1000

    MAX_OUTPUT_LENGTH = 50_000

    ACTIONS = frozenset(
        {
            "status",
            "diff",
            "log",
            "show",
            "blame",
            "branches",
            "remotes",
            "root",
            "rev_parse",
            "clone",
            "ls_remote",
        }
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="git",
            description=(
                "Inspect Git repositories and clone repositories for "
                "exploration. Supports status, diff, log, show, blame, "
                "branches, remotes, root, rev_parse, clone, and ls_remote. "
                "Cannot stage, commit, push, pull, fetch, checkout, reset, "
                "merge, rebase, clean, stash, or modify repository history. "
                "Clone uses network access and defaults to an ephemeral "
                "directory under @tmp/repos."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description=(
                                "Git operation: status, diff, log, show, "
                                "blame, branches, remotes, root, rev_parse, "
                                "clone, or ls_remote."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "Repository directory. Relative paths resolve "
                                "from the isolated agent workspace. @source "
                                "addresses the original read-only repository, "
                                "and @tmp addresses disposable storage. "
                                "Defaults to @source."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="ref",
                        schema=JsonSchema.string(
                            description=(
                                "Revision or object name for log, show, "
                                "blame, or rev_parse."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="file",
                        schema=JsonSchema.string(
                            description=(
                                "Repository-relative file path for diff "
                                "or blame."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="staged",
                        schema=JsonSchema.boolean(
                            description=(
                                "For diff, inspect already-staged changes "
                                "instead of working-tree changes. This does "
                                "not modify the index. Defaults to false."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="limit",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum log entries. Defaults to 30; "
                                "maximum 200."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="url",
                        schema=JsonSchema.string(
                            description=(
                                "HTTPS repository URL for clone or ls_remote."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="destination",
                        schema=JsonSchema.string(
                            description=(
                                "Clone destination. If omitted, a unique "
                                "directory under @tmp/repos is created."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="depth",
                        schema=JsonSchema.integer(
                            description=(
                                "Clone history depth. Defaults to 1. "
                                "Maximum 1000."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="timeout",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum execution time in seconds. "
                                "Defaults to 30."
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
            "git"
        ):
            raise RuntimeError(
                "Git is not available in the current execution context."
            )

        action: str = arguments["action"]

        if action not in self.ACTIONS:
            raise ValueError(
                f"Unsupported Git action: {action}"
            )

        timeout: int = arguments.get(
            "timeout",
            self.DEFAULT_TIMEOUT_SECONDS,
        )

        if timeout <= 0:
            raise ValueError(
                "'timeout' must be greater than zero."
            )

        if action == "clone":
            return self._clone(
                arguments,
                timeout,
            )

        if action == "ls_remote":
            return self._ls_remote(
                arguments,
                timeout,
            )

        return self._inspect(
            action,
            arguments,
            timeout,
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        action = arguments.get("action", "?")
        parts = [f"action={action}"]

        path = arguments.get("path")
        if path is not None:
            parts.append(f"path={path}")

        ref = arguments.get("ref")
        if ref is not None:
            parts.append(f"ref={ref}")

        file = arguments.get("file")
        if file is not None:
            parts.append(f"file={file}")

        url = arguments.get("url")
        if url is not None:
            parts.append(f"url={url}")

        limit = arguments.get("limit")
        if limit is not None:
            parts.append(f"limit={limit}")

        if arguments.get("staged"):
            parts.append("staged=true")

        return " | ".join(parts)

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        text = str(result)

        if not text:
            return "empty output"

        lines = text.splitlines()
        parts = [f"{len(lines)} lines", f"{len(text)} chars"]

        if "(timed out after " in text:
            parts.append("timed-out")

        if "error: git exited with code " in text:
            match = re.search(
                r"error: git exited with code (\d+)",
                text,
            )
            if match:
                parts.append(f"exit={match.group(1)}")

        return " | ".join(parts)

    def _inspect(
        self,
        action: str,
        arguments: dict[str, Any],
        timeout: int,
    ) -> str:
        repository = self.context.workspace.resolve_path(
            arguments.get(
                "path",
                "@source",
            )
        )

        if not repository.is_dir():
            raise NotADirectoryError(
                f"Repository path is not a directory: {repository}"
            )

        git_arguments = self._inspection_arguments(
            action,
            arguments,
        )

        result = self.context.sandbox.run(
            [
                *self._git_prefix(),
                "-C",
                str(repository),
                *git_arguments,
            ],
            timeout=timeout,
            network=False,
            environment=self._git_environment(),
        )

        return self._format_result(
            result.output,
            result.returncode,
            result.timed_out,
            timeout,
        )

    def _clone(
        self,
        arguments: dict[str, Any],
        timeout: int,
    ) -> str:
        url = arguments.get(
            "url"
        )

        if not url:
            raise ValueError(
                "'url' is required for action 'clone'."
            )

        self._validate_remote_url(
            url
        )

        depth: int = arguments.get(
            "depth",
            self.DEFAULT_CLONE_DEPTH,
        )

        if not (
            1
            <= depth
            <= self.MAX_CLONE_DEPTH
        ):
            raise ValueError(
                "'depth' must be between 1 and "
                f"{self.MAX_CLONE_DEPTH}."
            )

        destination_raw = arguments.get(
            "destination"
        )

        if destination_raw:
            destination = (
                self.context.workspace
                .require_writable_path(
                    destination_raw
                )
            )

            if destination.exists():
                raise FileExistsError(
                    "Clone destination already exists: "
                    f"{destination}"
                )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        else:
            destination = (
                self._allocate_clone_destination(
                    url
                )
            )

        result = self.context.sandbox.run(
            [
                *self._git_prefix(),

                "clone",

                "--depth",
                str(depth),

                "--single-branch",
                "--no-tags",
                "--no-recurse-submodules",

                "--",
                url,
                str(destination),
            ],
            cwd=self.context.workspace.tmp,
            timeout=timeout,
            network=True,
            environment=self._git_environment(),
        )

        output = self._format_result(
            result.output,
            result.returncode,
            result.timed_out,
            timeout,
        )

        if result.returncode != 0:
            return output

        shown = self.context.workspace.display_path(
            destination
        )

        return (
            f"Cloned to {shown}\n"
            f"{output}"
        )

    def _ls_remote(
        self,
        arguments: dict[str, Any],
        timeout: int,
    ) -> str:
        url = arguments.get(
            "url"
        )

        if not url:
            raise ValueError(
                "'url' is required for action 'ls_remote'."
            )

        self._validate_remote_url(
            url
        )

        result = self.context.sandbox.run(
            [
                *self._git_prefix(),

                "ls-remote",

                "--",
                url,
            ],
            cwd=self.context.workspace.tmp,
            timeout=timeout,
            network=True,
            environment=self._git_environment(),
        )

        return self._format_result(
            result.output,
            result.returncode,
            result.timed_out,
            timeout,
        )

    def _inspection_arguments(
        self,
        action: str,
        arguments: dict[str, Any],
    ) -> list[str]:
        if action == "status":
            return [
                "status",
                "--short",
                "--branch",
                "--untracked-files=all",
                "--ignore-submodules=all",
            ]

        if action == "diff":
            command = [
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--ignore-submodules=all",
            ]

            if arguments.get(
                "staged",
                False,
            ):
                command.append(
                    "--cached"
                )

            file = arguments.get(
                "file"
            )

            if file:
                self._validate_repository_file(
                    file
                )

                command.extend(
                    [
                        "--",
                        file,
                    ]
                )

            return command

        if action == "log":
            limit: int = arguments.get(
                "limit",
                self.DEFAULT_LOG_LIMIT,
            )

            if not (
                1
                <= limit
                <= self.MAX_LOG_LIMIT
            ):
                raise ValueError(
                    "'limit' must be between 1 and "
                    f"{self.MAX_LOG_LIMIT}."
                )

            command = [
                "log",
                f"--max-count={limit}",
                "--decorate=short",
                "--date=iso-strict",
                "--format=%h %ad %an%d%n    %s",
            ]

            ref = arguments.get(
                "ref"
            )

            if ref:
                self._validate_ref(
                    ref
                )

                command.append(
                    ref
                )

            return command

        if action == "show":
            ref = arguments.get(
                "ref",
                "HEAD",
            )

            self._validate_ref(
                ref
            )

            return [
                "show",
                "--no-ext-diff",
                "--no-textconv",
                "--stat",
                "--patch",
                ref,
            ]

        if action == "blame":
            file = arguments.get(
                "file"
            )

            if not file:
                raise ValueError(
                    "'file' is required for action 'blame'."
                )

            self._validate_repository_file(
                file
            )

            command = [
                "blame",
                "--no-progress",
            ]

            ref = arguments.get(
                "ref"
            )

            if ref:
                self._validate_ref(
                    ref
                )

                command.append(
                    ref
                )

            command.extend(
                [
                    "--",
                    file,
                ]
            )

            return command

        if action == "branches":
            return [
                "branch",
                "--all",
                "--verbose",
                "--no-abbrev",
            ]

        if action == "remotes":
            return [
                "remote",
                "--verbose",
            ]

        if action == "root":
            return [
                "rev-parse",
                "--show-toplevel",
            ]

        if action == "rev_parse":
            ref = arguments.get(
                "ref",
                "HEAD",
            )

            self._validate_ref(
                ref
            )

            return [
                "rev-parse",
                "--verify",
                ref,
            ]

        raise ValueError(
            f"Unsupported inspection action: {action}"
        )

    @staticmethod
    def _git_prefix() -> list[str]:
        return [
            "git",

            "--no-pager",
            "--no-optional-locks",

            "-c",
            "credential.helper=",

            "-c",
            "core.hooksPath=/dev/null",

            "-c",
            "core.fsmonitor=false",

            "-c",
            "core.pager=cat",

            "-c",
            "pager.diff=false",

            "-c",
            "protocol.ext.allow=never",
        ]

    @staticmethod
    def _git_environment() -> dict[str, str]:
        return {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "/bin/false",
            "SSH_ASKPASS": "/bin/false",

            "GIT_PAGER": "cat",
            "GIT_EDITOR": "true",
            "GIT_SEQUENCE_EDITOR": "true",

            "GIT_OPTIONAL_LOCKS": "0",

            # Ignore real user/system Git configuration.
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_ATTR_NOSYSTEM": "1",

            # Networked Git operations are HTTPS-only.
            "GIT_ALLOW_PROTOCOL": "https",
        }

    def _allocate_clone_destination(
        self,
        url: str,
    ) -> Path:
        repositories = (
            self.context.workspace.tmp
            / "repos"
        )

        repositories.mkdir(
            parents=True,
            exist_ok=True,
        )

        name = self._repository_name(
            url
        )

        return Path(
            tempfile.mkdtemp(
                prefix=f"{name}-",
                dir=repositories,
            )
        ).resolve()

    @staticmethod
    def _repository_name(
        url: str,
    ) -> str:
        parsed = urlparse(
            url
        )

        name = Path(
            parsed.path
        ).name

        if name.endswith(
            ".git"
        ):
            name = name[:-4]

        name = re.sub(
            r"[^A-Za-z0-9._-]+",
            "-",
            name,
        ).strip(
            ".-"
        )

        return (
            name
            or "repository"
        )

    @staticmethod
    def _validate_remote_url(
        url: str,
    ) -> None:
        parsed = urlparse(
            url
        )

        if parsed.scheme != "https":
            raise ValueError(
                "Git network operations require an HTTPS URL."
            )

        if not parsed.hostname:
            raise ValueError(
                "Git remote URL has no hostname."
            )

        if (
            parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                "Credentials must not be embedded in Git URLs."
            )

    @staticmethod
    def _validate_ref(
        ref: str,
    ) -> None:
        if not ref:
            raise ValueError(
                "Git revision cannot be empty."
            )

        if ref.startswith(
            "-"
        ):
            raise ValueError(
                "Git revision cannot begin with '-'."
            )

        if any(
            character in ref
            for character in (
                "\x00",
                "\n",
                "\r",
            )
        ):
            raise ValueError(
                "Git revision contains invalid characters."
            )

    @staticmethod
    def _validate_repository_file(
        file: str,
    ) -> None:
        path = Path(
            file
        )

        if path.is_absolute():
            raise ValueError(
                "'file' must be repository-relative."
            )

        if ".." in path.parts:
            raise ValueError(
                "'file' cannot escape the repository."
            )

    def _format_result(
        self,
        output: str,
        returncode: int,
        timed_out: bool,
        timeout: int,
    ) -> str:
        output = output.strip()

        if len(output) > self.MAX_OUTPUT_LENGTH:
            omitted = (
                len(output)
                - self.MAX_OUTPUT_LENGTH
            )

            output = (
                output[
                    :self.MAX_OUTPUT_LENGTH
                ]
                + "\n"
                + f"... <truncated {omitted} characters>"
            )

        if timed_out:
            marker = (
                f"(timed out after {timeout}s)"
            )

            return (
                f"{output}\n{marker}"
                if output
                else marker
            )

        if returncode != 0:
            if output:
                return (
                    f"error: git exited with code "
                    f"{returncode}\n{output}"
                )

            return (
                f"error: git exited with code "
                f"{returncode}"
            )

        return output or "(empty)"
