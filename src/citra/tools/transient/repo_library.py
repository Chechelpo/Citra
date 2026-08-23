from __future__ import annotations

from pathlib import Path
from typing import Any, override

from ...context import ExecutionContext
from ...context.libraries.repository_library import RepoFileEdit, RepositoryLibrary
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..tool import Tool


class RepoLibrary(Tool):
    """
    Agent-facing facade over the persistent repository documentation library.

    This tool exposes no arbitrary filesystem path under .citra. Every
    repository and documentation path is resolved and validated by
    RepositoryLibrary / DocumentedRepository.

    Repository creation and version creation use RepositoryLibrary's Git hook,
    allowing branch/commit references to be resolved by the configured Git
    utility without exposing arbitrary Git commands here.
    """

    DEFAULT_SEARCH_LIMIT = 50
    MAX_SEARCH_LIMIT = 200

    ACTIONS = frozenset(
        {
            "list",
            "get",
            "versions",
            "tree",
            "search",
            "read",
            "create",
            "create_version",
            "set_preferred",
            "delete_repo",
            "delete_version",
            "changed_source_files",
            "add",
            "replace",
            "edit",
            "move",
            "delete",
        }
    )

    _FILE_CONTENT_SCHEMA = JsonSchema.object(
        properties=(
            JsonProperty(
                name="path",
                schema=JsonSchema.string(
                    description="Relative Markdown path inside the documentation version."
                ),
            ),
            JsonProperty(
                name="content",
                schema=JsonSchema.string(
                    description="Complete file content."
                ),
            ),
        ),
        additional_properties=False,
    )

    _EDIT_SCHEMA = JsonSchema.object(
        properties=(
            JsonProperty(
                name="path",
                schema=JsonSchema.string(
                    description="Relative Markdown path inside the documentation version."
                ),
            ),
            JsonProperty(
                name="content",
                schema=JsonSchema.string(
                    description="Content to insert or use as replacement text."
                ),
            ),
            JsonProperty(
                name="at_line",
                schema=JsonSchema.integer(
                    description="First affected line. Lines are 1-indexed."
                ),
            ),
            JsonProperty(
                name="to_line",
                schema=JsonSchema.integer(
                    description=(
                        "Optional inclusive final line to replace. If omitted, "
                        "content is inserted before at_line."
                    ),
                ),
                required=False,
            ),
        ),
        additional_properties=False,
    )

    _MOVE_SCHEMA = JsonSchema.object(
        properties=(
            JsonProperty(
                name="source",
                schema=JsonSchema.string(
                    description="Existing relative documentation path."
                ),
            ),
            JsonProperty(
                name="destination",
                schema=JsonSchema.string(
                    description="New relative documentation path."
                ),
            ),
        ),
        additional_properties=False,
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="repo_library",
            description=(
                "Inspect and maintain Citra's persistent documentation library "
                "for external Git repositories. Use it before rescanning an "
                "obscure dependency: list/get/tree/search/read discover cached "
                "knowledge; create/create_version manage commit-scoped snapshots; "
                "add/replace/edit/move/delete maintain Markdown documentation. "
                "Repository versions are keyed by exact Git commits. All document "
                "paths are relative to the selected documentation version."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description=(
                                "Operation: list, get, versions, tree, search, read, "
                                "create, create_version, set_preferred, delete_repo, "
                                "delete_version, changed_source_files, add, replace, "
                                "edit, move, or delete."
                            ),
                            enum=tuple(sorted(ACTIONS)),
                        ),
                    ),
                    JsonProperty(
                        name="repo_url",
                        schema=JsonSchema.string(
                            description=(
                                "GitHub repository URL. Equivalent HTTPS, .git, and "
                                "GitHub SSH forms are canonicalized by the library."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="commit",
                        schema=JsonSchema.string(
                            description=(
                                "Documentation commit to use. For create/create_version, "
                                "this may be omitted and resolved through the configured "
                                "Git utility. For other versioned operations, omission "
                                "selects the preferred documented commit where applicable."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="branch",
                        schema=JsonSchema.string(
                            description=(
                                "Branch hint for create/create_version. May be omitted "
                                "and resolved through the configured Git utility."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="copy_from_commit",
                        schema=JsonSchema.string(
                            description=(
                                "For create_version, copy an existing documentation "
                                "snapshot before updating it for the new commit."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "Relative documentation file/directory used by tree or "
                                "search. Omit for the documentation root."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="query",
                        schema=JsonSchema.string(
                            description="Text query for search.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="max_results",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum search matches. Defaults to 50; maximum 200."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="case_sensitive",
                        schema=JsonSchema.boolean(
                            description="Whether search matching is case-sensitive."
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="files",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Relative Markdown paths to read. Used by action=read."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="documents",
                        schema=JsonSchema.array(
                            _FILE_CONTENT_SCHEMA,
                            description=(
                                "Documentation files for add/replace. Each item contains "
                                "a relative path and content."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="edits",
                        schema=JsonSchema.array(
                            _EDIT_SCHEMA,
                            description=(
                                "Ordered line edits. Earlier edits affect line numbers "
                                "seen by later edits to the same file."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="moves",
                        schema=JsonSchema.array(
                            _MOVE_SCHEMA,
                            description="Ordered source/destination documentation moves.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="paths",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description="Relative documentation paths to delete.",
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="recursive",
                        schema=JsonSchema.boolean(
                            description=(
                                "For delete, allow removal of non-empty documentation "
                                "directories. Defaults to false."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="from_commit",
                        schema=JsonSchema.string(
                            description=(
                                "Starting Git commit for changed_source_files."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="to_commit",
                        schema=JsonSchema.string(
                            description=(
                                "Ending Git commit for changed_source_files."
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

    @property
    def library(self) -> RepositoryLibrary:
        """
        Returns the repository library bound to the current execution context.

        Keeping this as a property rather than caching the object makes
        Tool.rebind_context() safe when transient tools are reused.
        """
        return self.context.libraries.repositories

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        action: str = arguments["action"]

        if action not in self.ACTIONS:
            raise ValueError(
                f"Unsupported repository library action: {action}"
            )

        if action == "list":
            self._reject_fields(
                arguments,
                allowed={"action"},
            )
            return self.library.list_repositories()

        repo_url = self._require_string(
            arguments,
            "repo_url",
            action,
        )

        if action == "get":
            self._reject_fields(
                arguments,
                allowed={"action", "repo_url", "commit"},
            )
            return self.library.get_repo(
                repo_url,
                commit=arguments.get("commit"),
            )

        if action == "versions":
            self._reject_fields(
                arguments,
                allowed={"action", "repo_url"},
            )
            return self.library.list_versions(repo_url)

        if action == "tree":
            self._reject_fields(
                arguments,
                allowed={"action", "repo_url", "commit", "path"},
            )
            return self.library.tree_docs(
                repo_url,
                path=self._optional_path(arguments.get("path")),
                commit=arguments.get("commit"),
            )

        if action == "search":
            self._reject_fields(
                arguments,
                allowed={
                    "action",
                    "repo_url",
                    "commit",
                    "path",
                    "query",
                    "max_results",
                    "case_sensitive",
                },
            )
            query = self._require_string(
                arguments,
                "query",
                action,
            )
            max_results = arguments.get(
                "max_results",
                self.DEFAULT_SEARCH_LIMIT,
            )
            if not 1 <= max_results <= self.MAX_SEARCH_LIMIT:
                raise ValueError(
                    "'max_results' must be between 1 and "
                    f"{self.MAX_SEARCH_LIMIT}."
                )
            return self.library.search_docs(
                repo_url,
                query,
                path=self._optional_path(arguments.get("path")),
                commit=arguments.get("commit"),
                max_results=max_results,
                case_sensitive=arguments.get("case_sensitive", False),
            )

        if action == "read":
            self._reject_fields(
                arguments,
                allowed={"action", "repo_url", "commit", "files"},
            )
            files = self._require_nonempty_list(
                arguments,
                "files",
                action,
            )
            return self.library.read_files(
                repo_url,
                [Path(path) for path in files],
                commit=arguments.get("commit"),
            )

        if action == "create":
            self._reject_fields(
                arguments,
                allowed={"action", "repo_url", "branch", "commit"},
            )
            return self.library.create_repo_from_git(
                repo_url,
                branch=arguments.get("branch"),
                commit=arguments.get("commit"),
            )

        if action == "create_version":
            self._reject_fields(
                arguments,
                allowed={
                    "action",
                    "repo_url",
                    "branch",
                    "commit",
                    "copy_from_commit",
                },
            )
            return self.library.create_version_from_git(
                repo_url,
                branch=arguments.get("branch"),
                commit=arguments.get("commit"),
                copy_from_commit=arguments.get("copy_from_commit"),
            )

        if action == "set_preferred":
            self._reject_fields(
                arguments,
                allowed={"action", "repo_url", "commit"},
            )
            commit = self._require_string(
                arguments,
                "commit",
                action,
            )
            return self.library.set_preferred_version(
                repo_url,
                commit,
            )

        if action == "delete_repo":
            self._reject_fields(
                arguments,
                allowed={"action", "repo_url"},
            )
            return self.library.delete_repo(repo_url)

        if action == "delete_version":
            self._reject_fields(
                arguments,
                allowed={"action", "repo_url", "commit"},
            )
            commit = self._require_string(
                arguments,
                "commit",
                action,
            )
            return self.library.delete_version(
                repo_url,
                commit,
            )

        if action == "changed_source_files":
            self._reject_fields(
                arguments,
                allowed={
                    "action",
                    "repo_url",
                    "from_commit",
                    "to_commit",
                },
            )
            return self.library.changed_source_files(
                repo_url,
                self._require_string(
                    arguments,
                    "from_commit",
                    action,
                ),
                self._require_string(
                    arguments,
                    "to_commit",
                    action,
                ),
            )

        if action in {"add", "replace"}:
            self._reject_fields(
                arguments,
                allowed={"action", "repo_url", "commit", "documents"},
            )
            documents_raw = self._require_nonempty_list(
                arguments,
                "documents",
                action,
            )
            documents: dict[Path, str] = {}
            for index, item in enumerate(documents_raw):
                path = Path(
                    self._require_mapping_string(
                        item,
                        "path",
                        f"documents[{index}]",
                    )
                )
                content = self._require_mapping_string(
                    item,
                    "content",
                    f"documents[{index}]",
                    allow_empty=True,
                )
                if path in documents:
                    raise ValueError(
                        f"Duplicate documentation path in 'documents': {path.as_posix()}"
                    )
                documents[path] = content

            if action == "add":
                return self.library.add_files(
                    repo_url,
                    documents,
                    commit=arguments.get("commit"),
                )
            return self.library.replace_files(
                repo_url,
                documents,
                commit=arguments.get("commit"),
            )

        if action == "edit":
            self._reject_fields(
                arguments,
                allowed={"action", "repo_url", "commit", "edits"},
            )
            edits_raw = self._require_nonempty_list(
                arguments,
                "edits",
                action,
            )
            edits: list[RepoFileEdit] = []
            for index, item in enumerate(edits_raw):
                label = f"edits[{index}]"
                at_line = self._require_mapping_int(
                    item,
                    "at_line",
                    label,
                )
                if at_line < 1:
                    raise ValueError(
                        f"{label}.at_line must be >= 1."
                    )

                to_line_raw = item.get("to_line")
                if to_line_raw is not None:
                    if isinstance(to_line_raw, bool) or not isinstance(to_line_raw, int):
                        raise ValueError(
                            f"{label}.to_line must be an integer."
                        )
                    if to_line_raw < at_line:
                        raise ValueError(
                            f"{label}.to_line must be >= at_line."
                        )

                edits.append(
                    RepoFileEdit(
                        path=Path(
                            self._require_mapping_string(
                                item,
                                "path",
                                label,
                            )
                        ),
                        content=self._require_mapping_string(
                            item,
                            "content",
                            label,
                            allow_empty=True,
                        ),
                        at_line=at_line,
                        to_line=to_line_raw,
                    )
                )

            return self.library.edit_files(
                repo_url,
                edits,
                commit=arguments.get("commit"),
            )

        if action == "move":
            self._reject_fields(
                arguments,
                allowed={"action", "repo_url", "commit", "moves"},
            )
            moves_raw = self._require_nonempty_list(
                arguments,
                "moves",
                action,
            )
            moves: dict[Path, Path] = {}
            for index, item in enumerate(moves_raw):
                label = f"moves[{index}]"
                source = Path(
                    self._require_mapping_string(
                        item,
                        "source",
                        label,
                    )
                )
                destination = Path(
                    self._require_mapping_string(
                        item,
                        "destination",
                        label,
                    )
                )
                if source in moves:
                    raise ValueError(
                        f"Duplicate move source: {source.as_posix()}"
                    )
                moves[source] = destination

            return self.library.move_paths(
                repo_url,
                moves,
                commit=arguments.get("commit"),
            )

        if action == "delete":
            self._reject_fields(
                arguments,
                allowed={
                    "action",
                    "repo_url",
                    "commit",
                    "paths",
                    "recursive",
                },
            )
            paths = self._require_nonempty_list(
                arguments,
                "paths",
                action,
            )
            return self.library.delete_paths(
                repo_url,
                [Path(path) for path in paths],
                commit=arguments.get("commit"),
                recursive=arguments.get("recursive", False),
            )

        raise AssertionError(
            f"Unhandled repository library action: {action}"
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        action = arguments.get("action", "?")
        parts = [f"action={action}"]

        repo_url = arguments.get("repo_url")
        if repo_url is not None:
            parts.append(f"repo={repo_url}")

        commit = arguments.get("commit")
        if commit is not None:
            parts.append(f"commit={commit}")

        path = arguments.get("path")
        if path is not None:
            parts.append(f"path={path}")

        query = arguments.get("query")
        if query is not None:
            parts.append(f"query={self._truncate(str(query))}")

        files = arguments.get("files")
        if files:
            parts.append(f"files={len(files)}")

        documents = arguments.get("documents")
        if documents:
            parts.append(f"documents={len(documents)}")

        edits = arguments.get("edits")
        if edits:
            parts.append(f"edits={len(edits)}")

        moves = arguments.get("moves")
        if moves:
            parts.append(f"moves={len(moves)}")

        return " | ".join(parts)

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        text = str(result)

        if not text:
            return "empty"

        lines = text.splitlines()
        return f"{len(lines)} lines"

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= 120:
            return value
        return value[:120] + "..."

    @staticmethod
    def _require_string(
        arguments: dict[str, Any],
        name: str,
        action: str,
        *,
        allow_empty: bool = False,
    ) -> str:
        value = arguments.get(name)
        if not isinstance(value, str):
            raise ValueError(
                f"'{name}' is required for action '{action}'."
            )
        if not allow_empty and not value.strip():
            raise ValueError(
                f"'{name}' cannot be empty for action '{action}'."
            )
        return value

    @staticmethod
    def _require_nonempty_list(
        arguments: dict[str, Any],
        name: str,
        action: str,
    ) -> list[Any]:
        value = arguments.get(name)
        if not isinstance(value, list) or not value:
            raise ValueError(
                f"'{name}' must be a non-empty array for action '{action}'."
            )
        return value

    @staticmethod
    def _optional_path(
        value: Any,
    ) -> Path | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "'path' must be a non-empty string when provided."
            )
        return Path(value)

    @staticmethod
    def _require_mapping_string(
        mapping: Any,
        name: str,
        label: str,
        *,
        allow_empty: bool = False,
    ) -> str:
        if not isinstance(mapping, dict):
            raise ValueError(
                f"{label} must be an object."
            )
        value = mapping.get(name)
        if not isinstance(value, str):
            raise ValueError(
                f"{label}.{name} must be a string."
            )
        if not allow_empty and not value.strip():
            raise ValueError(
                f"{label}.{name} cannot be empty."
            )
        return value

    @staticmethod
    def _require_mapping_int(
        mapping: Any,
        name: str,
        label: str,
    ) -> int:
        if not isinstance(mapping, dict):
            raise ValueError(
                f"{label} must be an object."
            )
        value = mapping.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{label}.{name} must be an integer."
            )
        return value

    @staticmethod
    def _reject_fields(
        arguments: dict[str, Any],
        *,
        allowed: set[str],
    ) -> None:
        unexpected = sorted(
            set(arguments) - allowed
        )
        if unexpected:
            raise ValueError(
                "Arguments not valid for this action: "
                + ", ".join(unexpected)
            )
