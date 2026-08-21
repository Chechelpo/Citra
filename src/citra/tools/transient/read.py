import glob as globlib
from pathlib import Path
from typing import Any, override


from ...context import ExecutionContext
from ..tool import Tool
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ...utils.converters import convert

class Read(Tool):
    """
    Reads one or more files.

    Paths may be literal files or glob patterns.

    Every file passes through convert_to_readable() before its contents
    are opened, allowing non-text formats to be converted into an
    LLM-readable representation later.
    """

    MAX_PARENT_ENTRIES = 50
    MAX_REQUESTS = 20
    MAX_FILES_PER_CALL = 20

    READ_REQUEST_SCHEMA = JsonSchema.object(
        properties=(
            JsonProperty(
                name="path",
                schema=JsonSchema.string(
                    description=(
                        "File path or glob pattern to read. "
                        "Examples: 'main.py', '*.py', 'src/**/*.py', "
                        "or '@tmp/repos/**/*.toml'. Relative paths "
                        "resolve from the active workspace."
                    ),
                ),
            ),
            JsonProperty(
                name="offset",
                schema=JsonSchema.integer(
                    description=(
                        "Zero-based line offset applied to every file "
                        "matched by this request. Defaults to 0."
                    ),
                ),
                required=False,
            ),
            JsonProperty(
                name="limit",
                schema=JsonSchema.integer(
                    description=(
                        "Maximum number of lines returned from every file "
                        "matched by this request. If omitted, reads until "
                        "end of file."
                    ),
                ),
                required=False,
            ),
        ),
        additional_properties=False,
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="read",
            description=(
                "Read one or more files and return their contents with "
                "line numbers. Paths may be literal files or glob patterns "
                "such as '*.py' and 'src/**/*.py'. For one path or pattern "
                "use path, offset, and limit. For multiple independent paths "
                "or patterns use requests, where each request may specify its "
                "own offset and limit. At most 20 actual files are read per "
                "call after wildcard expansion. Relative paths resolve from "
                "the active workspace and filesystem aliases such as @tmp "
                "are supported. Missing literal files and unmatched patterns "
                "include a directory listing to help locate the intended "
                "file. Batch reads are best-effort: one failed request does "
                "not prevent other files from being returned."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "Single file path or glob pattern to read. "
                                "Use 'requests' for multiple independent "
                                "paths or patterns."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="offset",
                        schema=JsonSchema.integer(
                            description=(
                                "Zero-based line offset applied to every "
                                "file selected by top-level 'path'. "
                                "Defaults to 0."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="limit",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum number of lines returned from every "
                                "file selected by top-level 'path'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="requests",
                        schema=JsonSchema.array(
                            READ_REQUEST_SCHEMA,
                            description=(
                                "Independent file paths or glob patterns "
                                "to read as a batch. Each request may specify "
                                "its own offset and limit. At most 20 requests "
                                "and 20 expanded files are processed per call."
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
        path = arguments.get(
            "path"
        )

        requests = arguments.get(
            "requests"
        )

        if (
            path is not None
            and requests is not None
        ):
            raise ValueError(
                "Use either 'path' or 'requests', not both."
            )

        single_literal = False

        if path is not None:
            requests = [
                {
                    "path": path,
                    "offset": arguments.get(
                        "offset",
                        0,
                    ),
                    "limit": arguments.get(
                        "limit"
                    ),
                }
            ]

            single_literal = (
                not globlib.has_magic(
                    path
                )
            )

        else:
            if not requests:
                raise ValueError(
                    "'path' or 'requests' is required."
                )

            if (
                arguments.get("offset") is not None
                or arguments.get("limit") is not None
            ):
                raise ValueError(
                    "'offset' and 'limit' are only valid with "
                    "top-level 'path'. Batch requests specify "
                    "their own offset and limit."
                )

        if len(requests) > self.MAX_REQUESTS:
            raise ValueError(
                f"At most {self.MAX_REQUESTS} read requests "
                "may be submitted in one call."
            )

        results: list[str] = []

        seen: set[Path] = set()

        selected_count = 0
        omitted_count = 0

        for request in requests:
            requested_path: str = str(request[
                "path"
            ])

            offset: int = request.get(
                "offset",
                0,
            )

            limit: int | None = request.get(
                "limit"
            )

            try:
                self._validate_range(
                    offset=offset,
                    limit=limit,
                )

                matches = self._expand_path(
                    requested_path
                )

            except Exception as error:
                results.append(
                    f"===== {requested_path} =====\n"
                    f"error: {error}"
                )
                continue

            if not matches:
                results.append(
                    f"===== {requested_path} =====\n"
                    f"error: "
                    f"{self._missing_pattern_message(requested_path)}"
                )
                continue

            for resolved in matches:
                if resolved in seen:
                    continue

                seen.add(
                    resolved
                )

                if (
                    selected_count
                    >= self.MAX_FILES_PER_CALL
                ):
                    omitted_count += 1
                    continue

                shown_path = (
                    self.context.workspace
                    .display_path(
                        resolved
                    )
                )

                try:
                    content = self._read_resolved(
                        resolved,
                        offset=offset,
                        limit=limit,
                    )

                except Exception as error:
                    content = (
                        f"error: {error}"
                    )

                results.append(
                    f"===== {shown_path} =====\n"
                    f"{content}"
                )

                selected_count += 1

        if omitted_count:
            results.append(
                "===== truncated =====\n"
                f"{omitted_count} additional matching file(s) "
                f"were not read because the "
                f"{self.MAX_FILES_PER_CALL}-file limit "
                "was reached."
            )

        if (
            single_literal
            and selected_count == 1
            and len(results) == 1
        ):
            _, _, content = results[0].partition(
                "\n"
            )

            return content

        return "\n\n".join(
            results
        )

    def __convert_to_readable(
        self,
        path: Path,
    ) -> Path:
        workspace = self.context.workspace

        source = workspace.require_allowed_path(path)

        readable = convert(
            source,
            workspace=workspace,
        )

        return workspace.require_allowed_path(
            readable
        )
    
    def _expand_path(
        self,
        value: str,
    ) -> list[Path]:
        workspace = self.context.workspace

        if not globlib.has_magic(
            value
        ):
            resolved = workspace.resolve_path(
                value
            )

            if not resolved.exists():
                raise FileNotFoundError(
                    self._missing_file_message(
                        resolved
                    )
                )

            return [
                resolved
            ]

        base, pattern = self._split_glob(
            value
        )

        resolved_base = workspace.resolve_path(
            base
        )

        if not resolved_base.exists():
            raise FileNotFoundError(
                self._missing_file_message(
                    resolved_base
                )
            )

        if not resolved_base.is_dir():
            raise NotADirectoryError(
                "Glob search root is not a directory: "
                f"{workspace.display_path(resolved_base)}"
            )

        raw_matches = globlib.glob(
            str(
                resolved_base
                / pattern
            ),
            recursive=True,
        )

        matches: list[Path] = []

        for raw_match in raw_matches:
            resolved = workspace.resolve_path(
                raw_match
            )

            if not resolved.is_file():
                continue

            matches.append(
                resolved
            )

        return sorted(
            set(
                matches
            ),
            key=lambda path: (
                workspace.display_path(
                    path
                ).lower()
            ),
        )

    @staticmethod
    def _split_glob(
        value: str,
    ) -> tuple[str, str]:
        path = Path(
            value
        )

        static_parts: list[str] = []
        pattern_parts: list[str] = []

        found_magic = False

        for part in path.parts:
            if (
                not found_magic
                and globlib.has_magic(
                    part
                )
            ):
                found_magic = True

            if found_magic:
                pattern_parts.append(
                    part
                )
            else:
                static_parts.append(
                    part
                )

        if not pattern_parts:
            return (
                value,
                "",
            )

        base = (
            str(
                Path(
                    *static_parts
                )
            )
            if static_parts
            else "."
        )

        pattern = str(
            Path(
                *pattern_parts
            )
        )

        return (
            base,
            pattern,
        )

    @staticmethod
    def _validate_range(
        *,
        offset: int,
        limit: int | None,
    ) -> None:
        if offset < 0:
            raise ValueError(
                "'offset' cannot be negative."
            )

        if (
            limit is not None
            and limit < 0
        ):
            raise ValueError(
                "'limit' cannot be negative."
            )

    def _read_resolved(
        self,
        path: Path,
        *,
        offset: int,
        limit: int | None,
    ) -> str:
        workspace = self.context.workspace

        if not path.is_file():
            raise IsADirectoryError(
                "Path is not a file: "
                f"{workspace.display_path(path)}"
            )

        readable_path = self.__convert_to_readable(
            path
        )

        if not readable_path.exists():
            raise FileNotFoundError(
                "Readable conversion did not produce a file: "
                f"{readable_path}"
            )

        if not readable_path.is_file():
            raise IsADirectoryError(
                "Readable conversion did not produce a file: "
                f"{readable_path}"
            )

        with readable_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:
            lines = file.readlines()

        if limit is None:
            selected = lines[
                offset:
            ]
        else:
            selected = lines[
                offset:offset + limit
            ]

        return "".join(
            f"{offset + index + 1:4}| {line}"
            for index, line in enumerate(
                selected
            )
        )

    def _missing_file_message(
        self,
        path: Path,
    ) -> str:
        workspace = self.context.workspace

        shown_path = workspace.display_path(
            path
        )

        parent = path.parent

        if not parent.is_dir():
            return (
                f"File not found: {shown_path}\n"
                f"Parent directory does not exist: "
                f"{workspace.display_path(parent)}"
            )

        try:
            entries = sorted(
                parent.iterdir(),
                key=lambda entry: (
                    not entry.is_dir(),
                    entry.name.lower(),
                ),
            )

        except OSError as error:
            return (
                f"File not found: {shown_path}\n"
                f"Could not list parent directory "
                f"{workspace.display_path(parent)}: "
                f"{error}"
            )

        shown_parent = workspace.display_path(
            parent
        )

        if not entries:
            return (
                f"File not found: {shown_path}\n"
                f"Parent directory is empty: {shown_parent}"
            )

        visible = entries[
            :self.MAX_PARENT_ENTRIES
        ]

        listing = "\n".join(
            (
                f"  {entry.name}/"
                if entry.is_dir()
                else f"  {entry.name}"
            )
            for entry in visible
        )

        if len(entries) > len(visible):
            listing += (
                "\n"
                f"  ... +{len(entries) - len(visible)} more"
            )

        return (
            f"File not found: {shown_path}\n"
            f"Parent directory {shown_parent} contains:\n"
            f"{listing}"
        )

    def _missing_pattern_message(
        self,
        value: str,
    ) -> str:
        workspace = self.context.workspace

        base, _ = self._split_glob(
            value
        )

        try:
            directory = workspace.resolve_path(
                base
            )

        except Exception:
            return (
                f"No files matched pattern: {value}"
            )

        shown_directory = workspace.display_path(
            directory
        )

        if not directory.exists():
            return (
                f"No files matched pattern: {value}\n"
                f"Search directory does not exist: "
                f"{shown_directory}"
            )

        if not directory.is_dir():
            return (
                f"No files matched pattern: {value}\n"
                f"Search root is not a directory: "
                f"{shown_directory}"
            )

        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda entry: (
                    not entry.is_dir(),
                    entry.name.lower(),
                ),
            )

        except OSError as error:
            return (
                f"No files matched pattern: {value}\n"
                f"Could not list search directory "
                f"{shown_directory}: "
                f"{error}"
            )

        if not entries:
            return (
                f"No files matched pattern: {value}\n"
                f"Search directory is empty: "
                f"{shown_directory}"
            )

        visible = entries[
            :self.MAX_PARENT_ENTRIES
        ]

        listing = "\n".join(
            (
                f"  {entry.name}/"
                if entry.is_dir()
                else f"  {entry.name}"
            )
            for entry in visible
        )

        if len(entries) > len(visible):
            listing += (
                "\n"
                f"  ... +{len(entries) - len(visible)} more"
            )

        return (
            f"No files matched pattern: {value}\n"
            f"Directory {shown_directory} contains:\n"
            f"{listing}"
        )