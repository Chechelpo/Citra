from __future__ import annotations

from pathlib import Path
from typing import Any, override
import shutil

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ...utils.mermaid import (
    Mermaid,
    MermaidTheme,
)

from ..tool import Tool,ToolDefinition


class Diagram(Tool):
    """
    Create and maintain Mermaid diagrams for Citra documents.

    A project document named::

        architecture.citra.xml

    stores diagrams beneath::

        architecture.citra.assets/
            diagrams/
                runtime-flow.mmd
                runtime-flow.svg

    Library documents use the same layout beneath their library location::

        @library/python/
            architecture.citra.xml
            architecture.citra.assets/
                diagrams/
                    runtime-flow.mmd
                    runtime-flow.svg

    ``.mmd`` is the authoritative editable source.
    ``.svg`` is derived human-readable output.
    """

    DEFAULT_RENDER_TIMEOUT_SECONDS = 30
    TOOL_ID = "diagram"

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="diagram",
            description=(
                "Create, inspect, read, update, validate, render, list, "
                "and remove Mermaid diagrams associated with Citra documents. "
                "Diagrams are stored as editable Mermaid '.mmd' source and "
                "rendered SVG assets beneath "
                "'<document>.citra.assets/diagrams/' beside the owning "
                "document. Use 'location' to target either the active "
                "current project or Citra's persistent document library. "
                "Use simple Mermaid diagrams for architecture, flows, "
                "sequences, states, classes, ER models, timelines, Gantt "
                "charts, mindmaps, Git graphs, charts, and similar "
                "documentation. The returned Markdown reference can be "
                "inserted into a Citra document section."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="operation",
                        schema=JsonSchema.string(
                            description=(
                                "Operation to perform. Supported operations: "
                                "'create', 'update', 'read', 'inspect', "
                                "'list', 'validate', 'render', and 'remove'."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="document",
                        schema=JsonSchema.string(
                            description=(
                                "Logical Citra document name without the "
                                "'.citra.xml' suffix. For example, "
                                "'architecture' refers to "
                                "'architecture.citra.xml'."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="location",
                        schema=JsonSchema.string(
                            description=(
                                "Location containing the owning Citra document. "
                                "Defaults to '.'. Use '@library' or "
                                "'@library/<folder>' for persistent library "
                                "documents. A project-relative folder such as "
                                "'docs' is also supported."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="name",
                        schema=JsonSchema.string(
                            description=(
                                "Diagram identifier such as 'runtime-flow'. "
                                "Do not include '.mmd' or '.svg'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="source",
                        schema=JsonSchema.string(
                            description=(
                                "Mermaid source. Required for 'create' "
                                "and 'update'. A surrounding ```mermaid "
                                "Markdown fence is accepted but unnecessary."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="alt",
                        schema=JsonSchema.string(
                            description=(
                                "Concise human-readable alternative text "
                                "for the diagram. Used when returning the "
                                "Markdown image reference."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="render",
                        schema=JsonSchema.boolean(
                            description=(
                                "For 'create' and 'update', whether to render "
                                "the SVG immediately. Defaults to true."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="theme",
                        schema=JsonSchema.string(
                            description=(
                                "Optional Mermaid rendering theme. Supported "
                                "values are 'default', 'forest', 'dark', "
                                "and 'neutral'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="timeout",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum Mermaid rendering time in seconds. "
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

    _SUPPORTED_OPERATIONS = {
        "create",
        "update",
        "read",
        "inspect",
        "list",
        "validate",
        "render",
        "remove",
    }

    _ALLOWED_FIELDS = {
        "create": {
            "name",
            "source",
            "alt",
            "render",
            "theme",
            "timeout",
        },
        "update": {
            "name",
            "source",
            "alt",
            "render",
            "theme",
            "timeout",
        },
        "read": {
            "name",
        },
        "inspect": {
            "name",
            "alt",
        },
        "list": set(),
        "validate": {
            "name",
            "timeout",
        },
        "render": {
            "name",
            "alt",
            "theme",
            "timeout",
        },
        "remove": {
            "name",
        },
    }

    _REQUIRED_FIELDS = {
        "create": {
            "name",
            "source",
        },
        "update": {
            "name",
            "source",
        },
        "read": {
            "name",
        },
        "inspect": {
            "name",
        },
        "list": set(),
        "validate": {
            "name",
        },
        "render": {
            "name",
        },
        "remove": {
            "name",
        },
    }

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
        super().__init__(context)

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        self._validate_operation_arguments(
            arguments
        )

        operation = str(
            arguments["operation"]
        )

        document = self._normalize_document_name(
            str(
                arguments["document"]
            )
        )

        location = self._normalize_location(
            str(
                arguments.get(
                    "location",
                    ".",
                )
            )
        )

        self._require_document(
            document=document,
            location=location,
        )

        if operation == "list":
            return self._list(
                document=document,
                location=location,
            )

        name = Mermaid.validate_asset_name(
            str(
                arguments["name"]
            )
        )

        if operation == "create":
            return self._create(
                document=document,
                location=location,
                name=name,
                arguments=arguments,
            )

        if operation == "update":
            return self._update(
                document=document,
                location=location,
                name=name,
                arguments=arguments,
            )

        if operation == "read":
            return self._read(
                document=document,
                location=location,
                name=name,
            )

        if operation == "inspect":
            return self._inspect(
                document=document,
                location=location,
                name=name,
                alt=self._optional_string(
                    arguments.get(
                        "alt"
                    )
                ),
            )

        if operation == "validate":
            return self._validate(
                document=document,
                location=location,
                name=name,
                timeout=self._timeout(
                    arguments
                ),
            )

        if operation == "render":
            return self._render(
                document=document,
                location=location,
                name=name,
                alt=self._optional_string(
                    arguments.get(
                        "alt"
                    )
                ),
                theme=self._theme(
                    arguments.get(
                        "theme"
                    )
                ),
                timeout=self._timeout(
                    arguments
                ),
            )

        if operation == "remove":
            return self._remove(
                document=document,
                location=location,
                name=name,
            )

        raise ValueError(
            f"Unsupported diagram operation: "
            f"{operation!r}."
        )

    def _create(
        self,
        *,
        document: str,
        location: str,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        source_path = self._source_path(
            document=document,
            location=location,
            name=name,
        )

        rendered_path = self._rendered_path(
            document=document,
            location=location,
            name=name,
        )

        if (
            source_path.exists()
            or rendered_path.exists()
        ):
            raise FileExistsError(
                f"Diagram already exists: "
                f"{name!r}."
            )

        diagram = Mermaid(
            str(
                arguments["source"]
            )
        )

        return self._save(
            document=document,
            location=location,
            name=name,
            diagram=diagram,
            alt=self._optional_string(
                arguments.get(
                    "alt"
                )
            ),
            render=bool(
                arguments.get(
                    "render",
                    True,
                )
            ),
            theme=self._theme(
                arguments.get(
                    "theme"
                )
            ),
            timeout=self._timeout(
                arguments
            ),
            action="Created",
        )

    def _update(
        self,
        *,
        document: str,
        location: str,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        self._require_diagram(
            document=document,
            location=location,
            name=name,
        )

        diagram = Mermaid(
            str(
                arguments["source"]
            )
        )

        return self._save(
            document=document,
            location=location,
            name=name,
            diagram=diagram,
            alt=self._optional_string(
                arguments.get(
                    "alt"
                )
            ),
            render=bool(
                arguments.get(
                    "render",
                    True,
                )
            ),
            theme=self._theme(
                arguments.get(
                    "theme"
                )
            ),
            timeout=self._timeout(
                arguments
            ),
            action="Updated",
        )

    def _save(
        self,
        *,
        document: str,
        location: str,
        name: str,
        diagram: Mermaid,
        alt: str | None,
        render: bool,
        theme: MermaidTheme | None,
        timeout: int,
        action: str,
    ) -> str:
        directory = self._diagram_directory(
            document=document,
            location=location,
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        actual_alt = (
            alt
            or self._default_alt(
                name
            )
        )

        result = diagram.save_bundle(
            directory,
            name=name,
            alt=actual_alt,
            render=render,
            executable="mmdc",
            timeout=timeout,
            theme=theme,
            background="transparent",
        )

        lines = [
            f"{action} diagram {name!r}.",
            f"Document: {document}",
            f"Location: {location}",
            (
                "Type: "
                f"{diagram.diagram_type.value}"
            ),
            (
                "Source: "
                f"{self._display_path(result.source_path)}"
            ),
        ]

        if result.rendered_path is not None:
            lines.append(
                "Rendered: "
                f"{self._display_path(result.rendered_path)}"
            )

            lines.extend(
                (
                    "",
                    "Markdown:",
                    self._markdown_reference(
                        document=document,
                        name=name,
                        alt=actual_alt,
                    ),
                )
            )
        else:
            lines.extend(
                (
                    "Rendered: no",
                    "",
                    "Mermaid:",
                    diagram.to_markdown_fence().rstrip(),
                )
            )

        return "\n".join(
            lines
        )

    def _read(
        self,
        *,
        document: str,
        location: str,
        name: str,
    ) -> str:
        diagram = self._load_diagram(
            document=document,
            location=location,
            name=name,
        )

        info = diagram.inspect()

        return (
            f"Diagram: {name}\n"
            f"Document: {document}\n"
            f"Location: {location}\n"
            f"Type: {info.diagram_type.value}\n"
            f"Lines: {info.lines}\n"
            f"Digest: {info.digest[:16]}\n\n"
            f"{diagram.to_markdown_fence().rstrip()}"
        )

    def _inspect(
        self,
        *,
        document: str,
        location: str,
        name: str,
        alt: str | None,
    ) -> str:
        diagram = self._load_diagram(
            document=document,
            location=location,
            name=name,
        )

        info = diagram.inspect()

        source_path = self._source_path(
            document=document,
            location=location,
            name=name,
        )

        rendered_path = self._rendered_path(
            document=document,
            location=location,
            name=name,
        )

        actual_alt = (
            alt
            or self._default_alt(
                name
            )
        )

        lines = [
            f"Diagram: {name}",
            f"Document: {document}",
            f"Location: {location}",
            (
                "Type: "
                f"{info.diagram_type.value}"
            ),
            (
                "Declaration: "
                f"{info.declaration}"
            ),
            f"Lines: {info.lines}",
            f"Characters: {info.characters}",
            (
                "Digest: "
                f"{info.digest[:16]}"
            ),
            (
                "Source: "
                f"{self._display_path(source_path)}"
            ),
            (
                "Rendered: "
                + (
                    self._display_path(
                        rendered_path
                    )
                    if rendered_path.is_file()
                    else "(not rendered)"
                )
            ),
        ]

        if rendered_path.is_file():
            lines.extend(
                (
                    "",
                    "Markdown:",
                    self._markdown_reference(
                        document=document,
                        name=name,
                        alt=actual_alt,
                    ),
                )
            )

        return "\n".join(
            lines
        )

    def _validate(
        self,
        *,
        document: str,
        location: str,
        name: str,
        timeout: int,
    ) -> str:
        diagram = self._load_diagram(
            document=document,
            location=location,
            name=name,
        )

        diagram.validate_basic()

        if shutil.which(
            "mmdc"
        ) is None:
            return (
                f"Diagram {name!r} passed local validation.\n"
                f"Document: {document}\n"
                f"Location: {location}\n"
                "Mermaid CLI validation was not performed because "
                "'mmdc' is unavailable."
            )

        diagram.validate_with_cli(
            executable="mmdc",
            timeout=timeout,
        )

        return (
            f"Diagram {name!r} is valid.\n"
            f"Document: {document}\n"
            f"Location: {location}\n"
            "Local validation: ok\n"
            "Mermaid CLI validation: ok"
        )

    def _render(
        self,
        *,
        document: str,
        location: str,
        name: str,
        alt: str | None,
        theme: MermaidTheme | None,
        timeout: int,
    ) -> str:
        diagram = self._load_diagram(
            document=document,
            location=location,
            name=name,
        )

        directory = self._diagram_directory(
            document=document,
            location=location,
        )

        actual_alt = (
            alt
            or self._default_alt(
                name
            )
        )

        result = diagram.save_bundle(
            directory,
            name=name,
            alt=actual_alt,
            render=True,
            executable="mmdc",
            timeout=timeout,
            theme=theme,
            background="transparent",
        )

        if result.rendered_path is None:
            raise RuntimeError(
                "Mermaid bundle was rendered but no "
                "rendered path was returned."
            )

        return "\n".join(
            (
                f"Rendered diagram {name!r}.",
                f"Document: {document}",
                f"Location: {location}",
                (
                    "Source: "
                    f"{self._display_path(result.source_path)}"
                ),
                (
                    "Rendered: "
                    f"{self._display_path(result.rendered_path)}"
                ),
                "",
                "Markdown:",
                self._markdown_reference(
                    document=document,
                    name=name,
                    alt=actual_alt,
                ),
            )
        )

    def _remove(
        self,
        *,
        document: str,
        location: str,
        name: str,
    ) -> str:
        source_path = self._source_path(
            document=document,
            location=location,
            name=name,
        )

        rendered_path = self._rendered_path(
            document=document,
            location=location,
            name=name,
        )

        if (
            not source_path.exists()
            and not rendered_path.exists()
        ):
            raise FileNotFoundError(
                f"Diagram does not exist: "
                f"{name!r}."
            )

        removed: list[str] = []

        if source_path.is_file():
            source_path.unlink()

            removed.append(
                self._display_path(
                    source_path
                )
            )

        if rendered_path.is_file():
            rendered_path.unlink()

            removed.append(
                self._display_path(
                    rendered_path
                )
            )

        return (
            f"Removed diagram {name!r}.\n"
            f"Document: {document}\n"
            f"Location: {location}\n"
            + "\n".join(
                f"- {path}"
                for path in removed
            )
        )

    def _list(
        self,
        *,
        document: str,
        location: str,
    ) -> str:
        directory = self._diagram_directory(
            document=document,
            location=location,
        )

        if not directory.is_dir():
            return (
                f"Document: {document}\n"
                f"Location: {location}\n"
                "No diagrams."
            )

        names = sorted(
            {
                path.stem
                for path in directory.glob(
                    "*.mmd"
                )
                if path.is_file()
            }
        )

        if not names:
            return (
                f"Document: {document}\n"
                f"Location: {location}\n"
                "No diagrams."
            )

        lines = [
            f"Document: {document}",
            f"Location: {location}",
            f"Diagrams: {len(names)}",
            "",
        ]

        for name in names:
            try:
                diagram = Mermaid.load(
                    self._source_path(
                        document=document,
                        location=location,
                        name=name,
                    )
                )

                diagram_type = (
                    diagram.diagram_type.value
                )

            except Exception:
                diagram_type = "invalid"

            rendered = (
                self._rendered_path(
                    document=document,
                    location=location,
                    name=name,
                ).is_file()
            )

            lines.append(
                f"- {name} "
                f"[{diagram_type}] "
                + (
                    "[rendered]"
                    if rendered
                    else "[source only]"
                )
            )

        return "\n".join(
            lines
        )

    def _load_diagram(
        self,
        *,
        document: str,
        location: str,
        name: str,
    ) -> Mermaid:
        source_path = self._require_diagram(
            document=document,
            location=location,
            name=name,
        )

        return Mermaid.load(
            source_path
        )

    def _require_diagram(
        self,
        *,
        document: str,
        location: str,
        name: str,
    ) -> Path:
        source_path = self._source_path(
            document=document,
            location=location,
            name=name,
        )

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Diagram does not exist: "
                f"{name!r}."
            )

        return source_path

    def _require_document(
        self,
        *,
        document: str,
        location: str,
    ) -> Path:
        path = self._document_path(
            document=document,
            location=location,
        )

        if not path.is_file():
            raise FileNotFoundError(
                "Citra document does not exist: "
                f"{document!r} at {location!r}."
            )

        return path

    def _location_path(
        self,
        location: str,
    ) -> Path:
        if (
            location == "@library"
            or location.startswith(
                "@library/"
            )
        ):
            return (
                self.context.workspace.resolve_library_path(
                    location
                )
            )

        path = self.context.workspace.resolve_path(location)
        return self.context.workspace.require_writable_path(
            path
        )

    def _document_path(
        self,
        *,
        document: str,
        location: str,
    ) -> Path:
        return (
            self._location_path(
                location
            )
            / f"{document}.citra.xml"
        )

    def _diagram_directory(
        self,
        *,
        document: str,
        location: str,
    ) -> Path:
        directory = (
            self._location_path(
                location
            )
            / f"{document}.citra.assets"
            / "diagrams"
        )

        if (
            location == "@library"
            or location.startswith(
                "@library/"
            )
        ):
            return directory

        return self.context.workspace.require_writable_path(
            directory
        )

    def _source_path(
        self,
        *,
        document: str,
        location: str,
        name: str,
    ) -> Path:
        return (
            self._diagram_directory(
                document=document,
                location=location,
            )
            / f"{name}.mmd"
        )

    def _rendered_path(
        self,
        *,
        document: str,
        location: str,
        name: str,
    ) -> Path:
        return (
            self._diagram_directory(
                document=document,
                location=location,
            )
            / f"{name}.svg"
        )

    @staticmethod
    def _markdown_reference(
        *,
        document: str,
        name: str,
        alt: str,
    ) -> str:
        return Mermaid.markdown_image(
            (
                f"{document}.citra.assets/"
                f"diagrams/{name}.svg"
            ),
            alt=alt,
        )

    def _display_path(
        self,
        path: Path,
    ) -> str:
        return self.context.workspace.display_path(
            path
        )

    @classmethod
    def _normalize_document_name(
        cls,
        name: str,
    ) -> str:
        normalized = name.strip()

        if not normalized:
            raise ValueError(
                "Document name cannot be empty."
            )

        if (
            "/" in normalized
            or "\\" in normalized
            or "\n" in normalized
            or "\r" in normalized
            or "\x00" in normalized
        ):
            raise ValueError(
                "Document name must be a logical file name, "
                "not a path."
            )

        suffix = ".citra.xml"

        if normalized.endswith(
            suffix
        ):
            raise ValueError(
                "Pass the logical document name without "
                f"the {suffix!r} suffix."
            )

        if normalized in {
            ".",
            "..",
        }:
            raise ValueError(
                "Invalid document name."
            )

        return normalized

    @staticmethod
    def _normalize_location(
        location: str,
    ) -> str:
        normalized = location.strip()

        if not normalized:
            raise ValueError(
                "Document location cannot be empty."
            )

        if (
            "\n" in normalized
            or "\r" in normalized
            or "\x00" in normalized
            or "\\" in normalized
        ):
            raise ValueError(
                "Document location contains invalid characters."
            )

        if (
            normalized == "@library"
            or normalized.startswith(
                "@library/"
            )
        ):
            return normalized.rstrip(
                "/"
            )

        candidate = Path(normalized)
        if candidate.is_absolute() or ".." in candidate.parts or normalized.startswith("@"):
            raise ValueError(
                "Document location must stay inside the current project."
            )
        return normalized.rstrip("/")

    @classmethod
    def _validate_operation_arguments(
        cls,
        arguments: dict[str, Any],
    ) -> None:
        operation_value = arguments.get(
            "operation"
        )

        if operation_value is None:
            raise ValueError(
                "'operation' is required."
            )

        operation = str(
            operation_value
        )

        if operation not in cls._SUPPORTED_OPERATIONS:
            expected = ", ".join(
                repr(item)
                for item in sorted(
                    cls._SUPPORTED_OPERATIONS
                )
            )

            raise ValueError(
                f"Unsupported diagram operation "
                f"{operation!r}. "
                f"Expected one of: {expected}."
            )

        if "document" not in arguments:
            raise ValueError(
                "'document' is required."
            )

        required = cls._REQUIRED_FIELDS[
            operation
        ]

        missing = [
            field
            for field in sorted(
                required
            )
            if field not in arguments
        ]

        if missing:
            rendered = ", ".join(
                repr(field)
                for field in missing
            )

            raise ValueError(
                f"Operation {operation!r} is missing "
                f"required argument(s): {rendered}."
            )

        universally_allowed = {
            "operation",
            "document",
            "location",
        }

        unexpected = (
            set(
                arguments
            )
            - universally_allowed
            - cls._ALLOWED_FIELDS[
                operation
            ]
        )

        if unexpected:
            rendered = ", ".join(
                repr(field)
                for field in sorted(
                    unexpected
                )
            )

            raise ValueError(
                f"Argument(s) not valid for operation "
                f"{operation!r}: {rendered}."
            )

    @classmethod
    def _theme(
        cls,
        value: Any,
    ) -> MermaidTheme | None:
        if value is None:
            return None

        try:
            return MermaidTheme(
                str(
                    value
                )
            )
        except ValueError as error:
            allowed = ", ".join(
                repr(theme.value)
                for theme in MermaidTheme
            )

            raise ValueError(
                "Unsupported Mermaid theme "
                f"{value!r}. Expected one of: "
                f"{allowed}."
            ) from error

    @classmethod
    def _timeout(
        cls,
        arguments: dict[str, Any],
    ) -> int:
        timeout = int(
            arguments.get(
                "timeout",
                cls.DEFAULT_RENDER_TIMEOUT_SECONDS,
            )
        )

        if timeout <= 0:
            raise ValueError(
                "'timeout' must be greater than zero."
            )

        return timeout

    @staticmethod
    def _default_alt(
        name: str,
    ) -> str:
        words = (
            name
            .replace(
                "_",
                " ",
            )
            .replace(
                "-",
                " ",
            )
            .strip()
        )

        if not words:
            return "Diagram"

        return (
            words[0].upper()
            + words[1:]
        )

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        return str(
            value
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        operation = str(
            arguments.get(
                "operation",
                "?",
            )
        )

        document = str(
            arguments.get(
                "document",
                "?",
            )
        )

        location = str(
            arguments.get(
                "location",
                ".",
            )
        )

        parts = [
            f"operation={operation}",
            f"document={document!r}",
            f"location={location!r}",
        ]

        name = arguments.get(
            "name"
        )

        if name is not None:
            parts.append(
                f"diagram={name!r}"
            )

        source = arguments.get(
            "source"
        )

        if isinstance(
            source,
            str,
        ):
            parts.append(
                "source="
                f"{len(source.splitlines())} lines/"
                f"{len(source)} chars"
            )

        if arguments.get(
            "render"
        ) is False:
            parts.append(
                "render=false"
            )

        theme = arguments.get(
            "theme"
        )

        if theme is not None:
            parts.append(
                f"theme={theme!r}"
            )

        return " | ".join(
            parts
        )

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        text = str(
            result
        )

        if not text:
            return "empty result"

        return (
            f"{len(text.splitlines())} lines | "
            f"{len(text)} chars"
        )
