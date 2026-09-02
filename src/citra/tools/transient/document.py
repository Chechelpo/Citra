from __future__ import annotations

from hashlib import sha256
from html import escape
import os
from pathlib import Path
import re
import tempfile
from typing import Any, override
import xml.etree.ElementTree as ET

from markdown import markdown

from ...context import ExecutionContext
from ...utils.citra_doc import (
    CitraDoc,
    DEFAULT_LINES,
)
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)

from ..tool import Tool,ToolDefinition


class Document(Tool):
    """
    Read and edit structured Citra documents.

    Documents live either in the current project or in Citra's persistent
    document library. The model interacts with sections and Markdown and never
    needs direct XML access.

    Project document::

        ./architecture.citra.xml

    Library document::

        @library/architecture.citra.xml

    Nested library collections are supported through ``location``::

        @library/python/asyncio.citra.xml

    Every successful content mutation also regenerates a human-readable HTML
    companion beside the authoritative ``.citra.xml`` source.

    ``@library`` access is intentionally implemented directly by this trusted
    semantic tool. Generic filesystem tools do not need library authority.
    """
    TOOL_ID = "document"
    MAX_MUTATION_RESULT_CHARS = 8_000
    MAX_LIST_RESULTS = 200

    _ROOT_TAG = "citra-doc"
    _TITLE_TAG = "title"
    _VERSIONING_TAG = "versioning"
    _INDEX_TAG = "index"
    _ENTRY_TAG = "entry"
    _SECTIONS_TAG = "sections"
    _SECTION_TAG = "section"

    _PROJECT_ROOT = "."
    _LIBRARY_ALIAS = "@library"

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="document",
            description=(
                "Create, list, inspect, read, and edit structured Citra "
                "documents. Documents are stored as '<name>.citra.xml' and "
                "have an automatically generated '<name>.html' view. "
                "Use location='@library' for Citra's persistent document "
                "library, or omit location to use the current project. Nested "
                "project folders may use paths such as 'docs'; library folders "
                "use values such as '@library/python'. Pass the logical document name without "
                "the '.citra.xml' suffix. Use 'list' to discover documents, "
                "'inspect' before reading unfamiliar documents, and bounded "
                "section reads for large documents."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="operation",
                        schema=JsonSchema.string(
                            description=(
                                "Operation to perform. Supported operations: "
                                "'create', 'list', 'inspect', 'validate', "
                                "'read', 'create_section', 'update_section', "
                                "'remove_section', 'update_title', "
                                "'update_versioning', and 'render_html'."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="name",
                        schema=JsonSchema.string(
                            description=(
                                "Logical document name without the "
                                "'.citra.xml' suffix. Required for every "
                                "operation except 'list'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="location",
                        schema=JsonSchema.string(
                            description=(
                                "Document directory. Defaults to '.'. "
                                "Use '@library' for persistent documents. "
                                "Nested locations such as '@library/python' "
                                "and 'docs' are supported."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="recursive",
                        schema=JsonSchema.boolean(
                            description=(
                                "For 'list', recursively discover documents "
                                "below the selected location. Defaults to true."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="title",
                        schema=JsonSchema.string(
                            description=(
                                "Document title. Used only when creating "
                                "a new document."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="versioning",
                        schema=JsonSchema.string(
                            description=(
                                "Optional opaque version or freshness marker "
                                "used when creating a document."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="sections",
                        schema=JsonSchema.array(
                            JsonSchema.string(
                                description="Section name to read.",
                            ),
                            description=(
                                "Section names to read. Used only by the "
                                "'read' operation."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="section",
                        schema=JsonSchema.string(
                            description=(
                                "Section name used by section mutation "
                                "operations."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="parent",
                        schema=JsonSchema.string(
                            description=(
                                "Parent section under which a newly-created "
                                "section should appear. Omit to create the "
                                "section at the document root."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="content",
                        schema=JsonSchema.string(
                            description=(
                                "Markdown content. For 'create_section' this "
                                "is the initial section body. For "
                                "'update_section' it is replacement content. "
                                "For 'update_title' and 'update_versioning' "
                                "it is the new metadata value."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="from_line",
                        schema=JsonSchema.integer(
                            description=(
                                "Zero-based inclusive start line. Used by "
                                "'read' and 'update_section'. Defaults to 0."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="to_line",
                        schema=JsonSchema.integer(
                            description=(
                                "Zero-based exclusive end line. Omit when "
                                "updating to replace through the end of the "
                                "section. For reads, omission defaults to "
                                "a bounded read."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="expand_children",
                        schema=JsonSchema.boolean(
                            description=(
                                "When reading, also include indexed descendant "
                                "sections. Defaults to false."
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
        "list",
        "inspect",
        "validate",
        "read",
        "create_section",
        "update_section",
        "remove_section",
        "update_title",
        "update_versioning",
        "render_html",
    }

    _ALLOWED_FIELDS = {
        "create": {
            "title",
            "versioning",
        },
        "list": {
            "recursive",
        },
        "inspect": set(),
        "validate": set(),
        "read": {
            "sections",
            "from_line",
            "to_line",
            "expand_children",
        },
        "create_section": {
            "section",
            "content",
            "parent",
        },
        "update_section": {
            "section",
            "content",
            "from_line",
            "to_line",
        },
        "remove_section": {
            "section",
        },
        "update_title": {
            "content",
        },
        "update_versioning": {
            "content",
        },
        "render_html": set(),
    }

    _REQUIRED_FIELDS = {
        "create": {
            "name",
            "title",
        },
        "list": set(),
        "inspect": {
            "name",
        },
        "validate": {
            "name",
        },
        "read": {
            "name",
            "sections",
        },
        "create_section": {
            "name",
            "section",
            "content",
        },
        "update_section": {
            "name",
            "section",
            "content",
        },
        "remove_section": {
            "name",
            "section",
        },
        "update_title": {
            "name",
            "content",
        },
        "update_versioning": {
            "name",
            "content",
        },
        "render_html": {
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

        location = self._normalize_location(
            self._optional_string(
                arguments.get(
                    "location"
                )
            )
        )

        if operation == "list":
            return self._list_documents(
                location=location,
                recursive=bool(
                    arguments.get(
                        "recursive",
                        True,
                    )
                ),
            )

        name = self._normalize_document_name(
            str(arguments["name"])
        )

        if operation == "create":
            return self._create(
                name=name,
                location=location,
                arguments=arguments,
            )

        document = self._open_document(
            name=name,
            location=location,
        )

        if operation == "inspect":
            return self._inspect(
                name=name,
                location=location,
                document=document,
            )

        if operation == "validate":
            document.validate()

            return (
                f"Document {name!r} is valid.\n"
                f"File: {self._display_path(self._document_path(name, location))}"
            )

        if operation == "render_html":
            html_path = self._render_html_file(
                name=name,
                location=location,
                document=document,
            )

            return (
                f"Rendered HTML for document {name!r}.\n"
                f"File: {self._display_path(html_path)}"
            )

        if operation == "read":
            return self._read(
                document=document,
                arguments=arguments,
            )

        if operation == "create_section":
            result = document.create_section(
                name=str(
                    arguments["section"]
                ),
                content=str(
                    arguments["content"]
                ),
                parent=self._optional_string(
                    arguments.get(
                        "parent"
                    )
                ),
            )

            return self._mutation_with_html(
                name=name,
                location=location,
                document=document,
                result=result,
            )

        if operation == "update_section":
            result = document.update_section(
                name=str(
                    arguments["section"]
                ),
                new_content=str(
                    arguments["content"]
                ),
                from_line=int(
                    arguments.get(
                        "from_line",
                        0,
                    )
                ),
                to_line=self._optional_int(
                    arguments.get(
                        "to_line"
                    )
                ),
            )

            return self._mutation_with_html(
                name=name,
                location=location,
                document=document,
                result=result,
            )

        if operation == "remove_section":
            result = document.remove_section(
                str(
                    arguments["section"]
                )
            )

            return self._mutation_with_html(
                name=name,
                location=location,
                document=document,
                result=result,
            )

        if operation == "update_title":
            result = document.update_title(
                str(
                    arguments["content"]
                )
            )

            return self._mutation_with_html(
                name=name,
                location=location,
                document=document,
                result=result,
            )

        if operation == "update_versioning":
            result = document.update_versioning(
                str(
                    arguments["content"]
                )
            )

            return self._mutation_with_html(
                name=name,
                location=location,
                document=document,
                result=result,
            )

        raise ValueError(
            f"Unsupported document operation: {operation!r}."
        )

    def _create(
        self,
        *,
        name: str,
        location: str,
        arguments: dict[str, Any],
    ) -> str:
        path = self._document_path(
            name,
            location,
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        document = CitraDoc.create(
            path,
            title=str(
                arguments["title"]
            ),
            versioning=self._optional_string(
                arguments.get(
                    "versioning"
                )
            ),
        )

        versioning = document.read_versioning()

        lines = [
            f"Created document {name!r}.",
            f"File: {self._display_path(path)}",
            f"Title: {document.read_title()}",
        ]

        if versioning:
            lines.append(
                f"Versioning: {versioning}"
            )

        lines.extend(
            (
                "",
                document.read_index(),
            )
        )

        try:
            html_path = self._render_html_file(
                name=name,
                location=location,
                document=document,
            )
            lines.extend(
                (
                    "",
                    f"Rendered HTML: {self._display_path(html_path)}",
                )
            )
        except Exception as error:
            lines.extend(
                (
                    "",
                    "WARNING: document was created, but HTML rendering failed: "
                    f"{error}",
                )
            )

        return "\n".join(
            lines
        )

    def _list_documents(
        self,
        *,
        location: str,
        recursive: bool,
    ) -> str:
        directory = self._location_path(
            location
        )

        if not directory.exists():
            return (
                f"No documents found under {location}. "
                "The location does not exist."
            )

        if not directory.is_dir():
            raise NotADirectoryError(
                "Document location is not a directory: "
                f"{self._display_path(directory)}"
            )

        if (
            location == "@library"
            or location.startswith(
                "@library/"
            )
        ):
            paths = (
                self.context.workspace.list_library_documents(
                    location=location,
                    recursive=recursive,
                )
            )
        else:
            iterator = (
                directory.rglob(
                    "*.citra.xml"
                )
                if recursive
                else directory.glob(
                    "*.citra.xml"
                )
            )

            paths = tuple(
                sorted(
                    (
                        path
                        for path in iterator
                        if path.is_file()
                    ),
                    key=lambda path: (
                        self._display_path(
                            path
                        ).casefold()
                    ),
                )
            )

        if not paths:
            return (
                "No Citra documents found under "
                f"{self._display_path(directory)}."
            )

        shown = paths[
            :self.MAX_LIST_RESULTS
        ]

        lines = [
            f"Documents under {self._display_path(directory)}:"
        ]

        for path in shown:
            display = self._display_path(
                path
            )

            try:
                title = CitraDoc(
                    path
                ).read_title()
            except Exception as error:
                lines.append(
                    f"- {display} — invalid: {error}"
                )
                continue

            lines.append(
                f"- {display} — {title}"
            )

        remaining = (
            len(paths)
            - len(shown)
        )

        if remaining > 0:
            lines.append(
                f"... +{remaining} more document(s) not shown."
            )

        return "\n".join(
            lines
        )

    def _inspect(
        self,
        *,
        name: str,
        location: str,
        document: CitraDoc,
    ) -> str:
        title = document.read_title()
        versioning = document.read_versioning()
        index = document.read_index()

        path = self._document_path(
            name,
            location,
        )

        html_path = self._html_path(
            name,
            location,
        )

        size_bytes = path.stat().st_size

        html_status = (
            f"{self._display_path(html_path)} "
            f"({html_path.stat().st_size} bytes)"
            if html_path.is_file()
            else f"{self._display_path(html_path)} (not rendered)"
        )

        lines = [
            f"Document: {name}",
            f"Location: {location}",
            f"File: {self._display_path(path)}",
            f"HTML: {html_status}",
            f"Title: {title}",
            (
                f"Versioning: {versioning}"
                if versioning
                else "Versioning: (none)"
            ),
            f"Size: {size_bytes} bytes",
            "",
            index,
        ]

        return "\n".join(
            lines
        )

    def _read(
        self,
        *,
        document: CitraDoc,
        arguments: dict[str, Any],
    ) -> str:
        raw_sections = arguments[
            "sections"
        ]

        if not isinstance(
            raw_sections,
            list,
        ):
            raise ValueError(
                "'sections' must be an array."
            )

        if not raw_sections:
            raise ValueError(
                "'sections' must contain at least one section name."
            )

        sections = [
            str(section)
            for section in raw_sections
        ]

        from_line = int(
            arguments.get(
                "from_line",
                0,
            )
        )

        if "to_line" in arguments:
            to_line = self._optional_int(
                arguments.get(
                    "to_line"
                )
            )
        else:
            to_line = (
                from_line
                + DEFAULT_LINES
            )

        return document.read_sections(
            sections,
            from_line=from_line,
            to_line=to_line,
            expand_children=bool(
                arguments.get(
                    "expand_children",
                    False,
                )
            ),
        )

    def _open_document(
        self,
        *,
        name: str,
        location: str,
    ) -> CitraDoc:
        path = self._document_path(
            name,
            location,
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Document not found: {self._display_path(path)}"
            )

        return CitraDoc(
            path
        )

    def _document_path(
        self,
        name: str,
        location: str,
    ) -> Path:
        directory = self._location_path(
            location
        )

        path = (
            directory
            / f"{name}.citra.xml"
        ).resolve()

        self._require_within_location(
            location=location,
            path=path,
        )

        return path

    def _html_path(
        self,
        name: str,
        location: str,
    ) -> Path:
        directory = self._location_path(
            location
        )

        path = (
            directory
            / f"{name}.html"
        ).resolve()

        self._require_within_location(
            location=location,
            path=path,
        )

        return path

    def _location_path(
        self,
        location: str,
    ) -> Path:
        workspace = self.context.workspace

        if location == self._LIBRARY_ALIAS:
            return workspace.library.resolve()

        if location.startswith(
            f"{self._LIBRARY_ALIAS}/"
        ):
            remainder = location[
                len(self._LIBRARY_ALIAS) + 1:
            ]

            resolved = (
                workspace.library
                / remainder
            ).resolve()

            if not self._is_within(
                workspace.library.resolve(),
                resolved,
            ):
                raise ValueError(
                    "Document location escapes @library."
                )

            return resolved

        resolved = workspace.resolve_path(
            location
        )

        workspace_root = workspace.workspace.resolve()

        if not self._is_within(
            workspace_root,
            resolved,
        ):
            raise ValueError(
                "Document locations outside the project and @library "
                "are not supported."
            )

        return resolved

    def _require_within_location(
        self,
        *,
        location: str,
        path: Path,
    ) -> None:
        root = self._location_path(
            location
        )

        if not self._is_within(
            root,
            path,
        ):
            raise ValueError(
                "Document path escapes its selected location."
            )

    def _display_path(
        self,
        path: str | Path,
    ) -> str:
        resolved = Path(
            path
        ).resolve()

        library = self.context.workspace.library.resolve()
        workspace = self.context.workspace.workspace.resolve()

        if self._is_within(
            library,
            resolved,
        ):
            relative = resolved.relative_to(
                library
            )

            if not relative.parts:
                return self._LIBRARY_ALIAS

            return (
                f"{self._LIBRARY_ALIAS}/"
                f"{relative.as_posix()}"
            )

        if self._is_within(
            workspace,
            resolved,
        ):
            relative = resolved.relative_to(
                workspace
            )

            if not relative.parts:
                return self._PROJECT_ROOT

            return (
                f"./{relative.as_posix()}"
            )

        return self.context.workspace.display_path(
            resolved
        )

    def _mutation_with_html(
        self,
        *,
        name: str,
        location: str,
        document: CitraDoc,
        result: str,
    ) -> str:
        diagnostic = self._mutation_result(
            result
        )

        try:
            html_path = self._render_html_file(
                name=name,
                location=location,
                document=document,
            )
        except Exception as error:
            return (
                f"{diagnostic}\n"
                "WARNING: XML mutation succeeded, but HTML rendering failed: "
                f"{error}"
            )

        return (
            f"{diagnostic}\n"
            f"Rendered HTML: {self._display_path(html_path)}"
        )

    def _render_html_file(
        self,
        *,
        name: str,
        location: str,
        document: CitraDoc,
    ) -> Path:
        document.validate()

        xml_path = self._document_path(
            name,
            location,
        )

        tree = ET.parse(
            xml_path
        )

        html_content = self._build_html(
            tree.getroot()
        )

        html_path = self._html_path(
            name,
            location,
        )

        self._atomic_write_text(
            html_path,
            html_content,
        )

        return html_path

    # Plain (non-f-string) page script. Rendered HTML is a static file, so
    # all interactivity — theme selection, collapsible sections, and the
    # title/content search filter — is implemented client-side.
    _INTERACTIVE_SCRIPT = """
        (() => {
            const root = document.documentElement;
            const articles = Array.from(
                document.querySelectorAll('main > .document-section')
            );
            const collapsed = new Set();

            // --- Theme ---------------------------------------------------
            const THEME_KEY = 'citra-doc-theme';
            const themeButton = document.getElementById('theme-toggle');
            const applyTheme = value => {
                if (value === 'light' || value === 'dark') {
                    root.dataset.theme = value;
                } else {
                    delete root.dataset.theme;
                }
                if (themeButton) {
                    const label = {
                        auto: 'Auto',
                        light: 'Light',
                        dark: 'Dark',
                    }[value] || 'Auto';
                    themeButton.textContent = 'Theme: ' + label;
                    themeButton.setAttribute('aria-label', 'Color theme: ' + label);
                }
            };
            let storedTheme = null;
            try {
                storedTheme = localStorage.getItem(THEME_KEY);
            } catch (error) {
                storedTheme = null;
            }
            applyTheme(
                storedTheme === 'light' || storedTheme === 'dark'
                    ? storedTheme
                    : 'auto'
            );
            if (themeButton) {
                themeButton.addEventListener('click', () => {
                    const current = root.dataset.theme || 'auto';
                    const next = current === 'auto'
                        ? 'light'
                        : current === 'light' ? 'dark' : 'auto';
                    try {
                        localStorage.setItem(THEME_KEY, next);
                    } catch (error) {
                        // Private mode; theme simply does not persist.
                    }
                    applyTheme(next);
                });
            }

            // --- Collapsible sections --------------------------------------
            const ancestorsOf = new Map();
            {
                const stack = [];
                for (const article of articles) {
                    const depth = Number(article.dataset.depth);
                    while (
                        stack.length
                        && stack[stack.length - 1].depth >= depth
                    ) {
                        stack.pop();
                    }
                    ancestorsOf.set(
                        article.id,
                        stack.map(item => item.id)
                    );
                    stack.push({ depth: depth, id: article.id });
                }
            }

            const navItems = Array.from(
                document.querySelectorAll('.document-index li')
            );
            const searchInput = document.getElementById('document-search');
            const statusBox = document.getElementById('search-status');

            const syncToggleButtons = () => {
                document.querySelectorAll('.section-toggle').forEach(button => {
                    const expanded = !collapsed.has(button.dataset.target);
                    button.setAttribute('aria-expanded', String(expanded));
                    button.textContent = expanded ? '\\u2212' : '+';
                    button.setAttribute(
                        'aria-label',
                        expanded ? 'Collapse section' : 'Expand section'
                    );
                });
            };

            const applyState = () => {
                const query = searchInput
                    ? searchInput.value.trim().toLowerCase()
                    : '';
                const searching = query.length > 0;
                let visibleCount = 0;

                const indexRoot = document.querySelector('.document-index');
                if (indexRoot) {
                    indexRoot.classList.toggle('index-searching', searching);
                }

                for (const article of articles) {
                    const hiddenByCollapse = !searching
                        && ancestorsOf.get(article.id)
                            .some(id => collapsed.has(id));
                    const matches = !searching
                        || (article.dataset.searchText || '').includes(query);

                    article.classList.toggle('descendant-hidden', hiddenByCollapse);
                    article.classList.toggle('search-hidden', searching && !matches);

                    if ((searching && matches) || (!searching && !hiddenByCollapse)) {
                        visibleCount++;
                    }
                }

                for (const item of navItems) {
                    const link = item.querySelector(
                        ':scope > a[href^="#section-"]'
                    ) || item.querySelector(
                        ':scope > .index-row > a[href^="#section-"]'
                    );
                    let show = true;
                    if (link) {
                        const target = document.getElementById(
                            link.getAttribute('href').slice(1)
                        );
                        if (target) {
                            show = searching
                                ? !target.classList.contains('search-hidden')
                                : !ancestorsOf.get(target.id)
                                    .some(id => collapsed.has(id));
                        }
                    }
                    item.classList.toggle('nav-hidden', !show);
                }

                if (statusBox) {
                    statusBox.hidden = !searching;
                    statusBox.textContent = searching
                        ? visibleCount + ' matching section'
                            + (visibleCount === 1 ? '' : 's')
                        : '';
                }
            };

            document.querySelectorAll('.section-toggle').forEach(button => {
                button.addEventListener('click', () => {
                    const id = button.dataset.target;
                    if (!id) return;
                    if (collapsed.has(id)) {
                        collapsed.delete(id);
                    } else {
                        collapsed.add(id);
                    }
                    syncToggleButtons();
                    applyState();
                });
            });

            // --- Collapsible index tree -----------------------------------
            // Sidebar toggles fold only the index; section bodies are not
            // affected by them.
            document.querySelectorAll('.document-index .index-toggle').forEach(button => {
                button.addEventListener('click', () => {
                    const item = button.closest('li');
                    if (!item) return;
                    const expanded = !item.classList.toggle('nav-branch-collapsed');
                    button.setAttribute('aria-expanded', String(expanded));
                    button.textContent = expanded ? '\\u2212' : '+';
                    button.setAttribute(
                        'aria-label',
                        expanded ? 'Collapse subsections' : 'Expand subsections'
                    );
                });
            });

            // Opening an anchor inside collapsed sections expands ancestors.
            document.querySelectorAll('.document-index a[href^="#section-"]')
                .forEach(link => {
                    link.addEventListener('click', () => {
                        const id = link.getAttribute('href').slice(1);
                        let changed = false;
                        for (const ancestor of ancestorsOf.get(id) || []) {
                            if (collapsed.has(ancestor)) {
                                collapsed.delete(ancestor);
                                changed = true;
                            }
                        }
                        if (changed) {
                            syncToggleButtons();
                            applyState();
                        }
                    });
                });

            if (searchInput) {
                searchInput.addEventListener('input', applyState);
            }
        })();
    """

    @classmethod
    def _build_html(
        cls,
        root: ET.Element,
    ) -> str:
        if root.tag != cls._ROOT_TAG:
            raise ValueError(
                f"Expected <{cls._ROOT_TAG}> root, got <{root.tag}>."
            )

        title_element = cls._required_direct_child(
            root,
            cls._TITLE_TAG,
        )

        versioning_element = cls._optional_direct_child(
            root,
            cls._VERSIONING_TAG,
        )

        index_element = cls._required_direct_child(
            root,
            cls._INDEX_TAG,
        )

        sections_element = cls._required_direct_child(
            root,
            cls._SECTIONS_TAG,
        )

        title = title_element.text or ""
        versioning = (
            ""
            if versioning_element is None
            else versioning_element.text or ""
        )

        sections_by_name: dict[str, ET.Element] = {}

        for section in sections_element:
            if section.tag != cls._SECTION_TAG:
                continue

            section_name = section.get(
                "name"
            )

            if section_name is None:
                raise ValueError(
                    "<section> is missing required 'name' attribute."
                )

            sections_by_name[
                section_name
            ] = section

        navigation, ordered_sections = cls._render_index_html(
            index_element
        )

        # A section has children when the immediately following entry in
        # pre-order is one level deeper.
        sections_with_children: set[str] = {
            entry[0]
            for position, entry in enumerate(ordered_sections)
            if position + 1 < len(ordered_sections)
            and entry[1] + 1 == ordered_sections[position + 1][1]
        }

        rendered_sections: list[str] = []

        for section_name, depth, ancestors in ordered_sections:
            section = sections_by_name.get(
                section_name
            )

            if section is None:
                raise ValueError(
                    f"Index entry {section_name!r} has no matching section."
                )

            body = markdown(
                section.text or "",
                extensions=(
                    "fenced_code",
                    "tables",
                    "sane_lists",
                ),
                output_format="html",
            )

            section_id = cls._section_anchor(
                section_name
            )

            heading_level = min(
                2 + depth,
                6,
            )

            escaped_name = escape(
                section_name
            )

            if ancestors:
                breadcrumb_parts = [
                    escape(item)
                    for item in ancestors
                ] + [
                    escaped_name
                ]

                breadcrumb = (
                    '<div class="breadcrumb">'
                    + '<span class="breadcrumb-separator"> / </span>'.join(
                        breadcrumb_parts
                    )
                    + "</div>"
                )
            else:
                breadcrumb = ""

            if section_name in sections_with_children:
                toggle_button = (
                    '<button type="button" class="section-toggle" '
                    f'data-target="{section_id}" '
                    'aria-expanded="true" '
                    'aria-label="Collapse section">'
                    "&minus;</button>"
                )
                heading_row_class = "section-heading-row"
            else:
                toggle_button = ""
                heading_row_class = "section-heading-row section-heading-row-single"

            # Raw Markdown plus the section name; searched case-insensitively
            # by the generated page's client-side filter.
            search_text = escape(
                f"{section_name}\n{section.text or ''}".casefold(),
            )

            rendered_sections.append(
                "\n".join(
                    (
                        (
                            '<article class="document-section" '
                            f'id="{section_id}" data-depth="{depth}" '
                            f'data-search-text="{search_text}">'
                        ),
                        '<header class="section-header">',
                        breadcrumb,
                        f'<div class="{heading_row_class}">',
                        toggle_button,
                        (
                            f"<h{heading_level}>"
                            f"{escaped_name}"
                            f"</h{heading_level}>"
                        ),
                        "</div>",
                        "</header>",
                        '<div class="section-body">',
                        body,
                        "</div>",
                        (
                            '<a class="back-to-top" href="#document-top">'
                            "Back to contents"
                            "</a>"
                        ),
                        "</article>",
                    )
                )
            )

        sections_html = (
            "\n".join(
                rendered_sections
            )
            if rendered_sections
            else (
                '<section class="empty-document">'
                "<h2>No sections yet</h2>"
                "<p>This document does not contain any sections.</p>"
                "</section>"
            )
        )

        escaped_title = escape(
            title
        )

        if versioning:
            version_html = (
                '<div class="versioning">'
                '<span class="versioning-label">Version</span>'
                f'<span class="versioning-value">{escape(versioning)}</span>'
                "</div>"
            )
        else:
            version_html = (
                '<div class="versioning versioning-empty">'
                "No version marker"
                "</div>"
            )

        return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="generator" content="Citra Document">
    <title>{escaped_title}</title>
    <style>
        :root {{
            color-scheme: light;
            --background: #ffffff;
            --sidebar-background: #f6f8fa;
            --text: #1f2328;
            --muted: #656d76;
            --border: #d0d7de;
            --accent: #0969da;
            --accent-soft: #ddf4ff;
            --code-background: #f6f8fa;
        }}
        :root[data-theme="dark"] {{
            color-scheme: dark;
            --background: #0d1117;
            --sidebar-background: #161b22;
            --text: #e6edf3;
            --muted: #8b949e;
            --border: #30363d;
            --accent: #58a6ff;
            --accent-soft: #13233a;
            --code-background: #161b22;
        }}
        @media (prefers-color-scheme: dark) {{
            :root:not([data-theme="light"]) {{
                color-scheme: dark;
                --background: #0d1117;
                --sidebar-background: #161b22;
                --text: #e6edf3;
                --muted: #8b949e;
                --border: #30363d;
                --accent: #58a6ff;
                --accent-soft: #13233a;
                --code-background: #161b22;
            }}
        }}
        * {{ box-sizing: border-box; }}
        html {{ scroll-behavior: smooth; background: var(--background); }}
        body {{
            margin: 0;
            color: var(--text);
            background: var(--background);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 16px;
            line-height: 1.65;
        }}
        a {{ color: var(--accent); }}
        .page {{
            display: grid;
            grid-template-columns: minmax(230px, 300px) minmax(0, 920px);
            gap: 56px;
            width: min(1320px, 100%);
            margin: 0 auto;
            padding: 44px 32px 72px;
        }}
        .sidebar {{
            position: sticky;
            top: 24px;
            align-self: start;
            max-height: calc(100vh - 48px);
            overflow-y: auto;
            padding: 22px;
            border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--sidebar-background);
        }}
        .document-title {{ margin: 0 0 8px; font-size: 1.45rem; }}
        .versioning {{
            display: flex;
            gap: 8px;
            margin-bottom: 26px;
            color: var(--muted);
            font-size: 0.82rem;
        }}
        .versioning-label {{ font-weight: 650; }}
        .versioning-empty {{ font-style: italic; }}
        .contents-title {{
            margin: 0 0 10px;
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }}
        .document-index, .document-index ul {{
            list-style: none;
            margin: 0;
            padding: 0;
        }}
        .document-index ul {{
            margin: 5px 0 5px 12px;
            padding-left: 12px;
            border-left: 1px solid var(--border);
        }}
        .index-row {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .index-row-single {{ gap: 0; }}
        .index-toggle {{
            flex: none;
            width: 18px;
            height: 18px;
            margin-left: -24px;
            padding: 0;
            border: 0;
            border-radius: 4px;
            background: none;
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1;
            cursor: pointer;
        }}
        .index-toggle:hover {{
            color: var(--accent);
        }}
        /* Index toggles fold only the sidebar tree; section bodies in main
           are never hidden by them. */
        .document-index li.nav-branch-collapsed > ul {{
            display: none;
        }}
        /* While searching, matches must remain reachable even inside a
           folded branch. */
        .document-index.index-searching li.nav-branch-collapsed > ul {{
            display: block;
        }}
        .document-index a {{
            display: block;
            padding: 5px 7px;
            border-radius: 6px;
            color: var(--text);
            font-size: 0.9rem;
            text-decoration: none;
            overflow-wrap: anywhere;
        }}
        .document-index a:hover,
        .document-index a.active {{
            background: var(--accent-soft);
            color: var(--accent);
        }}
        main {{ min-width: 0; padding-top: 4px; }}
        .document-section {{
            scroll-margin-top: 24px;
            margin: 0 0 58px;
            padding: 0 0 52px;
            border-bottom: 1px solid var(--border);
        }}
        .section-header {{ margin-bottom: 20px; }}
        .section-heading-row {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section-heading-row-single {{ gap: 0; }}
        .section-toggle {{
            flex: none;
            width: 26px;
            height: 26px;
            padding: 0;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--sidebar-background);
            color: var(--muted);
            font-size: 1rem;
            line-height: 1;
            cursor: pointer;
        }}
        .section-toggle:hover {{
            border-color: var(--accent);
            color: var(--accent);
        }}
        .document-section.descendant-hidden,
        .document-section.search-hidden,
        .document-index li.nav-hidden {{
            display: none;
        }}
        .search-box {{
            display: block;
            margin-bottom: 8px;
        }}
        .search-label {{
            display: block;
            margin-bottom: 4px;
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }}
        .search-box input {{
            width: 100%;
            padding: 7px 10px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--background);
            color: var(--text);
            font-size: 0.88rem;
        }}
        .search-box input:focus {{
            outline: none;
            border-color: var(--accent);
        }}
        .search-status {{
            margin-bottom: 10px;
            color: var(--accent);
            font-size: 0.8rem;
        }}
        .breadcrumb {{
            margin-bottom: 7px;
            color: var(--muted);
            font-size: 0.82rem;
        }}
        pre {{
            max-width: 100%;
            overflow-x: auto;
            padding: 16px;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--code-background);
        }}
        code {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco,
                Consolas, "Liberation Mono", monospace;
            font-size: 0.9em;
        }}
        table {{
            display: block;
            width: max-content;
            max-width: 100%;
            overflow-x: auto;
            border-spacing: 0;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 8px 12px;
            border: 1px solid var(--border);
            text-align: left;
            vertical-align: top;
        }}
        img {{ max-width: 100%; height: auto; }}
        .sidebar-heading-row {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 10px;
        }}
        #theme-toggle {{
            flex: none;
            margin-top: 4px;
            padding: 4px 9px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--background);
            color: var(--muted);
            font-size: 0.75rem;
            cursor: pointer;
        }}
        #theme-toggle:hover {{
            border-color: var(--accent);
            color: var(--accent);
        }}
        /* Mermaid SVGs are rendered with a fixed light theme; recolor them
           in dark mode so black strokes stay visible. */
        :root[data-theme="dark"] .section-body img[src$=".svg"] {{
            filter: invert(0.92) hue-rotate(180deg);
        }}
        @media (prefers-color-scheme: dark) {{
            :root:not([data-theme="light"]) .section-body img[src$=".svg"] {{
                filter: invert(0.92) hue-rotate(180deg);
            }}
        }}
        .back-to-top {{
            display: inline-block;
            margin-top: 28px;
            color: var(--muted);
            font-size: 0.8rem;
            text-decoration: none;
        }}
        @media (max-width: 820px) {{
            .page {{ display: block; padding: 22px 18px 48px; }}
            .sidebar {{ position: static; max-height: none; margin-bottom: 42px; }}
        }}
        @media print {{
            .page {{ display: block; width: 100%; padding: 0; }}
            .sidebar {{ position: static; max-height: none; border: 0; padding: 0; }}
            .back-to-top {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="page" id="document-top">
        <aside class="sidebar" aria-label="Document contents">
            <div class="sidebar-heading-row">
                <h1 class="document-title">{escaped_title}</h1>
                <button
                    type="button"
                    id="theme-toggle"
                    aria-label="Color theme: Auto"
                >Theme: Auto</button>
            </div>
            {version_html}
            <label class="search-box">
                <span class="search-label">Search</span>
                <input
                    type="search"
                    id="document-search"
                    placeholder="Titles and content…"
                    autocomplete="off"
                >
            </label>
            <div class="search-status" id="search-status" hidden></div>
            <div class="contents-title">Contents</div>
            {navigation}
        </aside>
        <main>
            {sections_html}
        </main>
    </div>
    <script>
        (() => {{
            const links = Array.from(
                document.querySelectorAll('.document-index a[href^="#section-"]')
            );
            const sections = links
                .map(link => document.querySelector(link.getAttribute('href')))
                .filter(Boolean);
            if (!links.length || !sections.length || !('IntersectionObserver' in window)) {{
                return;
            }}
            const byId = new Map(
                links.map(link => [link.getAttribute('href').slice(1), link])
            );
            const activate = id => {{
                links.forEach(link => link.classList.remove('active'));
                const active = byId.get(id);
                if (active) active.classList.add('active');
            }};
            const observer = new IntersectionObserver(
                entries => {{
                    const visible = entries
                        .filter(entry => entry.isIntersecting)
                        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
                    if (visible.length) activate(visible[0].target.id);
                }},
                {{ rootMargin: '-12% 0px -70% 0px', threshold: 0 }}
            );
            sections.forEach(section => observer.observe(section));
        }})();
    </script>
    <script>
{cls._INTERACTIVE_SCRIPT}
    </script>
</body>
</html>
"""

    @classmethod
    def _render_index_html(
        cls,
        index: ET.Element,
    ) -> tuple[
        str,
        list[tuple[str, int, tuple[str, ...]]],
    ]:
        root_entries = [
            child
            for child in index
            if child.tag == cls._ENTRY_TAG
        ]

        if not root_entries:
            return (
                '<p class="empty-index">No sections.</p>',
                [],
            )

        ordered: list[
            tuple[str, int, tuple[str, ...]]
        ] = []

        items = [
            cls._render_index_entry_html(
                entry,
                depth=0,
                ancestors=(),
                ordered=ordered,
            )
            for entry in root_entries
        ]

        return (
            '<ul class="document-index">'
            + "".join(items)
            + "</ul>",
            ordered,
        )

    @classmethod
    def _render_index_entry_html(
        cls,
        entry: ET.Element,
        *,
        depth: int,
        ancestors: tuple[str, ...],
        ordered: list[tuple[str, int, tuple[str, ...]]],
    ) -> str:
        name = entry.get(
            "name"
        )

        if name is None:
            raise ValueError(
                "<entry> is missing required 'name' attribute."
            )

        ordered.append(
            (
                name,
                depth,
                ancestors,
            )
        )

        anchor = cls._section_anchor(
            name
        )

        children = [
            child
            for child in entry
            if child.tag == cls._ENTRY_TAG
        ]

        nested = ""
        has_children = bool(children)

        if children:
            child_ancestors = (
                *ancestors,
                name,
            )

            nested_items = [
                cls._render_index_entry_html(
                    child,
                    depth=depth + 1,
                    ancestors=child_ancestors,
                    ordered=ordered,
                )
                for child in children
            ]

            nested = (
                "<ul>"
                + "".join(nested_items)
                + "</ul>"
            )

        # Sidebar branches get their own toggle. It only folds this entry's
        # nested list inside the index; section bodies are never affected.
        if has_children:
            index_toggle = (
                '<button type="button" class="index-toggle" '
                'aria-expanded="true" '
                'aria-label="Collapse subsections">'
                "&minus;</button>"
            )
            row_class = "index-row"
        else:
            index_toggle = ""
            row_class = "index-row index-row-single"

        return (
            f'<li data-depth="{depth}">'
            f'<div class="{row_class}">'
            f"{index_toggle}"
            f'<a href="#{anchor}">{escape(name)}</a>'
            "</div>"
            f"{nested}"
            "</li>"
        )

    @staticmethod
    def _section_anchor(
        name: str,
    ) -> str:
        slug = re.sub(
            r"[^a-z0-9]+",
            "-",
            name.casefold(),
        ).strip(
            "-"
        )

        if not slug:
            slug = "section"

        slug = slug[:48].rstrip(
            "-"
        ) or "section"

        digest = sha256(
            name.encode(
                "utf-8"
            )
        ).hexdigest()[:10]

        return (
            f"section-{slug}-{digest}"
        )

    @staticmethod
    def _required_direct_child(
        parent: ET.Element,
        tag: str,
    ) -> ET.Element:
        matches = [
            child
            for child in parent
            if child.tag == tag
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one <{tag}> inside <{parent.tag}>."
            )

        return matches[0]

    @staticmethod
    def _optional_direct_child(
        parent: ET.Element,
        tag: str,
    ) -> ET.Element | None:
        matches = [
            child
            for child in parent
            if child.tag == tag
        ]

        if len(matches) > 1:
            raise ValueError(
                f"Expected at most one <{tag}> inside <{parent.tag}>."
            )

        if not matches:
            return None

        return matches[0]

    @staticmethod
    def _atomic_write_text(
        path: Path,
        content: str,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )

        temporary_path = Path(
            temporary_name
        )

        try:
            with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="\n",
            ) as stream:
                stream.write(
                    content
                )
                stream.flush()
                os.fsync(
                    stream.fileno()
                )

            os.replace(
                temporary_path,
                path,
            )

        except Exception:
            temporary_path.unlink(
                missing_ok=True
            )
            raise

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
            "\n" in normalized
            or "\r" in normalized
            or "\x00" in normalized
        ):
            raise ValueError(
                "Document name contains invalid characters."
            )

        if (
            "/" in normalized
            or "\\" in normalized
        ):
            raise ValueError(
                "Document name must be a file name, not a path."
            )

        if normalized in {
            ".",
            "..",
        }:
            raise ValueError(
                "Invalid document name."
            )

        suffix = ".citra.xml"

        if normalized.endswith(
            suffix
        ):
            raise ValueError(
                "Pass the logical document name without "
                f"the {suffix!r} suffix."
            )

        return normalized

    @classmethod
    def _normalize_location(
        cls,
        location: str | None,
    ) -> str:
        if location is None:
            return cls._PROJECT_ROOT

        normalized = location.strip()

        normalized = normalized.rstrip(
            "/"
        )

        if not normalized:
            raise ValueError(
                "Document location cannot be empty."
            )

        if "\\" in normalized:
            raise ValueError(
                "Document locations must use forward slashes."
            )

        if (
            normalized == cls._LIBRARY_ALIAS
            or normalized.startswith(
                f"{cls._LIBRARY_ALIAS}/"
            )
        ):
            return normalized

        candidate = Path(normalized)
        if candidate.is_absolute() or ".." in candidate.parts or normalized.startswith("@"):
            raise ValueError(
                "Document location must stay inside the current project."
            )
        return normalized


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
            rendered = ", ".join(
                repr(item)
                for item in sorted(
                    cls._SUPPORTED_OPERATIONS
                )
            )

            raise ValueError(
                f"Unsupported document operation {operation!r}. "
                f"Expected one of: {rendered}."
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
            "name",
            "location",
        }

        operation_allowed = cls._ALLOWED_FIELDS[
            operation
        ]

        unexpected = (
            set(arguments)
            - universally_allowed
            - operation_allowed
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

        if (
            operation == "read"
            and not arguments.get(
                "sections"
            )
        ):
            raise ValueError(
                "'sections' must contain at least one section name."
            )

    def _mutation_result(
        self,
        result: str,
    ) -> str:
        if len(
            result
        ) <= self.MAX_MUTATION_RESULT_CHARS:
            return result

        truncated_chars = (
            len(result)
            - self.MAX_MUTATION_RESULT_CHARS
        )

        return (
            result[
                :self.MAX_MUTATION_RESULT_CHARS
            ]
            + "\n"
            + (
                "... mutation diagnostic truncated "
                f"({truncated_chars} additional chars)"
            )
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

        parts = [
            f"operation={operation}",
        ]

        name = arguments.get(
            "name"
        )

        if name is not None:
            parts.append(
                f"document={name!r}"
            )

        location = arguments.get(
            "location"
        )

        parts.append(
            f"location={location or self._PROJECT_ROOT!r}"
        )

        if "recursive" in arguments:
            parts.append(
                f"recursive={bool(arguments['recursive'])}"
            )

        section = arguments.get(
            "section"
        )

        if section is not None:
            parts.append(
                f"section={section!r}"
            )

        sections = arguments.get(
            "sections"
        )

        if isinstance(
            sections,
            list,
        ):
            preview = ", ".join(
                repr(str(item))
                for item in sections[:5]
            )

            remaining = (
                len(sections)
                - 5
            )

            if remaining > 0:
                preview += (
                    f", +{remaining} more"
                )

            parts.append(
                f"sections=[{preview}]"
            )

        parent = arguments.get(
            "parent"
        )

        if parent is not None:
            parts.append(
                f"parent={parent!r}"
            )

        if "from_line" in arguments:
            parts.append(
                f"from_line={arguments['from_line']}"
            )

        if "to_line" in arguments:
            parts.append(
                f"to_line={arguments['to_line']}"
            )

        content = arguments.get(
            "content"
        )

        if isinstance(
            content,
            str,
        ):
            parts.append(
                "content="
                f"{len(content.splitlines())} lines/"
                f"{len(content)} chars"
            )

        title = arguments.get(
            "title"
        )

        if isinstance(
            title,
            str,
        ):
            parts.append(
                f"title={title!r}"
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

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        return str(
            value
        )

    @staticmethod
    def _optional_int(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        return int(
            value
        )

    @staticmethod
    def _is_within(
        root: Path,
        path: Path,
    ) -> bool:
        try:
            path.resolve().relative_to(
                root.resolve()
            )
        except ValueError:
            return False

        return True
