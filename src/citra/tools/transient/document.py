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
from ...utils.citra_doc import CitraDoc, DEFAULT_LINES
from ...utils.json_schema import ChatCompletionTool, FunctionDefinition, JsonProperty, JsonSchema
from ..tool import Tool, ToolDefinition

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
    TOOL_ID = 'document'
    MAX_MUTATION_RESULT_CHARS = 8000
    MAX_LIST_RESULTS = 200
    _ROOT_TAG = 'citra-doc'
    _TITLE_TAG = 'title'
    _VERSIONING_TAG = 'versioning'
    _INDEX_TAG = 'index'
    _ENTRY_TAG = 'entry'
    _SECTIONS_TAG = 'sections'
    _SECTION_TAG = 'section'
    _PROJECT_ROOT = '.'
    _LIBRARY_ALIAS = '@library'
    DEFINITION = ChatCompletionTool(function=FunctionDefinition(name='document', description="Create, list, inspect, read, and edit structured Citra documents. Documents are stored as '<name>.citra.xml' and have an automatically generated '<name>.html' view. Use location='@library' for Citra's persistent document library, or omit location to use the current project. Nested project folders may use paths such as 'docs'; library folders use values such as '@library/python'. Pass the logical document name without the '.citra.xml' suffix. Use 'list' to discover documents, 'inspect' before reading unfamiliar documents, and bounded section reads for large documents.", parameters=JsonSchema.object(properties=(JsonProperty(name='operation', schema=JsonSchema.string(description="Operation to perform. Supported operations: 'create', 'list', 'inspect', 'validate', 'read', 'create_section', 'update_section', 'remove_section', 'update_title', 'update_versioning', and 'render_html'.")), JsonProperty(name='name', schema=JsonSchema.string(description="Logical document name without the '.citra.xml' suffix. Required for every operation except 'list'."), required=False), JsonProperty(name='location', schema=JsonSchema.string(description="Document directory. Defaults to '.'. Use '@library' for persistent documents. Nested locations such as '@library/python' and 'docs' are supported."), required=False), JsonProperty(name='recursive', schema=JsonSchema.boolean(description="For 'list', recursively discover documents below the selected location. Defaults to true."), required=False), JsonProperty(name='title', schema=JsonSchema.string(description='Document title. Used only when creating a new document.'), required=False), JsonProperty(name='versioning', schema=JsonSchema.string(description='Optional opaque version or freshness marker used when creating a document.'), required=False), JsonProperty(name='sections', schema=JsonSchema.array(JsonSchema.string(description='Section name to read.'), description="Section names to read. Used only by the 'read' operation."), required=False), JsonProperty(name='section', schema=JsonSchema.string(description='Section name used by section mutation operations.'), required=False), JsonProperty(name='parent', schema=JsonSchema.string(description='Parent section under which a newly-created section should appear. Omit to create the section at the document root.'), required=False), JsonProperty(name='content', schema=JsonSchema.string(description="Markdown content. For 'create_section' this is the initial section body. For 'update_section' it is replacement content. For 'update_title' and 'update_versioning' it is the new metadata value."), required=False), JsonProperty(name='from_line', schema=JsonSchema.integer(description="Zero-based inclusive start line. Used by 'read' and 'update_section'. Defaults to 0."), required=False), JsonProperty(name='to_line', schema=JsonSchema.integer(description='Zero-based exclusive end line. Omit when updating to replace through the end of the section. For reads, omission defaults to a bounded read.'), required=False), JsonProperty(name='expand_children', schema=JsonSchema.boolean(description='When reading, also include indexed descendant sections. Defaults to false.'), required=False)), additional_properties=False)))
    _SUPPORTED_OPERATIONS = {'create', 'list', 'inspect', 'validate', 'read', 'create_section', 'update_section', 'remove_section', 'update_title', 'update_versioning', 'render_html'}
    _ALLOWED_FIELDS = {'create': {'title', 'versioning'}, 'list': {'recursive'}, 'inspect': set(), 'validate': set(), 'read': {'sections', 'from_line', 'to_line', 'expand_children'}, 'create_section': {'section', 'content', 'parent'}, 'update_section': {'section', 'content', 'from_line', 'to_line'}, 'remove_section': {'section'}, 'update_title': {'content'}, 'update_versioning': {'content'}, 'render_html': set()}
    _REQUIRED_FIELDS = {'create': {'name', 'title'}, 'list': set(), 'inspect': {'name'}, 'validate': {'name'}, 'read': {'name', 'sections'}, 'create_section': {'name', 'section', 'content'}, 'update_section': {'name', 'section', 'content'}, 'remove_section': {'name', 'section'}, 'update_title': {'name', 'content'}, 'update_versioning': {'name', 'content'}, 'render_html': {'name'}}

    @classmethod
    @override
    def definitions_for_context(cls, context: ExecutionContext) -> tuple[ToolDefinition, ...]:
        """Handle definitions for context."""
        del context
        return (ToolDefinition(definition=cls.DEFINITION),)

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize the instance."""
        super().__init__(context)

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        """Execute the execute operation."""
        self._validate_operation_arguments(arguments)
        operation = str(arguments['operation'])
        location = self._normalize_location(self._optional_string(arguments.get('location')))
        if operation == 'list':
            return self._list_documents(location=location, recursive=bool(arguments.get('recursive', True)))
        name = self._normalize_document_name(str(arguments['name']))
        if operation == 'create':
            return self._create(name=name, location=location, arguments=arguments)
        document = self._open_document(name=name, location=location)
        if operation == 'inspect':
            return self._inspect(name=name, location=location, document=document)
        if operation == 'validate':
            document.validate()
            return f'Document {name!r} is valid.\nFile: {self._display_path(self._document_path(name, location))}'
        if operation == 'render_html':
            html_path = self._render_html_file(name=name, location=location, document=document)
            return f'Rendered HTML for document {name!r}.\nFile: {self._display_path(html_path)}'
        if operation == 'read':
            return self._read(document=document, arguments=arguments)
        if operation == 'create_section':
            result = document.create_section(name=str(arguments['section']), content=str(arguments['content']), parent=self._optional_string(arguments.get('parent')))
            return self._mutation_with_html(name=name, location=location, document=document, result=result)
        if operation == 'update_section':
            result = document.update_section(name=str(arguments['section']), new_content=str(arguments['content']), from_line=int(arguments.get('from_line', 0)), to_line=self._optional_int(arguments.get('to_line')))
            return self._mutation_with_html(name=name, location=location, document=document, result=result)
        if operation == 'remove_section':
            result = document.remove_section(str(arguments['section']))
            return self._mutation_with_html(name=name, location=location, document=document, result=result)
        if operation == 'update_title':
            result = document.update_title(str(arguments['content']))
            return self._mutation_with_html(name=name, location=location, document=document, result=result)
        if operation == 'update_versioning':
            result = document.update_versioning(str(arguments['content']))
            return self._mutation_with_html(name=name, location=location, document=document, result=result)
        raise ValueError(f'Unsupported document operation: {operation!r}.')

    def _create(self, *, name: str, location: str, arguments: dict[str, Any]) -> str:
        """Handle create."""
        path = self._document_path(name, location)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = CitraDoc.create(path, title=str(arguments['title']), versioning=self._optional_string(arguments.get('versioning')))
        versioning = document.read_versioning()
        lines = [f'Created document {name!r}.', f'File: {self._display_path(path)}', f'Title: {document.read_title()}']
        if versioning:
            lines.append(f'Versioning: {versioning}')
        lines.extend(('', document.read_index()))
        try:
            html_path = self._render_html_file(name=name, location=location, document=document)
            lines.extend(('', f'Rendered HTML: {self._display_path(html_path)}'))
        except Exception as error:
            lines.extend(('', f'WARNING: document was created, but HTML rendering failed: {error}'))
        return '\n'.join(lines)

    def _list_documents(self, *, location: str, recursive: bool) -> str:
        """Handle list documents."""
        directory = self._location_path(location)
        if not directory.exists():
            return f'No documents found under {location}. The location does not exist.'
        if not directory.is_dir():
            raise NotADirectoryError(f'Document location is not a directory: {self._display_path(directory)}')
        if location == '@library' or location.startswith('@library/'):
            paths = self.context.workspace.list_library_documents(location=location, recursive=recursive)
        else:
            iterator = directory.rglob('*.citra.xml') if recursive else directory.glob('*.citra.xml')
            paths = tuple(sorted((path for path in iterator if path.is_file()), key=lambda path: self._display_path(path).casefold()))
        if not paths:
            return f'No Citra documents found under {self._display_path(directory)}.'
        shown = paths[:self.MAX_LIST_RESULTS]
        lines = [f'Documents under {self._display_path(directory)}:']
        for path in shown:
            display = self._display_path(path)
            try:
                title = CitraDoc(path).read_title()
            except Exception as error:
                lines.append(f'- {display} — invalid: {error}')
                continue
            lines.append(f'- {display} — {title}')
        remaining = len(paths) - len(shown)
        if remaining > 0:
            lines.append(f'... +{remaining} more document(s) not shown.')
        return '\n'.join(lines)

    def _inspect(self, *, name: str, location: str, document: CitraDoc) -> str:
        """Handle inspect."""
        title = document.read_title()
        versioning = document.read_versioning()
        index = document.read_index()
        path = self._document_path(name, location)
        html_path = self._html_path(name, location)
        size_bytes = path.stat().st_size
        html_status = f'{self._display_path(html_path)} ({html_path.stat().st_size} bytes)' if html_path.is_file() else f'{self._display_path(html_path)} (not rendered)'
        lines = [f'Document: {name}', f'Location: {location}', f'File: {self._display_path(path)}', f'HTML: {html_status}', f'Title: {title}', f'Versioning: {versioning}' if versioning else 'Versioning: (none)', f'Size: {size_bytes} bytes', '', index]
        return '\n'.join(lines)

    def _read(self, *, document: CitraDoc, arguments: dict[str, Any]) -> str:
        """Handle read."""
        raw_sections = arguments['sections']
        if not isinstance(raw_sections, list):
            raise ValueError("'sections' must be an array.")
        if not raw_sections:
            raise ValueError("'sections' must contain at least one section name.")
        sections = [str(section) for section in raw_sections]
        from_line = int(arguments.get('from_line', 0))
        if 'to_line' in arguments:
            to_line = self._optional_int(arguments.get('to_line'))
        else:
            to_line = from_line + DEFAULT_LINES
        return document.read_sections(sections, from_line=from_line, to_line=to_line, expand_children=bool(arguments.get('expand_children', False)))

    def _open_document(self, *, name: str, location: str) -> CitraDoc:
        """Handle open document."""
        path = self._document_path(name, location)
        if not path.is_file():
            raise FileNotFoundError(f'Document not found: {self._display_path(path)}')
        return CitraDoc(path)

    def _document_path(self, name: str, location: str) -> Path:
        """Handle document path."""
        directory = self._location_path(location)
        path = (directory / f'{name}.citra.xml').resolve()
        self._require_within_location(location=location, path=path)
        return path

    def _html_path(self, name: str, location: str) -> Path:
        """Handle html path."""
        directory = self._location_path(location)
        path = (directory / f'{name}.html').resolve()
        self._require_within_location(location=location, path=path)
        return path

    def _location_path(self, location: str) -> Path:
        """Handle location path."""
        workspace = self.context.workspace
        if location == self._LIBRARY_ALIAS:
            return workspace.library.resolve()
        if location.startswith(f'{self._LIBRARY_ALIAS}/'):
            remainder = location[len(self._LIBRARY_ALIAS) + 1:]
            resolved = (workspace.library / remainder).resolve()
            if not self._is_within(workspace.library.resolve(), resolved):
                raise ValueError('Document location escapes @library.')
            return resolved
        resolved = workspace.resolve_path(location)
        workspace_root = workspace.workspace.resolve()
        if not self._is_within(workspace_root, resolved):
            raise ValueError('Document locations outside the project and @library are not supported.')
        return resolved

    def _require_within_location(self, *, location: str, path: Path) -> None:
        """Handle require within location."""
        root = self._location_path(location)
        if not self._is_within(root, path):
            raise ValueError('Document path escapes its selected location.')

    def _display_path(self, path: str | Path) -> str:
        """Handle display path."""
        resolved = Path(path).resolve()
        library = self.context.workspace.library.resolve()
        workspace = self.context.workspace.workspace.resolve()
        if self._is_within(library, resolved):
            relative = resolved.relative_to(library)
            if not relative.parts:
                return self._LIBRARY_ALIAS
            return f'{self._LIBRARY_ALIAS}/{relative.as_posix()}'
        if self._is_within(workspace, resolved):
            relative = resolved.relative_to(workspace)
            if not relative.parts:
                return self._PROJECT_ROOT
            return f'./{relative.as_posix()}'
        return self.context.workspace.display_path(resolved)

    def _mutation_with_html(self, *, name: str, location: str, document: CitraDoc, result: str) -> str:
        """Handle mutation with html."""
        diagnostic = self._mutation_result(result)
        try:
            html_path = self._render_html_file(name=name, location=location, document=document)
        except Exception as error:
            return f'{diagnostic}\nWARNING: XML mutation succeeded, but HTML rendering failed: {error}'
        return f'{diagnostic}\nRendered HTML: {self._display_path(html_path)}'

    def _render_html_file(self, *, name: str, location: str, document: CitraDoc) -> Path:
        """Handle render html file."""
        document.validate()
        xml_path = self._document_path(name, location)
        tree = ET.parse(xml_path)
        html_content = self._build_html(tree.getroot())
        html_path = self._html_path(name, location)
        self._atomic_write_text(html_path, html_content)
        return html_path
    _INTERACTIVE_SCRIPT = '\n        (() => {\n            const root = document.documentElement;\n            const articles = Array.from(\n                document.querySelectorAll(\'main > .document-section\')\n            );\n            const collapsed = new Set();\n\n            // --- Theme ---------------------------------------------------\n            const THEME_KEY = \'citra-doc-theme\';\n            const themeButton = document.getElementById(\'theme-toggle\');\n            const applyTheme = value => {\n                if (value === \'light\' || value === \'dark\') {\n                    root.dataset.theme = value;\n                } else {\n                    delete root.dataset.theme;\n                }\n                if (themeButton) {\n                    const label = {\n                        auto: \'Auto\',\n                        light: \'Light\',\n                        dark: \'Dark\',\n                    }[value] || \'Auto\';\n                    themeButton.textContent = \'Theme: \' + label;\n                    themeButton.setAttribute(\'aria-label\', \'Color theme: \' + label);\n                }\n            };\n            let storedTheme = null;\n            try {\n                storedTheme = localStorage.getItem(THEME_KEY);\n            } catch (error) {\n                storedTheme = null;\n            }\n            applyTheme(\n                storedTheme === \'light\' || storedTheme === \'dark\'\n                    ? storedTheme\n                    : \'auto\'\n            );\n            if (themeButton) {\n                themeButton.addEventListener(\'click\', () => {\n                    const current = root.dataset.theme || \'auto\';\n                    const next = current === \'auto\'\n                        ? \'light\'\n                        : current === \'light\' ? \'dark\' : \'auto\';\n                    try {\n                        localStorage.setItem(THEME_KEY, next);\n                    } catch (error) {\n                        // Private mode; theme simply does not persist.\n                    }\n                    applyTheme(next);\n                });\n            }\n\n            // --- Collapsible sections --------------------------------------\n            const ancestorsOf = new Map();\n            {\n                const stack = [];\n                for (const article of articles) {\n                    const depth = Number(article.dataset.depth);\n                    while (\n                        stack.length\n                        && stack[stack.length - 1].depth >= depth\n                    ) {\n                        stack.pop();\n                    }\n                    ancestorsOf.set(\n                        article.id,\n                        stack.map(item => item.id)\n                    );\n                    stack.push({ depth: depth, id: article.id });\n                }\n            }\n\n            const navItems = Array.from(\n                document.querySelectorAll(\'.document-index li\')\n            );\n            const searchInput = document.getElementById(\'document-search\');\n            const statusBox = document.getElementById(\'search-status\');\n\n            const syncToggleButtons = () => {\n                document.querySelectorAll(\'.section-toggle\').forEach(button => {\n                    const expanded = !collapsed.has(button.dataset.target);\n                    button.setAttribute(\'aria-expanded\', String(expanded));\n                    button.textContent = expanded ? \'\\u2212\' : \'+\';\n                    button.setAttribute(\n                        \'aria-label\',\n                        expanded ? \'Collapse section\' : \'Expand section\'\n                    );\n                });\n            };\n\n            const applyState = () => {\n                const query = searchInput\n                    ? searchInput.value.trim().toLowerCase()\n                    : \'\';\n                const searching = query.length > 0;\n                let visibleCount = 0;\n\n                const indexRoot = document.querySelector(\'.document-index\');\n                if (indexRoot) {\n                    indexRoot.classList.toggle(\'index-searching\', searching);\n                }\n\n                for (const article of articles) {\n                    const hiddenByCollapse = !searching\n                        && ancestorsOf.get(article.id)\n                            .some(id => collapsed.has(id));\n                    const matches = !searching\n                        || (article.dataset.searchText || \'\').includes(query);\n\n                    article.classList.toggle(\'descendant-hidden\', hiddenByCollapse);\n                    article.classList.toggle(\'search-hidden\', searching && !matches);\n\n                    if ((searching && matches) || (!searching && !hiddenByCollapse)) {\n                        visibleCount++;\n                    }\n                }\n\n                for (const item of navItems) {\n                    const link = item.querySelector(\n                        \':scope > a[href^="#section-"]\'\n                    ) || item.querySelector(\n                        \':scope > .index-row > a[href^="#section-"]\'\n                    );\n                    let show = true;\n                    if (link) {\n                        const target = document.getElementById(\n                            link.getAttribute(\'href\').slice(1)\n                        );\n                        if (target) {\n                            show = searching\n                                ? !target.classList.contains(\'search-hidden\')\n                                : !ancestorsOf.get(target.id)\n                                    .some(id => collapsed.has(id));\n                        }\n                    }\n                    item.classList.toggle(\'nav-hidden\', !show);\n                }\n\n                if (statusBox) {\n                    statusBox.hidden = !searching;\n                    statusBox.textContent = searching\n                        ? visibleCount + \' matching section\'\n                            + (visibleCount === 1 ? \'\' : \'s\')\n                        : \'\';\n                }\n            };\n\n            document.querySelectorAll(\'.section-toggle\').forEach(button => {\n                button.addEventListener(\'click\', () => {\n                    const id = button.dataset.target;\n                    if (!id) return;\n                    if (collapsed.has(id)) {\n                        collapsed.delete(id);\n                    } else {\n                        collapsed.add(id);\n                    }\n                    syncToggleButtons();\n                    applyState();\n                });\n            });\n\n            // --- Collapsible index tree -----------------------------------\n            // Sidebar toggles fold only the index; section bodies are not\n            // affected by them.\n            document.querySelectorAll(\'.document-index .index-toggle\').forEach(button => {\n                button.addEventListener(\'click\', () => {\n                    const item = button.closest(\'li\');\n                    if (!item) return;\n                    const expanded = !item.classList.toggle(\'nav-branch-collapsed\');\n                    button.setAttribute(\'aria-expanded\', String(expanded));\n                    button.textContent = expanded ? \'\\u2212\' : \'+\';\n                    button.setAttribute(\n                        \'aria-label\',\n                        expanded ? \'Collapse subsections\' : \'Expand subsections\'\n                    );\n                });\n            });\n\n            // Opening an anchor inside collapsed sections expands ancestors.\n            document.querySelectorAll(\'.document-index a[href^="#section-"]\')\n                .forEach(link => {\n                    link.addEventListener(\'click\', () => {\n                        const id = link.getAttribute(\'href\').slice(1);\n                        let changed = false;\n                        for (const ancestor of ancestorsOf.get(id) || []) {\n                            if (collapsed.has(ancestor)) {\n                                collapsed.delete(ancestor);\n                                changed = true;\n                            }\n                        }\n                        if (changed) {\n                            syncToggleButtons();\n                            applyState();\n                        }\n                    });\n                });\n\n            if (searchInput) {\n                searchInput.addEventListener(\'input\', applyState);\n            }\n        })();\n    '

    @classmethod
    def _build_html(cls, root: ET.Element) -> str:
        """Handle build html."""
        if root.tag != cls._ROOT_TAG:
            raise ValueError(f'Expected <{cls._ROOT_TAG}> root, got <{root.tag}>.')
        title_element = cls._required_direct_child(root, cls._TITLE_TAG)
        versioning_element = cls._optional_direct_child(root, cls._VERSIONING_TAG)
        index_element = cls._required_direct_child(root, cls._INDEX_TAG)
        sections_element = cls._required_direct_child(root, cls._SECTIONS_TAG)
        title = title_element.text or ''
        versioning = '' if versioning_element is None else versioning_element.text or ''
        sections_by_name: dict[str, ET.Element] = {}
        for section in sections_element:
            if section.tag != cls._SECTION_TAG:
                continue
            section_name = section.get('name')
            if section_name is None:
                raise ValueError("<section> is missing required 'name' attribute.")
            sections_by_name[section_name] = section
        navigation, ordered_sections = cls._render_index_html(index_element)
        sections_with_children: set[str] = {entry[0] for position, entry in enumerate(ordered_sections) if position + 1 < len(ordered_sections) and entry[1] + 1 == ordered_sections[position + 1][1]}
        rendered_sections: list[str] = []
        for section_name, depth, ancestors in ordered_sections:
            section = sections_by_name.get(section_name)
            if section is None:
                raise ValueError(f'Index entry {section_name!r} has no matching section.')
            body = markdown(section.text or '', extensions=('fenced_code', 'tables', 'sane_lists'), output_format='html')
            section_id = cls._section_anchor(section_name)
            heading_level = min(2 + depth, 6)
            escaped_name = escape(section_name)
            if ancestors:
                breadcrumb_parts = [escape(item) for item in ancestors] + [escaped_name]
                breadcrumb = '<div class="breadcrumb">' + '<span class="breadcrumb-separator"> / </span>'.join(breadcrumb_parts) + '</div>'
            else:
                breadcrumb = ''
            if section_name in sections_with_children:
                toggle_button = f'<button type="button" class="section-toggle" data-target="{section_id}" aria-expanded="true" aria-label="Collapse section">&minus;</button>'
                heading_row_class = 'section-heading-row'
            else:
                toggle_button = ''
                heading_row_class = 'section-heading-row section-heading-row-single'
            search_text = escape(f"{section_name}\n{section.text or ''}".casefold())
            rendered_sections.append('\n'.join((f'<article class="document-section" id="{section_id}" data-depth="{depth}" data-search-text="{search_text}">', '<header class="section-header">', breadcrumb, f'<div class="{heading_row_class}">', toggle_button, f'<h{heading_level}>{escaped_name}</h{heading_level}>', '</div>', '</header>', '<div class="section-body">', body, '</div>', '<a class="back-to-top" href="#document-top">Back to contents</a>', '</article>')))
        sections_html = '\n'.join(rendered_sections) if rendered_sections else '<section class="empty-document"><h2>No sections yet</h2><p>This document does not contain any sections.</p></section>'
        escaped_title = escape(title)
        if versioning:
            version_html = f'<div class="versioning"><span class="versioning-label">Version</span><span class="versioning-value">{escape(versioning)}</span></div>'
        else:
            version_html = '<div class="versioning versioning-empty">No version marker</div>'
        return f"""<!doctype html>\n<html lang="en">\n<head>\n    <meta charset="utf-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1">\n    <meta name="generator" content="Citra Document">\n    <title>{escaped_title}</title>\n    <style>\n        :root {{\n            color-scheme: light;\n            --background: #ffffff;\n            --sidebar-background: #f6f8fa;\n            --text: #1f2328;\n            --muted: #656d76;\n            --border: #d0d7de;\n            --accent: #0969da;\n            --accent-soft: #ddf4ff;\n            --code-background: #f6f8fa;\n        }}\n        :root[data-theme="dark"] {{\n            color-scheme: dark;\n            --background: #0d1117;\n            --sidebar-background: #161b22;\n            --text: #e6edf3;\n            --muted: #8b949e;\n            --border: #30363d;\n            --accent: #58a6ff;\n            --accent-soft: #13233a;\n            --code-background: #161b22;\n        }}\n        @media (prefers-color-scheme: dark) {{\n            :root:not([data-theme="light"]) {{\n                color-scheme: dark;\n                --background: #0d1117;\n                --sidebar-background: #161b22;\n                --text: #e6edf3;\n                --muted: #8b949e;\n                --border: #30363d;\n                --accent: #58a6ff;\n                --accent-soft: #13233a;\n                --code-background: #161b22;\n            }}\n        }}\n        * {{ box-sizing: border-box; }}\n        html {{ scroll-behavior: smooth; background: var(--background); }}\n        body {{\n            margin: 0;\n            color: var(--text);\n            background: var(--background);\n            font-family: Inter, ui-sans-serif, system-ui, -apple-system,\n                BlinkMacSystemFont, "Segoe UI", sans-serif;\n            font-size: 16px;\n            line-height: 1.65;\n        }}\n        a {{ color: var(--accent); }}\n        .page {{\n            display: grid;\n            grid-template-columns: minmax(230px, 300px) minmax(0, 920px);\n            gap: 56px;\n            width: min(1320px, 100%);\n            margin: 0 auto;\n            padding: 44px 32px 72px;\n        }}\n        .sidebar {{\n            position: sticky;\n            top: 24px;\n            align-self: start;\n            max-height: calc(100vh - 48px);\n            overflow-y: auto;\n            padding: 22px;\n            border: 1px solid var(--border);\n            border-radius: 12px;\n            background: var(--sidebar-background);\n        }}\n        .document-title {{ margin: 0 0 8px; font-size: 1.45rem; }}\n        .versioning {{\n            display: flex;\n            gap: 8px;\n            margin-bottom: 26px;\n            color: var(--muted);\n            font-size: 0.82rem;\n        }}\n        .versioning-label {{ font-weight: 650; }}\n        .versioning-empty {{ font-style: italic; }}\n        .contents-title {{\n            margin: 0 0 10px;\n            color: var(--muted);\n            font-size: 0.75rem;\n            font-weight: 700;\n            letter-spacing: 0.09em;\n            text-transform: uppercase;\n        }}\n        .document-index, .document-index ul {{\n            list-style: none;\n            margin: 0;\n            padding: 0;\n        }}\n        .document-index ul {{\n            margin: 5px 0 5px 12px;\n            padding-left: 12px;\n            border-left: 1px solid var(--border);\n        }}\n        .index-row {{\n            display: flex;\n            align-items: center;\n            gap: 4px;\n        }}\n        .index-row-single {{ gap: 0; }}\n        .index-toggle {{\n            flex: none;\n            width: 18px;\n            height: 18px;\n            margin-left: -24px;\n            padding: 0;\n            border: 0;\n            border-radius: 4px;\n            background: none;\n            color: var(--muted);\n            font-size: 0.95rem;\n            line-height: 1;\n            cursor: pointer;\n        }}\n        .index-toggle:hover {{\n            color: var(--accent);\n        }}\n        /* Index toggles fold only the sidebar tree; section bodies in main\n           are never hidden by them. */\n        .document-index li.nav-branch-collapsed > ul {{\n            display: none;\n        }}\n        /* While searching, matches must remain reachable even inside a\n           folded branch. */\n        .document-index.index-searching li.nav-branch-collapsed > ul {{\n            display: block;\n        }}\n        .document-index a {{\n            display: block;\n            padding: 5px 7px;\n            border-radius: 6px;\n            color: var(--text);\n            font-size: 0.9rem;\n            text-decoration: none;\n            overflow-wrap: anywhere;\n        }}\n        .document-index a:hover,\n        .document-index a.active {{\n            background: var(--accent-soft);\n            color: var(--accent);\n        }}\n        main {{ min-width: 0; padding-top: 4px; }}\n        .document-section {{\n            scroll-margin-top: 24px;\n            margin: 0 0 58px;\n            padding: 0 0 52px;\n            border-bottom: 1px solid var(--border);\n        }}\n        .section-header {{ margin-bottom: 20px; }}\n        .section-heading-row {{\n            display: flex;\n            align-items: center;\n            gap: 10px;\n        }}\n        .section-heading-row-single {{ gap: 0; }}\n        .section-toggle {{\n            flex: none;\n            width: 26px;\n            height: 26px;\n            padding: 0;\n            border: 1px solid var(--border);\n            border-radius: 6px;\n            background: var(--sidebar-background);\n            color: var(--muted);\n            font-size: 1rem;\n            line-height: 1;\n            cursor: pointer;\n        }}\n        .section-toggle:hover {{\n            border-color: var(--accent);\n            color: var(--accent);\n        }}\n        .document-section.descendant-hidden,\n        .document-section.search-hidden,\n        .document-index li.nav-hidden {{\n            display: none;\n        }}\n        .search-box {{\n            display: block;\n            margin-bottom: 8px;\n        }}\n        .search-label {{\n            display: block;\n            margin-bottom: 4px;\n            color: var(--muted);\n            font-size: 0.75rem;\n            font-weight: 700;\n            letter-spacing: 0.09em;\n            text-transform: uppercase;\n        }}\n        .search-box input {{\n            width: 100%;\n            padding: 7px 10px;\n            border: 1px solid var(--border);\n            border-radius: 6px;\n            background: var(--background);\n            color: var(--text);\n            font-size: 0.88rem;\n        }}\n        .search-box input:focus {{\n            outline: none;\n            border-color: var(--accent);\n        }}\n        .search-status {{\n            margin-bottom: 10px;\n            color: var(--accent);\n            font-size: 0.8rem;\n        }}\n        .breadcrumb {{\n            margin-bottom: 7px;\n            color: var(--muted);\n            font-size: 0.82rem;\n        }}\n        pre {{\n            max-width: 100%;\n            overflow-x: auto;\n            padding: 16px;\n            border: 1px solid var(--border);\n            border-radius: 8px;\n            background: var(--code-background);\n        }}\n        code {{\n            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco,\n                Consolas, "Liberation Mono", monospace;\n            font-size: 0.9em;\n        }}\n        table {{\n            display: block;\n            width: max-content;\n            max-width: 100%;\n            overflow-x: auto;\n            border-spacing: 0;\n            border-collapse: collapse;\n        }}\n        th, td {{\n            padding: 8px 12px;\n            border: 1px solid var(--border);\n            text-align: left;\n            vertical-align: top;\n        }}\n        img {{ max-width: 100%; height: auto; }}\n        .sidebar-heading-row {{\n            display: flex;\n            align-items: flex-start;\n            justify-content: space-between;\n            gap: 10px;\n        }}\n        #theme-toggle {{\n            flex: none;\n            margin-top: 4px;\n            padding: 4px 9px;\n            border: 1px solid var(--border);\n            border-radius: 6px;\n            background: var(--background);\n            color: var(--muted);\n            font-size: 0.75rem;\n            cursor: pointer;\n        }}\n        #theme-toggle:hover {{\n            border-color: var(--accent);\n            color: var(--accent);\n        }}\n        /* Mermaid SVGs are rendered with a fixed light theme; recolor them\n           in dark mode so black strokes stay visible. */\n        :root[data-theme="dark"] .section-body img[src$=".svg"] {{\n            filter: invert(0.92) hue-rotate(180deg);\n        }}\n        @media (prefers-color-scheme: dark) {{\n            :root:not([data-theme="light"]) .section-body img[src$=".svg"] {{\n                filter: invert(0.92) hue-rotate(180deg);\n            }}\n        }}\n        .back-to-top {{\n            display: inline-block;\n            margin-top: 28px;\n            color: var(--muted);\n            font-size: 0.8rem;\n            text-decoration: none;\n        }}\n        @media (max-width: 820px) {{\n            .page {{ display: block; padding: 22px 18px 48px; }}\n            .sidebar {{ position: static; max-height: none; margin-bottom: 42px; }}\n        }}\n        @media print {{\n            .page {{ display: block; width: 100%; padding: 0; }}\n            .sidebar {{ position: static; max-height: none; border: 0; padding: 0; }}\n            .back-to-top {{ display: none; }}\n        }}\n    </style>\n</head>\n<body>\n    <div class="page" id="document-top">\n        <aside class="sidebar" aria-label="Document contents">\n            <div class="sidebar-heading-row">\n                <h1 class="document-title">{escaped_title}</h1>\n                <button\n                    type="button"\n                    id="theme-toggle"\n                    aria-label="Color theme: Auto"\n                >Theme: Auto</button>\n            </div>\n            {version_html}\n            <label class="search-box">\n                <span class="search-label">Search</span>\n                <input\n                    type="search"\n                    id="document-search"\n                    placeholder="Titles and content…"\n                    autocomplete="off"\n                >\n            </label>\n            <div class="search-status" id="search-status" hidden></div>\n            <div class="contents-title">Contents</div>\n            {navigation}\n        </aside>\n        <main>\n            {sections_html}\n        </main>\n    </div>\n    <script>\n        (() => {{\n            const links = Array.from(\n                document.querySelectorAll('.document-index a[href^="#section-"]')\n            );\n            const sections = links\n                .map(link => document.querySelector(link.getAttribute('href')))\n                .filter(Boolean);\n            if (!links.length || !sections.length || !('IntersectionObserver' in window)) {{\n                return;\n            }}\n            const byId = new Map(\n                links.map(link => [link.getAttribute('href').slice(1), link])\n            );\n            const activate = id => {{\n                links.forEach(link => link.classList.remove('active'));\n                const active = byId.get(id);\n                if (active) active.classList.add('active');\n            }};\n            const observer = new IntersectionObserver(\n                entries => {{\n                    const visible = entries\n                        .filter(entry => entry.isIntersecting)\n                        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);\n                    if (visible.length) activate(visible[0].target.id);\n                }},\n                {{ rootMargin: '-12% 0px -70% 0px', threshold: 0 }}\n            );\n            sections.forEach(section => observer.observe(section));\n        }})();\n    </script>\n    <script>\n{cls._INTERACTIVE_SCRIPT}\n    </script>\n</body>\n</html>\n"""

    @classmethod
    def _render_index_html(cls, index: ET.Element) -> tuple[str, list[tuple[str, int, tuple[str, ...]]]]:
        """Handle render index html."""
        root_entries = [child for child in index if child.tag == cls._ENTRY_TAG]
        if not root_entries:
            return ('<p class="empty-index">No sections.</p>', [])
        ordered: list[tuple[str, int, tuple[str, ...]]] = []
        items = [cls._render_index_entry_html(entry, depth=0, ancestors=(), ordered=ordered) for entry in root_entries]
        return ('<ul class="document-index">' + ''.join(items) + '</ul>', ordered)

    @classmethod
    def _render_index_entry_html(cls, entry: ET.Element, *, depth: int, ancestors: tuple[str, ...], ordered: list[tuple[str, int, tuple[str, ...]]]) -> str:
        """Handle render index entry html."""
        name = entry.get('name')
        if name is None:
            raise ValueError("<entry> is missing required 'name' attribute.")
        ordered.append((name, depth, ancestors))
        anchor = cls._section_anchor(name)
        children = [child for child in entry if child.tag == cls._ENTRY_TAG]
        nested = ''
        has_children = bool(children)
        if children:
            child_ancestors = (*ancestors, name)
            nested_items = [cls._render_index_entry_html(child, depth=depth + 1, ancestors=child_ancestors, ordered=ordered) for child in children]
            nested = '<ul>' + ''.join(nested_items) + '</ul>'
        if has_children:
            index_toggle = '<button type="button" class="index-toggle" aria-expanded="true" aria-label="Collapse subsections">&minus;</button>'
            row_class = 'index-row'
        else:
            index_toggle = ''
            row_class = 'index-row index-row-single'
        return f'<li data-depth="{depth}"><div class="{row_class}">{index_toggle}<a href="#{anchor}">{escape(name)}</a></div>{nested}</li>'

    @staticmethod
    def _section_anchor(name: str) -> str:
        """Handle section anchor."""
        slug = re.sub('[^a-z0-9]+', '-', name.casefold()).strip('-')
        if not slug:
            slug = 'section'
        slug = slug[:48].rstrip('-') or 'section'
        digest = sha256(name.encode('utf-8')).hexdigest()[:10]
        return f'section-{slug}-{digest}'

    @staticmethod
    def _required_direct_child(parent: ET.Element, tag: str) -> ET.Element:
        """Handle required direct child."""
        matches = [child for child in parent if child.tag == tag]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one <{tag}> inside <{parent.tag}>.')
        return matches[0]

    @staticmethod
    def _optional_direct_child(parent: ET.Element, tag: str) -> ET.Element | None:
        """Handle optional direct child."""
        matches = [child for child in parent if child.tag == tag]
        if len(matches) > 1:
            raise ValueError(f'Expected at most one <{tag}> inside <{parent.tag}>.')
        if not matches:
            return None
        return matches[0]

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        """Handle atomic write text."""
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    @classmethod
    def _normalize_document_name(cls, name: str) -> str:
        """Handle normalize document name."""
        normalized = name.strip()
        if not normalized:
            raise ValueError('Document name cannot be empty.')
        if '\n' in normalized or '\r' in normalized or '\x00' in normalized:
            raise ValueError('Document name contains invalid characters.')
        if '/' in normalized or '\\' in normalized:
            raise ValueError('Document name must be a file name, not a path.')
        if normalized in {'.', '..'}:
            raise ValueError('Invalid document name.')
        suffix = '.citra.xml'
        if normalized.endswith(suffix):
            raise ValueError(f'Pass the logical document name without the {suffix!r} suffix.')
        return normalized

    @classmethod
    def _normalize_location(cls, location: str | None) -> str:
        """Handle normalize location."""
        if location is None:
            return cls._PROJECT_ROOT
        normalized = location.strip()
        normalized = normalized.rstrip('/')
        if not normalized:
            raise ValueError('Document location cannot be empty.')
        if '\\' in normalized:
            raise ValueError('Document locations must use forward slashes.')
        if normalized == cls._LIBRARY_ALIAS or normalized.startswith(f'{cls._LIBRARY_ALIAS}/'):
            return normalized
        candidate = Path(normalized)
        if candidate.is_absolute() or '..' in candidate.parts or normalized.startswith('@'):
            raise ValueError('Document location must stay inside the current project.')
        return normalized

    @classmethod
    def _validate_operation_arguments(cls, arguments: dict[str, Any]) -> None:
        """Handle validate operation arguments."""
        operation_value = arguments.get('operation')
        if operation_value is None:
            raise ValueError("'operation' is required.")
        operation = str(operation_value)
        if operation not in cls._SUPPORTED_OPERATIONS:
            rendered = ', '.join((repr(item) for item in sorted(cls._SUPPORTED_OPERATIONS)))
            raise ValueError(f'Unsupported document operation {operation!r}. Expected one of: {rendered}.')
        required = cls._REQUIRED_FIELDS[operation]
        missing = [field for field in sorted(required) if field not in arguments]
        if missing:
            rendered = ', '.join((repr(field) for field in missing))
            raise ValueError(f'Operation {operation!r} is missing required argument(s): {rendered}.')
        universally_allowed = {'operation', 'name', 'location'}
        operation_allowed = cls._ALLOWED_FIELDS[operation]
        unexpected = set(arguments) - universally_allowed - operation_allowed
        if unexpected:
            rendered = ', '.join((repr(field) for field in sorted(unexpected)))
            raise ValueError(f'Argument(s) not valid for operation {operation!r}: {rendered}.')
        if operation == 'read' and (not arguments.get('sections')):
            raise ValueError("'sections' must contain at least one section name.")

    def _mutation_result(self, result: str) -> str:
        """Handle mutation result."""
        if len(result) <= self.MAX_MUTATION_RESULT_CHARS:
            return result
        truncated_chars = len(result) - self.MAX_MUTATION_RESULT_CHARS
        return result[:self.MAX_MUTATION_RESULT_CHARS] + '\n' + f'... mutation diagnostic truncated ({truncated_chars} additional chars)'

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        """Handle format call log."""
        operation = str(arguments.get('operation', '?'))
        parts = [f'operation={operation}']
        name = arguments.get('name')
        if name is not None:
            parts.append(f'document={name!r}')
        location = arguments.get('location')
        parts.append(f'location={location or self._PROJECT_ROOT!r}')
        if 'recursive' in arguments:
            parts.append(f"recursive={bool(arguments['recursive'])}")
        section = arguments.get('section')
        if section is not None:
            parts.append(f'section={section!r}')
        sections = arguments.get('sections')
        if isinstance(sections, list):
            preview = ', '.join((repr(str(item)) for item in sections[:5]))
            remaining = len(sections) - 5
            if remaining > 0:
                preview += f', +{remaining} more'
            parts.append(f'sections=[{preview}]')
        parent = arguments.get('parent')
        if parent is not None:
            parts.append(f'parent={parent!r}')
        if 'from_line' in arguments:
            parts.append(f"from_line={arguments['from_line']}")
        if 'to_line' in arguments:
            parts.append(f"to_line={arguments['to_line']}")
        content = arguments.get('content')
        if isinstance(content, str):
            parts.append(f'content={len(content.splitlines())} lines/{len(content)} chars')
        title = arguments.get('title')
        if isinstance(title, str):
            parts.append(f'title={title!r}')
        return ' | '.join(parts)

    @override
    def format_result_log(self, result: Any) -> str:
        """Handle format result log."""
        text = str(result)
        if not text:
            return 'empty result'
        return f'{len(text.splitlines())} lines | {len(text)} chars'

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        """Handle optional string."""
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        """Handle optional int."""
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _is_within(root: Path, path: Path) -> bool:
        """Handle is within."""
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            return False
        return True
