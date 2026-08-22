from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
import os
import tempfile
import xml.etree.ElementTree as ET


DEFAULT_LINES = 200

ROOT_TAG = "citra-doc"
TITLE_TAG = "title"
VERSIONING_TAG = "versioning"
INDEX_TAG = "index"
ENTRY_TAG = "entry"
SECTIONS_TAG = "sections"
SECTION_TAG = "section"


class CitraDocFormatError(ValueError):
    """Raised when a Citra document does not match the expected XML format."""


class Index:
    """
    Read and mutate the hierarchical ``<index>`` element of a Citra document.

    Expected XML format::

        <index>
            <entry name="overview" />
            <entry name="architecture">
                <entry name="storage" />
                <entry name="networking" />
            </entry>
        </index>

    Entry names are globally unique across the entire index.

    Hierarchy exists only in the index. Section bodies are stored separately
    under the document's flat ``<sections>`` element.
    """

    __slots__ = ("_element",)

    def __init__(
        self,
        element: ET.Element,
    ) -> None:
        if element.tag != INDEX_TAG:
            raise CitraDocFormatError(
                f"Expected <{INDEX_TAG}>, got <{element.tag}>."
            )

        self._element = element
        self._validate()

    def convert_to_md(
        self,
    ) -> str:
        """
        Convert the XML index into a Markdown tree.

        Example output::

            # Index

            - overview
            - architecture
              - storage
              - networking

        Returns ``_No sections._`` beneath the heading when the index is empty.
        """
        lines: list[str] = [
            "# Index",
            "",
        ]

        entries = self._direct_entries(
            self._element
        )

        if not entries:
            lines.append(
                "_No sections._"
            )
            return "\n".join(
                lines
            )

        for entry in entries:
            self._append_markdown_entry(
                entry,
                lines,
                depth=0,
            )

        return "\n".join(
            lines
        )

    def add_entry(
        self,
        name: str,
        parent: str | None = None,
    ) -> str:
        """
        Create a globally unique index entry.

        Args:
            name:
                Name of the new entry.

            parent:
                Existing entry under which the new entry should be placed.
                ``None`` places the entry at the index root.

        Returns:
            A human-readable diagnostic describing the change.

        Raises:
            ValueError:
                If ``name`` is empty or already exists.

            KeyError:
                If ``parent`` was supplied but does not exist.
        """
        normalized_name = self.normalize_name(
            name
        )

        if self.find(
            normalized_name
        ) is not None:
            raise ValueError(
                f"Index entry already exists: "
                f"{normalized_name!r}."
            )

        destination: ET.Element
        normalized_parent: str | None

        if parent is None:
            destination = self._element
            normalized_parent = None
        else:
            normalized_parent = self.normalize_name(
                parent
            )

            parent_entry = self.find(
                normalized_parent
            )

            if parent_entry is None:
                raise KeyError(
                    f"Parent index entry does not exist: "
                    f"{normalized_parent!r}."
                )

            destination = parent_entry

        ET.SubElement(
            destination,
            ENTRY_TAG,
            {
                "name": normalized_name,
            },
        )

        if normalized_parent is None:
            return (
                f"Added index entry "
                f"{normalized_name!r} at root."
            )

        return (
            f"Added index entry "
            f"{normalized_name!r} "
            f"under {normalized_parent!r}."
        )

    def remove_entry(
        self,
        name: str,
    ) -> str:
        """
        Remove an index entry and its complete descendant subtree.

        Removing::

            architecture
              storage
              networking

        removes all three entries.

        Returns:
            A diagnostic listing all removed entry names.

        Raises:
            KeyError:
                If the entry does not exist.
        """
        normalized_name = self.normalize_name(
            name
        )

        located = self._find_with_parent(
            normalized_name
        )

        if located is None:
            raise KeyError(
                f"Index entry does not exist: "
                f"{normalized_name!r}."
            )

        parent, entry = located

        removed_names = (
            self._entry_subtree_names(
                entry
            )
        )

        parent.remove(
            entry
        )

        rendered = ", ".join(
            repr(item)
            for item in removed_names
        )

        return (
            f"Removed index entries: "
            f"{rendered}."
        )

    def find(
        self,
        name: str,
    ) -> ET.Element | None:
        """
        Return the entry named ``name``.

        Returns ``None`` when the entry does not exist.
        """
        for entry in self._iter_entries(
            self._element
        ):
            if entry.get(
                "name"
            ) == name:
                return entry

        return None

    def ordered_names(
        self,
    ) -> tuple[str, ...]:
        """
        Return all entry names in depth-first document order.

        For::

            A
              A1
              A2
            B

        returns::

            ("A", "A1", "A2", "B")
        """
        return tuple(
            self._required_entry_name(
                entry
            )
            for entry in self._iter_entries(
                self._element
            )
        )

    def subtree_names(
        self,
        name: str,
        *,
        include_self: bool = True,
    ) -> tuple[str, ...]:
        """
        Return an entry subtree in depth-first order.

        Args:
            name:
                Root entry.

            include_self:
                When true, the returned tuple starts with ``name``.
                Otherwise only descendants are returned.
        """
        normalized_name = self.normalize_name(
            name
        )

        entry = self.find(
            normalized_name
        )

        if entry is None:
            raise KeyError(
                f"Index entry does not exist: "
                f"{normalized_name!r}."
            )

        names = self._entry_subtree_names(
            entry
        )

        if include_self:
            return names

        return names[1:]

    def _validate(
        self,
    ) -> None:
        if self._element.attrib:
            raise CitraDocFormatError(
                "<index> does not accept attributes."
            )

        for child in self._element:
            if child.tag != ENTRY_TAG:
                raise CitraDocFormatError(
                    "<index> may contain only "
                    "<entry> children; "
                    f"found <{child.tag}>."
                )

        seen: set[str] = set()

        for entry in self._iter_entries(
            self._element
        ):
            if set(
                entry.attrib
            ) != {"name"}:
                raise CitraDocFormatError(
                    "Every <entry> must contain "
                    "exactly one attribute: 'name'."
                )

            name = self._required_entry_name(
                entry
            )

            normalized = self.normalize_name(
                name
            )

            if normalized != name:
                raise CitraDocFormatError(
                    "Index entry name must not have "
                    "surrounding whitespace: "
                    f"{name!r}."
                )

            if name in seen:
                raise CitraDocFormatError(
                    f"Duplicate index entry name: "
                    f"{name!r}."
                )

            seen.add(
                name
            )

            if (
                entry.text is not None
                and entry.text.strip()
            ):
                raise CitraDocFormatError(
                    f"<entry name={name!r}> "
                    "may not contain text."
                )

            for child in entry:
                if child.tag != ENTRY_TAG:
                    raise CitraDocFormatError(
                        f"<entry name={name!r}> "
                        "may contain only <entry> "
                        "children; "
                        f"found <{child.tag}>."
                    )

    def _find_with_parent(
        self,
        name: str,
    ) -> tuple[
        ET.Element,
        ET.Element,
    ] | None:
        return self._find_entry_with_parent(
            self._element,
            name,
        )

    @classmethod
    def _find_entry_with_parent(
        cls,
        parent: ET.Element,
        name: str,
    ) -> tuple[
        ET.Element,
        ET.Element,
    ] | None:
        for child in parent:
            if child.tag != ENTRY_TAG:
                continue

            if child.get(
                "name"
            ) == name:
                return (
                    parent,
                    child,
                )

            nested = cls._find_entry_with_parent(
                child,
                name,
            )

            if nested is not None:
                return nested

        return None

    @classmethod
    def _iter_entries(
        cls,
        parent: ET.Element,
    ) -> Iterable[ET.Element]:
        for child in parent:
            if child.tag != ENTRY_TAG:
                continue

            yield child

            yield from cls._iter_entries(
                child
            )

    @staticmethod
    def _direct_entries(
        parent: ET.Element,
    ) -> list[ET.Element]:
        return [
            child
            for child in parent
            if child.tag == ENTRY_TAG
        ]

    @classmethod
    def _entry_subtree_names(
        cls,
        entry: ET.Element,
    ) -> tuple[str, ...]:
        names: list[str] = [
            cls._required_entry_name(
                entry
            )
        ]

        names.extend(
            cls._required_entry_name(
                child
            )
            for child in cls._iter_entries(
                entry
            )
        )

        return tuple(
            names
        )

    @classmethod
    def _append_markdown_entry(
        cls,
        entry: ET.Element,
        lines: list[str],
        *,
        depth: int,
    ) -> None:
        name = cls._required_entry_name(
            entry
        )

        lines.append(
            f"{'  ' * depth}- {name}"
        )

        for child in cls._direct_entries(
            entry
        ):
            cls._append_markdown_entry(
                child,
                lines,
                depth=depth + 1,
            )

    @staticmethod
    def _required_entry_name(
        entry: ET.Element,
    ) -> str:
        name = entry.get(
            "name"
        )

        if name is None:
            raise CitraDocFormatError(
                "<entry> is missing required "
                "'name' attribute."
            )

        return name

    @staticmethod
    def normalize_name(
        name: str,
    ) -> str:
        """
        Validate and normalize a section/index name.

        Names:
        - must not be empty;
        - may not contain newlines;
        - are stripped of surrounding whitespace.
        """
        normalized = name.strip()

        if not normalized:
            raise ValueError(
                "Entry/section name cannot be empty."
            )

        if (
            "\n" in normalized
            or "\r" in normalized
        ):
            raise ValueError(
                "Entry/section names cannot "
                "contain newlines."
            )

        return normalized


@dataclass(slots=True)
class CitraDoc:
    """
    Read and mutate a Citra documentation XML file.

    Expected document format::

        <?xml version="1.0" encoding="utf-8"?>
        <citra-doc>
            <title>Citra Architecture</title>

            <versioning>
                main@abc123
            </versioning>

            <index>
                <entry name="overview" />

                <entry name="tools">
                    <entry name="browser" />
                </entry>
            </index>

            <sections>
                <section name="overview">
                    Markdown body...
                </section>

                <section name="tools">
                    Markdown body...
                </section>

                <section name="browser">
                    Markdown body...
                </section>
            </sections>
        </citra-doc>

    Exact format rules:

    - ``<citra-doc>`` is the required root element.
    - ``<title>`` is required exactly once.
    - ``<versioning>`` is optional and may appear at most once.
    - ``<index>`` is required exactly once.
    - ``<sections>`` is required exactly once.
    - index entry names are globally unique;
    - section names are globally unique;
    - each index entry has exactly one corresponding section;
    - each section has exactly one corresponding index entry;
    - hierarchy and ordering exist only in ``<index>``;
    - ``<sections>`` is deliberately flat;
    - section bodies contain Markdown as XML text;
    - section bodies may not contain nested XML elements.

    XML escaping:

    Section Markdown is assigned to ``Element.text``. Characters such as
    ``<``, ``>``, and ``&`` are escaped automatically when written and
    restored when parsed.

    Line ranges use Python slicing semantics:

    - ``from_line`` is zero-based and inclusive;
    - ``to_line`` is zero-based and exclusive;
    - ``to_line=None`` means through EOF;
    - reads clamp naturally when ``to_line`` exceeds the section length;
    - updates require their range to fit inside the existing section.

    Example::

        from_line=0, to_line=None
            whole section

        from_line=10, to_line=20
            lines [10, 20)

        from_line=5, to_line=5
            insertion before logical line 5
    """

    path: Path

    def __post_init__(
        self,
    ) -> None:
        self.path = Path(
            self.path
        )

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        title: str,
        versioning: str | None = None,
    ) -> CitraDoc:
        """
        Create a new empty Citra document.

        Args:
            path:
                Destination XML path. It must not already exist.

            title:
                Required human-readable document title.

            versioning:
                Optional opaque freshness/version marker.

                For repository documentation this could be, for example::

                    main@abc123

        Returns:
            A ``CitraDoc`` bound to the newly-created file.
        """
        destination = Path(
            path
        )

        if destination.exists():
            raise FileExistsError(
                "Citra document already exists: "
                f"{destination}"
            )

        normalized_title = (
            cls._normalize_metadata(
                title,
                field="title",
                allow_empty=False,
            )
        )

        root = ET.Element(
            ROOT_TAG
        )

        title_element = ET.SubElement(
            root,
            TITLE_TAG,
        )

        title_element.text = (
            normalized_title
        )

        if versioning is not None:
            versioning_element = (
                ET.SubElement(
                    root,
                    VERSIONING_TAG,
                )
            )

            versioning_element.text = (
                cls._normalize_metadata(
                    versioning,
                    field="versioning",
                    allow_empty=True,
                )
            )

        ET.SubElement(
            root,
            INDEX_TAG,
        )

        ET.SubElement(
            root,
            SECTIONS_TAG,
        )

        document = cls(
            destination
        )

        document._write_tree(
            ET.ElementTree(
                root
            )
        )

        return document

    def validate(
        self,
    ) -> str:
        """
        Parse and validate the complete document.

        Returns:
            ``"ok"`` when valid.

        Raises:
            FileNotFoundError:
                If the document does not exist.

            CitraDocFormatError:
                If its XML or Citra document structure is invalid.
        """
        self._load_tree()
        return "ok"

    def read_title(
        self,
    ) -> str:
        """
        Return the required document title.
        """
        root = self._load_root()

        title = self._require_single_child(
            root,
            TITLE_TAG,
        )

        return title.text or ""

    def update_title(
        self,
        new_content: str,
    ) -> str:
        """
        Replace the required title.

        The title cannot be empty after surrounding whitespace is removed.

        Returns:
            A unified diff between the old and new title.
        """
        normalized = self._normalize_metadata(
            new_content,
            field="title",
            allow_empty=False,
        )

        tree = self._load_tree()
        root = tree.getroot()

        title = self._require_single_child(
            root,
            TITLE_TAG,
        )

        old_content = title.text or ""

        title.text = normalized

        self._write_tree(
            tree
        )

        return self._text_diff(
            old_content,
            normalized,
            label="title",
        )

    def read_versioning(
        self,
    ) -> str:
        """
        Return the optional version marker.

        Returns:
            The versioning text, or ``""`` when ``<versioning>`` is absent.

        CitraDoc intentionally treats this value as opaque. A repository
        document might use a branch + commit, while another document type
        could use some other freshness marker.
        """
        root = self._load_root()

        versioning = (
            self._optional_single_child(
                root,
                VERSIONING_TAG,
            )
        )

        if versioning is None:
            return ""

        return versioning.text or ""

    def update_versioning(
        self,
        new_content: str,
    ) -> str:
        """
        Create or replace the optional ``<versioning>`` value.

        Empty versioning text is permitted.

        Returns:
            A unified diff between the old and new versioning values.
        """
        normalized = self._normalize_metadata(
            new_content,
            field="versioning",
            allow_empty=True,
        )

        tree = self._load_tree()
        root = tree.getroot()

        versioning = (
            self._optional_single_child(
                root,
                VERSIONING_TAG,
            )
        )

        old_content = ""

        if versioning is None:
            versioning = ET.Element(
                VERSIONING_TAG
            )

            title = (
                self._require_single_child(
                    root,
                    TITLE_TAG,
                )
            )

            title_position = (
                list(root).index(
                    title
                )
            )

            root.insert(
                title_position + 1,
                versioning,
            )

        else:
            old_content = (
                versioning.text or ""
            )

        versioning.text = normalized

        self._write_tree(
            tree
        )

        return self._text_diff(
            old_content,
            normalized,
            label="versioning",
        )

    def read_index(
        self,
    ) -> str:
        """
        Return the hierarchical XML index as Markdown.

        Example::

            # Index

            - overview
            - architecture
              - storage
              - networking
        """
        root = self._load_root()

        index_element = (
            self._require_single_child(
                root,
                INDEX_TAG,
            )
        )

        index = Index(
            index_element
        )

        return index.convert_to_md()

    def read_sections(
        self,
        names: Iterable[str],
        from_line: int = 0,
        to_line: int | None = DEFAULT_LINES,
        expand_children: bool = True,
    ) -> str:
        """
        Read several sections as Markdown.

        Args:
            names:
                Section names to read.

            from_line:
                Zero-based inclusive line offset, applied independently to
                every selected section.

            to_line:
                Zero-based exclusive end offset, applied independently to
                every selected section. ``None`` means EOF.

            expand_children:
                When true, each requested section expands to its entire index
                subtree in depth-first order.

        Overlapping requested subtrees are de-duplicated.

        Output format::

            ## Section: overview
            <!-- lines 0:12 of 12; end is exclusive -->

            Markdown body...

        Returns:
            Concatenated Markdown for all selected sections.
        """
        self._validate_slice(
            from_line,
            to_line,
        )

        tree = self._load_tree()
        root = tree.getroot()

        index_element = (
            self._require_single_child(
                root,
                INDEX_TAG,
            )
        )

        sections = (
            self._require_single_child(
                root,
                SECTIONS_TAG,
            )
        )

        index = Index(
            index_element
        )

        ordered_names: list[str] = []
        seen: set[str] = set()

        for raw_name in names:
            name = Index.normalize_name(
                raw_name
            )

            selected_names = (
                index.subtree_names(
                    name
                )
                if expand_children
                else (name,)
            )

            for selected_name in selected_names:
                if selected_name in seen:
                    continue

                seen.add(
                    selected_name
                )

                ordered_names.append(
                    selected_name
                )

        rendered: list[str] = []

        for name in ordered_names:
            section = self._require_section(
                sections,
                name,
            )

            rendered.append(
                self._render_section(
                    section,
                    from_line=from_line,
                    to_line=to_line,
                )
            )

        return "\n\n".join(
            rendered
        )

    def read_section(
        self,
        name: str,
        from_line: int = 0,
        to_line: int | None = DEFAULT_LINES,
        expand_children: bool = True,
    ) -> str:
        """
        Read one section.

        When ``expand_children=True``, all descendants beneath that section's
        index entry are included as well.

        This is a convenience wrapper around ``read_sections``.
        """
        return self.read_sections(
            (
                name,
            ),
            from_line=from_line,
            to_line=to_line,
            expand_children=expand_children,
        )

    def create_section(
        self,
        name: str,
        content: str,
        parent: str | None = None,
    ) -> str:
        """
        Create an index entry and corresponding section body atomically.

        Args:
            name:
                Globally unique section/index name.

            content:
                Markdown body stored as XML text.

            parent:
                Existing index entry beneath which this section should appear.
                ``None`` creates a root-level entry.

        The resulting structure is conceptually::

            <index>
                ...
                <entry name="NAME" />
            </index>

            <sections>
                ...
                <section name="NAME">
                    CONTENT
                </section>
            </sections>

        Returns:
            Diagnostics describing both changes.
        """
        normalized_name = (
            Index.normalize_name(
                name
            )
        )

        tree = self._load_tree()
        root = tree.getroot()

        index_element = (
            self._require_single_child(
                root,
                INDEX_TAG,
            )
        )

        sections = (
            self._require_single_child(
                root,
                SECTIONS_TAG,
            )
        )

        index = Index(
            index_element
        )

        existing = self._find_section(
            sections,
            normalized_name,
        )

        if existing is not None:
            raise ValueError(
                f"Section already exists: "
                f"{normalized_name!r}."
            )

        index_diagnostic = (
            index.add_entry(
                normalized_name,
                parent,
            )
        )

        section = ET.SubElement(
            sections,
            SECTION_TAG,
            {
                "name": normalized_name,
            },
        )

        section.text = content

        self._validate_root(
            root
        )

        self._write_tree(
            tree
        )

        return (
            f"{index_diagnostic}\n"
            f"Created section "
            f"{normalized_name!r}."
        )

    def update_section(
        self,
        name: str,
        new_content: str,
        from_line: int = 0,
        to_line: int | None = None,
    ) -> str:
        """
        Replace a logical line range inside one section.

        Slice semantics::

            from_line=0, to_line=None
                replace the entire section

            from_line=10, to_line=20
                replace lines [10, 20)

            from_line=5, to_line=5
                insert before line 5

        The resulting section is normalized to ``\\n`` line separators.

        Args:
            name:
                Existing section name.

            new_content:
                Replacement Markdown.

            from_line:
                Zero-based inclusive start line.

            to_line:
                Zero-based exclusive end line. ``None`` means EOF.

        Returns:
            A unified diff of the complete old and new section bodies.
        """
        self._validate_slice(
            from_line,
            to_line,
        )

        normalized_name = (
            Index.normalize_name(
                name
            )
        )

        tree = self._load_tree()
        root = tree.getroot()

        sections = (
            self._require_single_child(
                root,
                SECTIONS_TAG,
            )
        )

        section = self._require_section(
            sections,
            normalized_name,
        )

        old_content = (
            section.text or ""
        )

        old_lines = (
            old_content.splitlines()
        )

        if from_line > len(
            old_lines
        ):
            raise ValueError(
                f"from_line {from_line} is "
                "beyond the section's "
                f"{len(old_lines)} logical lines."
            )

        effective_to = (
            len(old_lines)
            if to_line is None
            else to_line
        )

        if effective_to > len(
            old_lines
        ):
            raise ValueError(
                f"to_line {effective_to} is "
                "beyond the section's "
                f"{len(old_lines)} logical lines."
            )

        replacement_lines = (
            new_content.splitlines()
        )

        updated_lines = (
            old_lines[:from_line]
            + replacement_lines
            + old_lines[effective_to:]
        )

        updated_content = "\n".join(
            updated_lines
        )

        section.text = updated_content

        self._write_tree(
            tree
        )

        return self._text_diff(
            old_content,
            updated_content,
            label=normalized_name,
        )

    def remove_section(
        self,
        name: str,
    ) -> str:
        """
        Remove a section and its complete indexed descendant subtree.

        Because index hierarchy defines section hierarchy, removing a parent
        also removes every corresponding descendant section body.

        Example:

            architecture
              storage
              networking

        Removing ``architecture`` removes all three index entries and all
        three matching ``<section>`` elements.

        Returns:
            Diagnostics listing removed index entries and sections.
        """
        normalized_name = (
            Index.normalize_name(
                name
            )
        )

        tree = self._load_tree()
        root = tree.getroot()

        index_element = (
            self._require_single_child(
                root,
                INDEX_TAG,
            )
        )

        sections = (
            self._require_single_child(
                root,
                SECTIONS_TAG,
            )
        )

        index = Index(
            index_element
        )

        removed_names = (
            index.subtree_names(
                normalized_name
            )
        )

        index_diagnostic = (
            index.remove_entry(
                normalized_name
            )
        )

        removed_sections: list[str] = []

        for section_name in removed_names:
            section = self._find_section(
                sections,
                section_name,
            )

            if section is None:
                raise CitraDocFormatError(
                    f"Index entry "
                    f"{section_name!r} "
                    "has no matching section."
                )

            sections.remove(
                section
            )

            removed_sections.append(
                section_name
            )

        self._validate_root(
            root
        )

        self._write_tree(
            tree
        )

        rendered = ", ".join(
            repr(item)
            for item in removed_sections
        )

        return (
            f"{index_diagnostic}\n"
            f"Removed sections: "
            f"{rendered}."
        )

    def _load_tree(
        self,
    ) -> ET.ElementTree:
        if not self.path.is_file():
            raise FileNotFoundError(
                "Citra document does not exist: "
                f"{self.path}"
            )

        try:
            tree = ET.parse(
                self.path
            )

        except ET.ParseError as error:
            raise CitraDocFormatError(
                "Invalid XML in Citra document "
                f"{self.path}: {error}"
            ) from error

        root = tree.getroot()

        self._validate_root(
            root
        )

        return tree

    def _load_root(
        self,
    ) -> ET.Element:
        tree = self._load_tree()
        return tree.getroot()

    @classmethod
    def _validate_root(
        cls,
        root: ET.Element,
    ) -> None:
        if root.tag != ROOT_TAG:
            raise CitraDocFormatError(
                f"Expected <{ROOT_TAG}> root, "
                f"got <{root.tag}>."
            )

        if root.attrib:
            raise CitraDocFormatError(
                f"<{ROOT_TAG}> "
                "does not accept attributes."
            )

        allowed_tags = {
            TITLE_TAG,
            VERSIONING_TAG,
            INDEX_TAG,
            SECTIONS_TAG,
        }

        for child in root:
            if child.tag not in allowed_tags:
                raise CitraDocFormatError(
                    "Unexpected top-level "
                    f"element <{child.tag}>."
                )

        title = cls._require_single_child(
            root,
            TITLE_TAG,
        )

        cls._validate_text_only_element(
            title,
            field="title",
            allow_empty=False,
        )

        versioning = (
            cls._optional_single_child(
                root,
                VERSIONING_TAG,
            )
        )

        if versioning is not None:
            cls._validate_text_only_element(
                versioning,
                field="versioning",
                allow_empty=True,
            )

        index_element = (
            cls._require_single_child(
                root,
                INDEX_TAG,
            )
        )

        sections = (
            cls._require_single_child(
                root,
                SECTIONS_TAG,
            )
        )

        index = Index(
            index_element
        )

        cls._validate_sections(
            sections,
            index,
        )

    @classmethod
    def _validate_sections(
        cls,
        sections: ET.Element,
        index: Index,
    ) -> None:
        if sections.attrib:
            raise CitraDocFormatError(
                "<sections> does not "
                "accept attributes."
            )

        if (
            sections.text is not None
            and sections.text.strip()
        ):
            raise CitraDocFormatError(
                "<sections> may contain only "
                "<section> children."
            )

        section_names: set[str] = set()

        for section in sections:
            if section.tag != SECTION_TAG:
                raise CitraDocFormatError(
                    "<sections> may contain only "
                    "<section> children; "
                    f"found <{section.tag}>."
                )

            if set(
                section.attrib
            ) != {"name"}:
                raise CitraDocFormatError(
                    "Every <section> must contain "
                    "exactly one attribute: 'name'."
                )

            name_value = section.get(
                "name"
            )

            if name_value is None:
                raise CitraDocFormatError(
                    "<section> is missing required "
                    "'name' attribute."
                )

            normalized_name = (
                Index.normalize_name(
                    name_value
                )
            )

            if normalized_name != name_value:
                raise CitraDocFormatError(
                    "Section name must not have "
                    "surrounding whitespace: "
                    f"{name_value!r}."
                )

            if normalized_name in section_names:
                raise CitraDocFormatError(
                    f"Duplicate section name: "
                    f"{normalized_name!r}."
                )

            section_names.add(
                normalized_name
            )

            if list(
                section
            ):
                raise CitraDocFormatError(
                    f"Section "
                    f"{normalized_name!r} "
                    "must contain text/Markdown "
                    "only, not nested XML elements."
                )

        index_names = set(
            index.ordered_names()
        )

        missing_sections = (
            index_names
            - section_names
        )

        if missing_sections:
            rendered = ", ".join(
                repr(name)
                for name in sorted(
                    missing_sections
                )
            )

            raise CitraDocFormatError(
                "Index entries without matching "
                f"sections: {rendered}."
            )

        orphan_sections = (
            section_names
            - index_names
        )

        if orphan_sections:
            rendered = ", ".join(
                repr(name)
                for name in sorted(
                    orphan_sections
                )
            )

            raise CitraDocFormatError(
                "Sections without matching "
                f"index entries: {rendered}."
            )

    @staticmethod
    def _require_single_child(
        parent: ET.Element,
        tag: str,
    ) -> ET.Element:
        """
        Return exactly one direct child named ``tag``.

        This helper is deliberately used instead of ``Element.find()`` so
        callers receive ``ET.Element`` rather than ``ET.Element | None``.
        """
        matches = [
            child
            for child in parent
            if child.tag == tag
        ]

        if not matches:
            raise CitraDocFormatError(
                f"<{parent.tag}> is missing "
                f"required <{tag}> element."
            )

        if len(matches) > 1:
            raise CitraDocFormatError(
                f"<{parent.tag}> contains "
                f"multiple <{tag}> elements."
            )

        return matches[0]

    @staticmethod
    def _optional_single_child(
        parent: ET.Element,
        tag: str,
    ) -> ET.Element | None:
        """
        Return zero or one direct child named ``tag``.

        Raises when more than one matching child exists.
        """
        matches = [
            child
            for child in parent
            if child.tag == tag
        ]

        if len(matches) > 1:
            raise CitraDocFormatError(
                f"<{parent.tag}> contains "
                f"multiple <{tag}> elements."
            )

        if not matches:
            return None

        return matches[0]

    @classmethod
    def _validate_text_only_element(
        cls,
        element: ET.Element,
        *,
        field: str,
        allow_empty: bool,
    ) -> None:
        if element.attrib:
            raise CitraDocFormatError(
                f"<{element.tag}> "
                "does not accept attributes."
            )

        if list(
            element
        ):
            raise CitraDocFormatError(
                f"<{element.tag}> "
                "must contain text only."
            )

        cls._normalize_metadata(
            element.text or "",
            field=field,
            allow_empty=allow_empty,
        )

    @staticmethod
    def _find_section(
        sections: ET.Element,
        name: str,
    ) -> ET.Element | None:
        for section in sections:
            if (
                section.tag == SECTION_TAG
                and section.get(
                    "name"
                ) == name
            ):
                return section

        return None

    @classmethod
    def _require_section(
        cls,
        sections: ET.Element,
        name: str,
    ) -> ET.Element:
        """
        Return the required section named ``name``.

        Unlike ``_find_section``, the return type is non-optional.
        """
        section = cls._find_section(
            sections,
            name,
        )

        if section is None:
            raise KeyError(
                f"Unknown section: {name!r}."
            )

        return section

    @staticmethod
    def _render_section(
        section: ET.Element,
        *,
        from_line: int,
        to_line: int | None,
    ) -> str:
        name = section.get(
            "name"
        )

        if name is None:
            raise CitraDocFormatError(
                "<section> is missing required "
                "'name' attribute."
            )

        content = (
            section.text or ""
        )

        lines = (
            content.splitlines()
        )

        selected = lines[
            from_line:to_line
        ]

        requested_end = (
            len(lines)
            if to_line is None
            else to_line
        )

        effective_end = min(
            len(lines),
            requested_end,
        )

        body = "\n".join(
            selected
        )

        if not body:
            body = (
                "_Empty section or empty "
                "selected range._"
            )

        return (
            f"## Section: {name}\n"
            f"<!-- lines "
            f"{from_line}:{effective_end} "
            f"of {len(lines)}; "
            "end is exclusive -->\n\n"
            f"{body}"
        )

    @staticmethod
    def _validate_slice(
        from_line: int,
        to_line: int | None,
    ) -> None:
        if from_line < 0:
            raise ValueError(
                "from_line must be greater "
                "than or equal to 0."
            )

        if to_line is None:
            return

        if to_line < 0:
            raise ValueError(
                "to_line must be greater "
                "than or equal to 0."
            )

        if to_line < from_line:
            raise ValueError(
                "to_line cannot be less "
                "than from_line."
            )

    @staticmethod
    def _normalize_metadata(
        value: str,
        *,
        field: str,
        allow_empty: bool,
    ) -> str:
        normalized = value.strip()

        if (
            not allow_empty
            and not normalized
        ):
            raise ValueError(
                f"{field} cannot be empty."
            )

        return normalized

    @staticmethod
    def _text_diff(
        old_content: str,
        new_content: str,
        *,
        label: str,
    ) -> str:
        """
        Produce a normal unified diff for a metadata value or section body.
        """
        if old_content == new_content:
            return "No changes."

        return "\n".join(
            unified_diff(
                old_content.splitlines(),
                new_content.splitlines(),
                fromfile=f"{label}:before",
                tofile=f"{label}:after",
                lineterm="",
            )
        )

    def _write_tree(
        self,
        tree: ET.ElementTree,
    ) -> None:
        """
        Validate and atomically persist an XML tree.

        A temporary file is created beside the target so ``os.replace`` remains
        an atomic same-filesystem replacement.
        """
        root = tree.getroot()
        if root is None:
            raise Exception("Root is none")

        self._validate_root(
            root
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        ET.indent(
            tree,
            space="    ",
        )

        descriptor, temporary_name = (
            tempfile.mkstemp(
                dir=self.path.parent,
                prefix=(
                    f".{self.path.name}."
                ),
                suffix=".tmp",
            )
        )

        temporary_path = Path(
            temporary_name
        )

        try:
            with os.fdopen(
                descriptor,
                "wb",
            ) as stream:
                tree.write(
                    stream,
                    encoding="utf-8",
                    xml_declaration=True,
                    short_empty_elements=True,
                )

            os.replace(
                temporary_path,
                self.path,
            )

        except Exception:
            temporary_path.unlink(
                missing_ok=True
            )
            raise