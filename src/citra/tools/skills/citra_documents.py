from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

from .skill import Skill

if TYPE_CHECKING:
    from citra.context import ExecutionContext


@dataclass(frozen=True)
class CitraDocsPromptEnvironment:
    source_documents: tuple[str, ...]
    workspace_documents: tuple[str, ...]


class CitraDocsSkill(Skill):
    """
    Explain when and how to use Citra structured documents.
    """

    def __init__(self) -> None:
        super().__init__(
            "citra-docs",
            "Explains Citra structured documents, discovers existing "
            "documents, and describes how to use them efficiently.",
            Path(),
        )

    @override
    def get_md(
        self,
        context: ExecutionContext,
    ) -> str:
        environment = _collect_environment(
            context
        )

        return _PROMPT.format(
            source_documents=_format_documents(
                environment.source_documents
            ),
            workspace_documents=_format_documents(
                environment.workspace_documents
            ),
        )


def _collect_environment(
    context: ExecutionContext,
) -> CitraDocsPromptEnvironment:
    workspace = context.workspace

    return CitraDocsPromptEnvironment(
        source_documents=_discover_documents(
            workspace.source_workspace,
            display_prefix="@source",
        ),
        workspace_documents=_discover_documents(
            workspace.workspace,
            display_prefix="@workspace",
        ),
    )


def _discover_documents(
    root: Path,
    *,
    display_prefix: str,
) -> tuple[str, ...]:
    if not root.is_dir():
        return ()

    documents: list[str] = []

    for path in root.rglob(
        "*.citra.xml"
    ):
        if not path.is_file():
            continue

        relative = path.relative_to(
            root
        )

        documents.append(
            f"{display_prefix}/{relative.as_posix()}"
        )

    return tuple(
        sorted(
            documents
        )
    )


def _format_documents(
    documents: tuple[str, ...],
) -> str:
    if not documents:
        return "- None detected."

    return "\n".join(
        f"- `{document}`"
        for document in documents
    )


_PROMPT = """
# Citra documents

Citra documents are Citra's structured format for long-form persistent
documents.

They are stored as:

```text
<name>.citra.xml
````

Use the `document` tool to work with them. Do not manually edit their XML
when the document tool can perform the operation.

## Existing documents

Before creating a new document, check whether an existing Citra document
already covers the subject.

### Source documents

These documents currently exist in the permanent source workspace:

{source_documents}

Source documents are authoritative and read-only to ordinary tools. Inspect
them when they contain relevant project knowledge. If one must be changed,
materialize the appropriate project file into the writable workspace and
follow Citra's normal workspace/Commit workflow rather than modifying
`@source` directly.

### Workspace documents

These Citra documents currently exist in the writable agent workspace:

{workspace_documents}

### Library documents

Citra also has a persistent document library available through the `document`
tool under `@library`.

Library documents are intended for reusable knowledge that should remain
available across projects, sessions, and turns.

Use the library when:

* relevant knowledge already exists there;
* material is broadly reusable beyond the current project;
* persistent reference documentation should be available to future Citra
  sessions.

Prefer reading and extending an appropriate existing library document over
creating duplicate knowledge.

The library is not a generic filesystem location. Access and modify its Citra
documents through the `document` tool rather than Bash or ordinary filesystem
tools.

Use `document` listing when the current library contents matter, since any
initial library list in the system prompt is only an orientation snapshot.

Workspace documents may be modified with the `document` tool.

The detected lists are informational snapshots generated when this skill is
loaded. Documents may be created, removed, or materialized later.

## When to use a Citra document

Use a Citra document for persistent long-form material that benefits from
being divided into independently readable and editable sections, especially:

* architecture and system-design documentation;
* technical specifications;
* implementation or project plans;
* research reports;
* manuals and knowledge bases;
* large project documentation;
* documents expected to grow beyond what should be loaded into model context
  at once.

Do not create one merely because text is being written. Short notes, small
READMEs, messages, code, configuration, and disposable scratch material
usually belong elsewhere.

Prefer extending a relevant existing document over creating a duplicate.

## Core workflow

For an unfamiliar document:

1. Use `inspect` to learn its title and hierarchical index.
2. Identify the smallest relevant section or sections.
3. Use `read` to load only those sections.
4. Make focused section updates.
5. Reread affected content when verification is useful.

Citra documents are specifically designed to keep very large documents out of
the model context. Do not defeat this by repeatedly reading the whole
document.

## Document names

The document tool uses a logical name rather than the XML filename.

For example:

```text
architecture.citra.xml
```

is addressed as:

```text
architecture
```

Do not include `.citra.xml` in the document tool's `name` argument.

The current document tool addresses documents in the active workspace by
logical name. A detected source document or a document nested somewhere the
tool cannot directly address may need to be materialized or placed in the
appropriate writable document location before editing.

## Structure

A Citra document has a hierarchical index and flat Markdown section bodies.

Use hierarchy to organize concepts:

```text
architecture
  execution
  storage

tools
  bash
  browser
  document
```

Sections should represent coherent topics: large enough to be meaningful,
small enough to read and edit independently.

Avoid both one enormous section and excessive one-paragraph fragmentation.

## Reading

Inspect before guessing section names.

Read narrowly. If only `bash` is relevant, read `bash` rather than the entire
`tools` subtree.

Keep `expand_children` false unless descendant sections are actually needed.

For large sections, use line ranges and continue reading only as necessary.
Line ranges are zero-based and end-exclusive:

```text
from_line=10
to_line=20
```

means lines `[10, 20)`.

## Writing

Section bodies are Markdown. Write Markdown, not Citra XML.

Prefer targeted updates when preserving existing material matters. Read enough
surrounding context before replacing a line range.

An update where `from_line == to_line` inserts content at that position.

A full-section replacement is appropriate when the section is small or the
task genuinely calls for a complete rewrite.

When adding material, choose its parent according to the document's information
architecture rather than simply appending everything at the root.

## Mermaid diagrams

Use the `diagram` tool when a Citra document would benefit from a simple visual
explanation such as:

* architecture or component flows;
* request and execution sequences;
* state transitions;
* class relationships;
* entity relationships;
* timelines and Gantt plans;
* mindmaps, Git graphs, and simple charts.

Prefer diagrams when they make structure or relationships substantially easier
to understand. Do not add diagrams merely for decoration.

The `diagram` tool manages Mermaid source associated with a Citra document.
For a document such as:

```text
architecture.citra.xml
```

a diagram may be stored as:

```text
architecture.citra.assets/diagrams/runtime-flow.mmd
architecture.citra.assets/diagrams/runtime-flow.svg
```

The `.mmd` Mermaid source is authoritative and editable. The `.svg` is derived
output intended for rendered documentation.

Do not manually edit generated SVG files.

### Diagram workflow

For a new diagram:

1. Create the Mermaid diagram with the `diagram` tool.
2. Use a concise diagram name and useful alternative text.
3. Let the tool validate and render the Mermaid source.
4. Insert the Markdown reference returned by the tool into the appropriate
   Citra document section using the `document` tool.

For an existing diagram:

1. Use `list` or `inspect` when its location or type is unclear.
2. Use `read` to retrieve the Mermaid source before making substantial changes.
3. Use `update` to replace the diagram source.
4. Let the tool regenerate the derived SVG.
5. Keep the existing Markdown reference unless the diagram name or location
   changes.

The diagram tool and document tool have separate responsibilities:

```text
diagram tool
    -> creates and maintains Mermaid assets

document tool
    -> places references to those assets in document Markdown
```

Do not have generic filesystem tools modify `.mmd` or `.svg` files when the
diagram tool can perform the operation.

### Mermaid in document sections

Use the Markdown image reference returned by the diagram tool rather than
inventing asset paths when possible.

Conceptually:

```markdown
![Runtime request flow](architecture.citra.assets/diagrams/runtime-flow.svg)
```

The generated Citra HTML can then display the rendered diagram as part of the
document.

Keep useful alternative text because it describes the diagram when the image
cannot be displayed and improves accessibility.

For advanced Mermaid syntax not covered by simple patterns, Mermaid source may
still be supplied directly through the diagram tool. Prefer clear, maintainable
diagrams over visually dense ones.

## Creating documents

Create a new Citra document only when no suitable existing document exists and
the work benefits from persistent structured storage.

Choose a concise logical filename and a descriptive title, then establish a
useful section structure early instead of writing the entire document into one
section.

Prefer using sections within existing documents over creating new ones

## Removing content

Removing a section can also remove its indexed descendant subtree.

Inspect the hierarchy before deleting a parent section and make sure useful
children are not being removed unintentionally.

Removing a section does not necessarily imply that referenced diagram assets
should also be removed. Remove a diagram separately with the `diagram` tool
only when it is no longer used elsewhere in the document.

## Generated HTML

A Citra document may also produce a human-readable companion such as:

```text
architecture.citra.xml
architecture.html
```

The `.citra.xml` file is authoritative. HTML and rendered diagram SVGs are
derived output for humans.

Do not edit generated HTML or SVG as a way of changing the document.

## Versioning and validation

Treat `versioning` as an optional opaque freshness marker. Do not invent one
unless the surrounding workflow gives it a real meaning.

Use document validation when integrity is in question, after unusual external
modification, or when explicitly required. Normal structured operations do not
need a separate validation call after every change.

Use diagram validation when Mermaid syntax is in question or a render fails.
Do not repeatedly retry invalid Mermaid without first inspecting and correcting
the source.

## Operating discipline

Use the document tool rather than Bash or generic file editing for normal
Citra document mutations.

Use the diagram tool rather than Bash or generic file editing for Mermaid
diagram mutations.

Use the index as a map. Keep only the relevant portions of a large document in
context, make localized changes, and let the persisted Citra document and its
managed diagram assets hold the rest.
"""

