from __future__ import annotations

from pathlib import Path
from typing import override
from .skill import Skill
from typing import TYPE_CHECKING
from ..lsp.language import IMPLEMENTED_LANGUAGES

if TYPE_CHECKING:
    from citra.context import ExecutionContext


class CodingConventions(Skill):
    
    def __init__(self) -> None:
        super().__init__("coding-conventions", "Describes coding conventions of a good citra agent, also detailing tools", Path())

    @override
    def get_md(self, context:ExecutionContext) -> str:
        return ""

__PROMPT:str = """
# Coding conventions

These conventions apply across languages unless the project already establishes
a stronger local convention.

Preserve the repository's existing architecture, naming, formatting, and style.
Prefer the smallest coherent change that solves the requested problem.

## Control flow

### Prefer early exits

When practical, establish failure cases, invalid states, and terminating
conditions near the beginning of a function.

Prefer:

```text
if invalid:
    return

if unavailable:
    raise ...

perform_main_work()
````

over deeply nesting the main behavior inside several conditional blocks.

Do not force early-return style when it makes control flow less clear.

## Semantic code intelligence

Citra provides an LSP tool for supported languages, currently Python and
JavaScript/TypeScript.

**Use LSP proactively.**

Do not treat LSP as an optional last resort when semantic information would make
the change safer or faster.

### Prefer LSP when symbol identity matters

Use LSP instead of textual search when answering questions such as:

* Where is this class, function, method, variable, property, or type defined?
* Which references refer to this exact symbol?
* What type does this expression or symbol have?
* What implementation or declaration corresponds to this symbol?
* What symbols exist in this file?
* What diagnostics does the language server report for this file?
* Will renaming, removing, or changing this symbol affect other code?

Useful LSP actions include:

* `document_symbols` to understand the semantic structure of a known file;
* `definition` or `go_to_definition` to follow a symbol to its definition;
* `references` to find semantic usages of a symbol;
* `hover` to inspect inferred or declared type/signature information;
* `implementation`, `declaration`, and `type_definition` when those relationships
  matter;
* `diagnostics` to check affected source files after modifications;
* `status` when you need to determine whether the relevant language server is
  available.

Position-based LSP actions use 1-based line and character values.

### Prefer textual search for textual questions

Use Grep when searching for:

* literal strings;
* configuration keys or values;
* comments;
* log messages;
* generated text;
* filenames or path fragments;
* broad occurrences where symbol identity is irrelevant.

Glob and Tree are appropriate when the question is primarily about file
location or project structure.

It is often useful to combine tools:

1. use Tree, Glob, Read, or Grep to locate likely files;
2. use LSP once the relevant symbol or source position is known;
3. use Read to inspect the surrounding implementation;
4. use LSP references or definitions before changing a shared symbol.

Do not substitute Grep for semantic references merely because the textual name
is easy to search. Different symbols may share the same spelling, aliases may
hide usages, and textual occurrences may not represent actual references.

## Working with unfamiliar code

When entering an unfamiliar implementation:

1. inspect the relevant files and surrounding project structure;
2. use `document_symbols` when it provides a faster semantic overview than
   manually scanning a large source file;
3. follow relevant definitions with LSP;
4. inspect references before changing shared APIs or behavior;
5. read the concrete implementation before editing.

Use semantic information to reduce unnecessary file inspection, not to replace
reading the code that will actually be changed.

## Diagnostics and verification

After modifying Python or JavaScript/TypeScript source, run focused LSP
diagnostics on the affected files when the language server is available.

Treat diagnostics as evidence to investigate. Do not silently ignore new
errors or warnings introduced by the change.

LSP diagnostics complement execution; they do not replace tests, builds,
type-checkers, compilers, or runtime verification when those materially verify
the task.

For broader refactors, use references before editing and diagnostics afterward.

## Types

Preserve and improve useful type information when doing so is consistent with
the project's conventions.

Before guessing the type, signature, or origin of a symbol in a supported
language, prefer LSP hover or definition information when available.

Do not introduce unnecessary casts, type suppressions, `Any`, or equivalent
escape hatches merely to silence diagnostics. Understand the underlying type
relationship first.

When a diagnostic exposes a real inconsistency, prefer fixing the code or type
model over suppressing the diagnostic.
"""