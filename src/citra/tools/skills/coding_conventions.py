from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

from .skill import Skill

if TYPE_CHECKING:
    from citra.context import ExecutionContext


class CodingConventions(Skill):

    def __init__(self) -> None:
        super().__init__(
            "coding",
            "Describes correct patterns and formatting for any code project",
            Path(),
        )

    @override
    def get_md(
        self,
        context: ExecutionContext,
    ) -> str:
        return _PROMPT


_PROMPT: str = """
# Coding conventions

These conventions apply across languages unless the project already establishes
a stronger local convention.

Preserve the repository's existing architecture, naming, formatting, and style.
Prefer the smallest coherent change that fully solves the requested problem.

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

## Structure

Keep functions and classes focused on coherent responsibilities.

Prefer straightforward control flow and explicit data movement over unnecessary
abstraction.

Reuse existing project abstractions when they fit. Do not introduce wrappers,
helpers, base classes, or indirection that provide no meaningful simplification
or reuse.

Avoid unrelated refactors while implementing a focused change.

When a change affects an existing abstraction, preserve its established
semantics unless changing them is part of the task.

## Naming

Use names that describe purpose and domain meaning.

Follow existing repository naming conventions before introducing a new style.

Avoid vague names when a more precise name is practical.

Do not rename existing public or shared symbols without a reason related to the
task.

## Types

Preserve and improve useful type information when consistent with the project's
conventions.

Do not introduce unnecessary casts, type suppressions, `Any`, or equivalent
escape hatches merely to silence diagnostics.

When a type diagnostic exposes a real inconsistency, prefer fixing the code or
type model over suppressing the diagnostic.

Avoid redundant annotations when the project's style clearly relies on
inference.

## Error handling

Handle failures at the layer that has enough context to respond meaningfully.

Do not silently swallow errors unless the behavior is intentionally best-effort
and that convention is established by the project.

Prefer specific validation and error messages over failures that obscure the
invalid state.

Do not catch broad exceptions merely to convert programming errors into normal
control flow.

Preserve existing exception and error-reporting conventions when modifying an
established API.

## State and mutation

Keep state ownership clear.

Prefer local, explicit mutation over hidden side effects when both approaches
are practical.

Do not duplicate authoritative state when an existing source of truth can be
used directly.

When multiple structures must remain synchronized, update them atomically when
practical and validate inputs before mutating state.

## Comments and documentation

Prefer code that communicates its intent through structure and naming.

Add comments when they explain non-obvious constraints, invariants, protocol
details, compatibility requirements, or reasoning that cannot be expressed
clearly in code alone.

Do not add comments that merely restate the code.

Update documentation or docstrings when a behavioral or public API change would
otherwise leave them incorrect.

## Compatibility

Preserve existing public behavior and compatibility boundaries unless the task
explicitly requires changing them.

Avoid changing serialized formats, schemas, public signatures, command
behavior, configuration semantics, or persistent state implicitly.

When compatibility must change, make the boundary explicit and keep the change
as narrow as practical.

## Diagnostics and verification

After modifying source code, treat the automatic post-edit verification as a
required feedback loop. `edit` and `write` run available language-server
diagnostics and every configured project lint rule automatically.

Treat new errors and warnings as evidence to investigate rather than suppressing
them without understanding the cause.

When a configured lint rule reports a violation caused by your change, fix the
violation before considering the edit complete unless the user's task explicitly
requires behavior that conflicts with that repository rule. Do not disable,
weaken, or bypass project lint configuration merely to make a change pass.

Diagnostics complement execution; they do not replace tests, builds,
type-checkers, compilers, or runtime verification when those materially verify
the task.

Do not claim that code works solely because it is syntactically valid or has no
language-server diagnostics.
"""


