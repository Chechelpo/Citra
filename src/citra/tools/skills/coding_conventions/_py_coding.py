from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

from citra.tools.skills.skill import Skill

if TYPE_CHECKING:
    from citra.context import ExecutionContext


class PythonCodingSkill(Skill):

    """Represent CodingConventions."""
    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(
            "python-coding-conventions",
            "Describes expected patterns of python source files",
            Path(),
        )

    @override
    def get_md(
        self,
        context: ExecutionContext,
    ) -> str:
        """Return get md."""
        return _SKILL

_SKILL = """
# Python coding conventions

When writing Python code, treat Python as a typed language rather than a dynamic scripting language. Prefer explicit contracts, clear dependencies, and code that works well with static analysis.

## On dicts

Avoid using `dict[str, Any]` as function arguments or internal domain models.

Prefer:

* Frozen dataclasses for configuration and value objects.
* NamedTuple for simple immutable records.
* Typed classes or Protocols for behavioral interfaces.

Use `dict[str, Any]` only at parsing, serialization, or external data boundaries. Convert untyped data into typed objects as early as possible.

## On Protocols

Use `Protocol` as the preferred mechanism for defining interfaces.

Prefer depending on behavior:

```python
class UserRepository(Protocol):
    def save(self, user: User) -> None:
        ...
```

rather than depending on concrete implementations.

Protocols provide explicit contracts without requiring inheritance.

## On typing

Always annotate function arguments and return values.

Prefer modern syntax:

```python
list[str]
str | None
```

over older forms:

```python
List[str]
Optional[str]
```

Prefer the most general useful type:

```python
Sequence[User]
Mapping[str, str]
```

instead of concrete collections unless mutation is required.

## On data modeling

Prefer `dataclass` for structured data.

Use `Enum` instead of magic strings or numbers.

Avoid representing invalid states with excessive `None` values. Model required data explicitly.

## On getattr() and setattr()

Avoid `getattr()` and `setattr()` in normal application logic.

Prefer direct attribute access and explicit imports.

Use dynamic attribute access only when implementing systems that require reflection, such as serializers, plugins, or frameworks.

## On errors

Prefer explicit exceptions over sentinel values when failure is exceptional.

Create domain-specific exceptions instead of raising generic `Exception`.

```python
class UserNotFoundError(Exception):
    pass
```

## On resources

Use context managers for resource ownership.

Prefer:

```python
with open(path) as file:
    ...
```

for files, locks, transactions, and similar resources.

## On imports and dependencies

Keep imports at module level.

Prefer importing the objects you reference directly.

If circular imports occur, resolve them with:

* `TYPE_CHECKING`
* `from __future__ import annotations`

Avoid hiding dependencies through runtime imports.

## On Python-specific pitfalls

Avoid mutable default arguments:

```python
def process(items=None):
    ...
```

Use composition, dependency injection, and Protocols instead of monkey patching or inheritance-based reuse.

Prefer keyword-only arguments for APIs where positional arguments are unclear.

Use generators when processing potentially large streams of data.

## General guidance

Prefer explicit code over clever code.

Make dependencies, contracts, and data shapes visible.

Optimize for readability, static analysis, and maintainability.

Use strong-typing as much as possible.

"""