from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

from .skill import Skill

if TYPE_CHECKING:
    from citra.context import ExecutionContext


class IntegrationSkill(Skill):

    """Represent Integrations."""
    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(
            "integrations",
            "Describes how to correctly create applications that integrate with third-party services/modules",
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
# Integrations

When writing with a third-party module or service (whether it be through binaries or APIs) take time to do the following:

    1. Search for any documentation using web-search.
    2. If possible, download the documentation under docs/ inside your workspace.

If no documentation is available, code inspection is still possible via cloning the repository and inspecting it with the Git tool.

## Where to write them

Integrations are particularly fragile pieces of code, so try to keep them isolated into a module or a source file whose only scope is this particular integration.
Make the reason why this module exists clear in its header.

Always check if an existing module already deals with this third-party service.

## Before writing an integration

Before writing an integration, check the documentation (whether it be from inspecting code or actual documentation) for what you are about to write.
After finding the information, document the source inside the function's documentation.
This documentation must declare:

    1. Endpoint or functions referenced.
    2. Datetime at which this function was created.
    3. Version of the target third-party module/API.

## Considerations

If this particular integration has alternatives (ex.: nanogpt API to openrouter's API | anthropic message API vs chat-completions API) make the module return a parsed response object,
so as to avoid internal dependency on the integration's api (letting future developers change provider more easily.)

Internal code must remain unaware of the particular integration's API.
"""