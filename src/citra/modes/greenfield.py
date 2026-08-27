from __future__ import annotations

from typing import TYPE_CHECKING, override

from citra.modes.mode import SandboxConfig, SandboxMode, StaticMode
from citra.tools.default_registry import all_tools
from citra.utils.prompt import collect_environment

if TYPE_CHECKING:
    from citra.context import ExecutionContext
    from citra.utils.prompt import PromptEnvironment



class GreenfieldMode(StaticMode):
    _NAME = "greenfield"
    _DESCRIPTION = "Provides a lengthy task steering and system prompt in order to guide greenfield projects."

    _CORE_TOOLS = all_tools(are_deferred=False)
    _ALLOWED_TOOLS = all_tools(are_deferred=True)

    _SANDBOX_CONFIG = SandboxConfig(
        mode=SandboxMode.FULL_SANDBOX
    )

    @override
    def get_system_prompt(self, context: ExecutionContext) -> str:
        environment: PromptEnvironment = collect_environment(context)
        return f"""
# Task
You are called "OG". Sign every document/decision/constraint you write with your name.
You are an expert developer, with a high expertise in systems architecture. You've been tasked to do a greenfield implementation project inside of a permissive sandbox.

## Environment

{environment.as_prompt_section()}

### Developing pipeline

#### Planning

## Considerations

- Your implementation must be mantainable and ordered.
- Do not over-engineer your implementation.

### Structure
Standard source root folder + tests folder. Citra doc documenting the project
"""
