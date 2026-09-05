"""System-prompt helpers with no controller-workspace disclosure."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from citra.context.environment_fetching import EnvironmentInfo
from citra.tools.skills.skill import Skill

if TYPE_CHECKING:
    from citra.context import ExecutionContext


__all__ = [
    "EnvironmentInfo",
    "build_system_prompt",
    "collect_environment",
    "format_skills",
]


def build_system_prompt(context: ExecutionContext) -> str:
    """Build the active workflow's system prompt for one model request."""
    return context.workflow.get_system_prompt(context)


def collect_environment(context: ExecutionContext) -> EnvironmentInfo:
    """Return public execution metadata without project-controller paths."""
    value = context.workspace.environment_info
    if isinstance(value, EnvironmentInfo):
        return value
    return EnvironmentInfo.collect_environment()


def format_skills(skills: Iterable[Skill]) -> str:
    """Handle format skills."""
    return "\n".join(
        f"- **{skill.name}**: {skill.description}"
        for skill in skills
    )

def basic_coding_conventions() -> str:
    return """
    When you code, follow these constraints:

        1. All function/classes must have documentation, regardless if its short or not.
        2. Early exits are encouraged for checks, but do not twist the function to force them.
        3. If a method is sufficiently long, divide it unto different private function.
        4. Logging is required throughout the function, at different levels (info,warn,error) depending on the context.
    """