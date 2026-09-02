"""System-prompt helpers with no controller-workspace disclosure."""

from __future__ import annotations

from collections.abc import Iterable

from citra.context.environment_fetching import EnvironmentInfo
from citra.tools.skills.skill import Skill


__all__ = [
    "EnvironmentInfo",
    "build_system_prompt",
    "collect_environment",
    "format_skills",
]


def build_system_prompt(context) -> str:
    """Build the active mode's system prompt for one model request."""
    return context.mode.get_system_prompt(context)


def collect_environment(context) -> EnvironmentInfo:
    """Return public execution metadata without project-controller paths."""
    workspace = getattr(context, "workspace", None)
    value = getattr(workspace, "environment_info", None)
    if isinstance(value, EnvironmentInfo):
        return value
    return EnvironmentInfo.collect_environment()


def format_skills(skills: Iterable[Skill]) -> str:
    return "\n".join(
        f"- **{skill.name}**: {skill.description}"
        for skill in skills
    )
