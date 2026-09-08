"""Skill implementations and the lazily exposed model-facing skill tool."""

from typing import Any

__all__ = ["SkillTool", "SkillToolGroup"]


def __getattr__(name: str) -> Any:
    if name == "SkillTool":
        from .tool import SkillTool

        return SkillTool
    if name == "SkillToolGroup":
        from .group import SkillToolGroup

        return SkillToolGroup
    raise AttributeError(name)
