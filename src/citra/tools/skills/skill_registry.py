from __future__ import annotations

from collections.abc import Iterable

import os

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from citra.tools.skills.sandbox_explanation import SandboxEnvironment
from citra.tools.skills.task_recognition import TaskRecognition
from .skill import Skill

if TYPE_CHECKING:
    from citra.context import ExecutionContext
    from citra.agent import AgentSession
    from citra.workflows import SingleModeWorkflow


class SkillRegistry:
    """Registry of installed Citra skills."""

    def __init__(
        self,
        agent_session: AgentSession,
        workflow: SingleModeWorkflow,
        skills_root: Path | None,
        *,
        memory_enabled: bool = True,
    ) -> None:
        if skills_root is None:
            citra_root = os.environ.get("CITRA_INSTALL_ROOT")
            if citra_root is None:
                raise RuntimeError("SkillRegistry couldn't get citra_root")
            skills_root = Path(citra_root, "skills")
        
        self.agent_session = agent_session
        self.skills_root = skills_root
        self.skills: dict[str, Skill] = {}
        self.workflow = workflow
        
        built_in_skills: tuple[Skill, ...] = workflow.skills
        if memory_enabled:
            built_in_skills = (TaskRecognition(), *built_in_skills)
        self._register(built_in_skills)
        
        self._load()
        
    def _load(self) -> None:
        """
        Load the declarative filesystem skills enabled for this workflow.
        """
        if not self.skills_root.is_dir():
            return

    

    def _register(self, skill: Skill | Iterable[Skill]) -> None:
        if isinstance(skill, Skill):
            if skill.name in self.skills:
                raise ValueError(f"Duplicate skill name: {skill.name!r}")

            self.skills[skill.name] = skill
            return

        for s in skill:
            self._register(s)


    def _load_skill(
        self,
        directory: Path,
        config_path: Path,
    ) -> Skill:
        import tomllib

        with config_path.open("rb") as file:
            config = tomllib.load(file)

        name = config.get("name")
        description = config.get("description")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Invalid skill name in {config_path}"
            )

        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"Invalid skill description in {config_path}"
            )

        return Skill(
            name=name.strip(),
            description=description.strip(),
            root=directory,
        )

    def get_skill(
        self,
        skill_name: str,
        context: ExecutionContext,
    ) -> str:
        """
        Return the requested skill instructions.
        """
        skill = self.skills.get(skill_name)

        if skill is None:
            available = ", ".join(sorted(self.skills)) or "(none)"
            raise KeyError(
                f"Unknown skill {skill_name!r}. "
                f"Available skills: {available}"
            )

        return skill.get_md( context )

    def format_for_prompt(
        self,
    ) -> str:
        if not self.skills:
            return "No optional skills are available."

        entries = "\n".join(
            f"- `{skill.name}`: {skill.description}"
            for skill in sorted(
                self.skills.values(),
                key=lambda item: item.name,
            )
        )

        return (
            "Load a relevant skill with the `skill` tool before "
            "performing the workflow it describes.\n\n"
            + entries
        )
