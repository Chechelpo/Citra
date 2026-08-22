from __future__ import annotations

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


class SkillRegistry:
    """Registry of installed Citra skills."""

    def __init__(
        self,
        agent_session: AgentSession,
        skills_root: Path | None,
    ) -> None:
        if skills_root is None:
            citra_root = os.environ.get("CITRA_INSTALL_ROOT")
            if citra_root is None:
                raise RuntimeError("SkillRegistry couldn't get citra_root")
            skills_root = Path(citra_root, "skills")
        
        self.agent_session = agent_session
        self.skills_root = skills_root
        self.skills: dict[str, Skill] = dict()
        
        self._register(TaskRecognition())
        self._register(SandboxEnvironment())

        self._load()
        
    def _load(self) -> None:
        """
        Discover valid skill directories and populate the registry.
        """
        if not self.skills_root.is_dir():
            return

        for directory in sorted(self.skills_root.iterdir()):
            if not directory.is_dir():
                continue

            config_path = directory / "skill.toml"
            markdown_path = directory / "SKILL.md"

            if not config_path.is_file() or not markdown_path.is_file():
                continue

            skill = self._load_skill(directory, config_path)

            self._register(skill)
    
    def _register(self, skill:Skill) -> None:
        if skill.name in self.skills:
            raise ValueError(
                   f"Duplicate skill name: {skill.name!r}"
            )
        
        self.skills[skill.name] = skill


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

    def list(self) -> str:
        """
        Return the compact skill index supplied to the model.
        """
        if not self.skills:
            return "# SKILLS\n\nNo skills are installed."

        entries = "\n".join(
            f"- `{skill.name}`: {skill.description}"
            for skill in sorted(
                self.skills.values(),
                key=lambda item: item.name,
            )
        )

        return (
            "# SKILLS\n\n"
            "Load a relevant skill before performing its workflow.\n\n"
            f"{entries}"
        )