from ._py_coding import PythonCodingSkill
from ._sql_coding import SQLCodingSkill
from citra.tools.skills.skill import Skill

def coding_skills() -> tuple[Skill,...]:
    return (
        PythonCodingSkill(),
        SQLCodingSkill()
    )

__all__ : tuple[str, ...] = (
    "PythonCodingSkill",
    "SQLCodingSkill"
)
