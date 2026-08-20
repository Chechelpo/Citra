from enum import Enum

class Language(Enum):
    PYTHON = "python"



def server_for_language(
    language: Language,
) -> str:
    match language:
        case Language.PYTHON:
            return "pyright"