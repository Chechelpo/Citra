"""System-prompt helpers with no controller-workspace disclosure."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from citra.context.environment_fetching import EnvironmentInfo
from citra.tools.skills.skill import Skill
from citra.workflows import SingleModeWorkflow
from citra.utils.directory_tree import render_tree
from .names import agent_name

if TYPE_CHECKING:
    from citra.context import ExecutionContext


__all__ = [
    "build_system_prompt"
]


def build_system_prompt(
    context: ExecutionContext, 
    preepend : str,
    append : str,
    give_name : bool = False,
    add_skills : bool = True,
    add_coding_convetions:bool = False,
    add_environment:bool = True,
    add_aider_tree:bool = True,
    add_directory_tree:bool = True
) -> str:
    """
    Build a system prompt.
    
    Returns:

    ```text
    
    preepend

    <injected text>

    append
    ```
    
    """
    worflow: SingleModeWorkflow = context.workflow
    result:str = preepend
    
    if give_name:
        result = _concat(f"You are {agent_name()}. Sign every piece of documentation with it",  result)

    if (add_environment):
        result = _concat(result, _collect_environment(context).as_prompt_section())
    
    if add_aider_tree or add_directory_tree:
        result = _concat(result, _tree_sections(context, add_aider_tree, add_directory_tree))

    if (add_coding_convetions):
        result = _concat(result, _basic_coding_conventions())
    
    if (add_skills):
        result = _concat(result, _format_skills(worflow)) 

    return result + append

def _concat(first:str, second:str, separator:str = "\n") -> str:
    return first + separator + second

def _collect_environment(context: ExecutionContext) -> EnvironmentInfo:
    """Return public execution metadata without project-controller paths."""
    value = context.workspace.environment_info
    if isinstance(value, EnvironmentInfo):
        return value
    return EnvironmentInfo.collect_environment()


def _format_skills(workflow: SingleModeWorkflow) -> str:
    """Handle formatting available skills into a prompt section."""
    if len(workflow.skills) == 0:
        return ""


    skill_list : str = "\n".join(
        f"- **{skill.name}**: {skill.description}"
        for skill in workflow.skills
    )

    return f"""
# Available skills

{skill_list}
        
Call them as soon as they're relevant to the current task at hand
"""

def _tree_sections(context:ExecutionContext, aider_tree:bool, directory_tree:bool) -> str:
    """Creates the prompt section for both the aider tree and directory tree from the workspace's root"""
    if not aider_tree and not directory_tree:
        return ""

    result = """
    # Workspace
    """
    if aider_tree:
        result = result + f"""
        
        # Directory tree
        
        {render_tree(context.workspace, directories_only=True)}

        """
    if directory_tree:
        result = result + f"""
        # Aider tree

        {context.repo_map.render(model_id=context.model_config().id)}
        """

    return result + "\n" + "Use these maps as an initial snapshot of your workspace"

def _basic_coding_conventions() -> str:
    return """
# Coding conventions
Here are some basic coding conventions you should follow for all your code
    
## Modules
    
When creating a module, whether it be a single or multi file module include the following:
        
    1. **Name of the module**
    2. **Date of creation**
    3. **Author: ** sign your edits/creation with your assigned name
    4. **Modification history:** edit/create sections with your assigned developer name
    5. **Synopsis: ** What this module does.
    6. **Different functions supported in the module along with their input output parameters**
    7. **Global variables accessed or modified by the module**

This can be included directly in the source code in case of a single file module (ex.: api.py) or as the native module aggregator for the specific language 
(ex.: package-info.java, __init__.py, etc.). In case there's no particular module aggregator and the module is a directory, document it on an AGENTS.md file.

Always prefer lowering the overall exports of a module. Try to keep the code that references this module routed through an interface or interface-like class.
Public-access members must be justified, the default is module-private.

## Naming conventions
When naming classes/variables/functions, follow these standards:

    1. Meaningful and understandable variables name helps anyone to understand the reason of using it.
    2. Local variables should be named using camel case lettering starting with small letter (e.g. localData) whereas Global variables names should start with a capital letter (e.g. GlobalData). 
    Constant names should be formed using capital letters only (e.g. CONSDATA). 
    3. It is better to avoid the use of digits in variable names.
    4. The names of the function should be written in the particular language coding conventions.
    5. The name of the function must describe the reason of using the function clearly and briefly.
    
## Logging

Your functions must provide logging with the following characteristics:

    1. **Level** (trace, debug, info, warn, error). Keep most logging at a debug level.
    2. **Thread** on multi-threading scenarios.
    3. **Source** add the source of this log into the message or, if possible, at the logger utility input.
    4. **Message** accurate message. Must include what's happening and variable values.

Investigate first for the existing logging utility. If there's none, invent your own. This utility must persist the latest log to disk.

## Function logic

Prefer: 
    
    1. **Pure functions:** wherever possible, decompose functions into simpler, pure functions.
    2. **Indentation** 
        - There must be a space after giving a comma between two function arguments
        - Each nested block should be properly indented and spaced. The maximum nest depth is 3. Anything else should use a sub-function.
        - Proper Indentation should be there at the beginning and at the end of each block in the program.
        - All braces should start from a new line and the code following the end of braces also start from a new line.
    3. **Early-exit** prefer checking params at the first lines of the function, with early returns. Avoid it if it'll make the code harder to read.

## Documentation
All functions/classes must contain documentation. For classes, its just what they're for and their data.

For functions, include:
    
    1. **Parameter information** list parameters and what they're expected to represent/how they'll be used etc. Name assumptions if any.
    2. **What it does** an explanation of what this function does.
    3. **Return** what this functions returns if any.
    4. **Exceptions** list out any expected exception of this function, along with the cases under which they may be thrown.

All global variables must have their purpose documented.

A developer must be able to understand all of the function without knowing the code. Keep in-line code comments updated to the logic.
"""