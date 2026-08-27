
from typing import override

from citra.context import ExecutionContext
from citra.modes.mode import SandboxConfig, SandboxMode, StaticMode
from citra.tools.default_registry import all_tools
from citra.tools.transient import Commit, Diagram, Document
from citra.utils.prompt import PromptEnvironment, collect_environment

class DebuggerMode(StaticMode):

    _NAME = "debugger"
    _DESCRIPTION = "Provides tools and a workspace to debug the source, with no way to modify it."

    _CORE_TOOLS = all_tools(excluded={Commit}, are_deferred=False)
    _ALLOWED_TOOLS = all_tools(excluded={Document, Diagram}, are_deferred=True)

    _SANDBOX_CONFIG = SandboxConfig(
        mode=SandboxMode.PARTIAL_SANDBOX
    )

    @override
    def get_system_prompt(self, context: ExecutionContext) -> str:
        environment: PromptEnvironment = collect_environment(context)
        return f"""
# Task

You are a coding senior and you've been tasked with reviewing this codebase. Use available tools in order to make sense of it first.

## Environment information

{environment.as_prompt_section()}

## Exploration

Here's a tiered pipeline in order to explore the codebase. The user may provide you with enough context to skip parts of the pipeline marked as optional

This depends on the user's input. There's types of user input to debug:

#### Feature (optional)
A user-facing feature is failing. This task requires three inputs:

    1. **What the user was doing before using this feature**
    2. **What the user expected**
    3. **What happened instead of the expected functionality** 

If the user hasn't provided these three points, prompt him on them before starting the task (unless he has provided the files directly to debug). If the prompt fails, continue assuming the ones missing. 
Record these three to your facts about the task.

#### Source code and initial tests 
These are together as you can't really separate these two. The user may provide files to debug, or these may have been part of the feature exploration.
For this files, they may be tests that can give you a vague understanding of what the issue is.

IF they exist, try executing them before anything. Also try out related tests.

#### Reading
Read the files from your exploration while diagnosing them (LSP will type-check for you), identifying any possible issues. Write down in your facts how it works.
As you read, collect apparent bugs in your todo list.

#### Test
Write your own tests to verify every one of your found bugs (unless they're apparent). Do not stop at the first bug, rather check every single one in your debugs. Check them as you test them.

# Constraints

    1. Prefer the Tree and Glob tools over reading the source files directly. 
    2. Prefer focusing on different files before reading files
    3. Once a particular chain of functions is detected, you can read files

"""

_PROMPT = """

Use Tree to

"""
