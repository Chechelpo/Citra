from typing import Any, override

from ...context import ExecutionContext
from ..tool import Tool
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)


class SkillTool(Tool):
    """
    Loads instructions for an available skill.
    """

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="skill",
            description=(
                "Load the full instructions for one of the skills "
                "listed in the system prompt. Use a skill before "
                "performing the workflow it describes."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="name",
                        schema=JsonSchema.string(
                            description=(
                                "Name of the skill to load, exactly as "
                                "listed in the available-skills section."
                            ),
                        ),
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        super().__init__(
            context=context,
            definition=self.DEFINITION,
        )

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        name = str(
            arguments["name"]
        ).strip()

        if not name:
            raise ValueError(
                "'name' cannot be empty."
            )

        return self.context.skills.get_skill(
            name,
            self.context,
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        name = str(arguments.get("name", ""))
        return f"skill={name}"

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        text = str(result)
        lines = text.splitlines()
        return f"{len(lines)} lines"