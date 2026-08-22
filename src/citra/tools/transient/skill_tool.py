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
    Lists and reads the available skills.

    Skills are referenced by name rather than arbitrary filesystem paths.
    """

    ACTIONS = frozenset(
        {
            "list",
            "read",
        }
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="skill",
            description=(
                "List or read the skills available to the current agent. "
                "Use 'list' to discover available skill names and 'read' "
                "to load the instructions for one skill. Skills are "
                "read-only and are referenced by their skill name."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description=(
                                "Skill operation to perform."
                            ),
                            enum=(
                                "list",
                                "read",
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="name",
                        schema=JsonSchema.string(
                            description=(
                                "Name of the skill to read. Required for "
                                "action 'read' and invalid for action 'list'."
                            ),
                        ),
                        required=False,
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
        action = str(
            arguments["action"]
        ).strip()

        if action not in self.ACTIONS:
            raise ValueError(
                f"Unsupported skill action: {action!r}"
            )

        if action == "list":
            if "name" in arguments:
                raise ValueError(
                    "'name' is not valid for skill action 'list'."
                )

            return self.context.skills.list()

        name = str(
            arguments.get("name", "")
        ).strip()

        if not name:
            raise ValueError(
                "'name' is required for skill action 'read'."
            )

        return self.context.skills.get_skill(name, self.context)