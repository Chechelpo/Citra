from typing import Any, override

from ...context import ExecutionContext
from ..capabilities import ToolCapabilities
from ..tool import Tool
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)


def _skill_definition(
    *,
    name: str,
    skill_parameter: str,
    include_args: bool = False,
    description: str,
) -> ChatCompletionTool:
    """Handle skill definition."""
    properties: list[JsonProperty] = [
        JsonProperty(
            name=skill_parameter,
            schema=JsonSchema.string(
                description=(
                    "Exact name of a skill listed in the "
                    "available-skills section."
                ),
            ),
        ),
    ]

    if include_args:
        properties.append(
            JsonProperty(
                name="args",
                schema=JsonSchema.string(
                    description=(
                        "Optional argument text to pass to the skill."
                    ),
                ),
                required=False,
            )
        )

    return ChatCompletionTool(
        function=FunctionDefinition(
            name=name,
            description=description,
            parameters=JsonSchema.object(
                properties=tuple(properties),
                additional_properties=False,
            ),
        ),
    )


class SkillTool(Tool):
    """
    Loads instructions for an available skill.
    """

    TOOL_ID = "skill"
    CAPABILITIES = ToolCapabilities()

    INVALIDATES_TOOL_CACHE = False

    # ------------------------------------------------------------------
    # Citra / OpenCode-compatible
    #
    # skill(
    #     name,
    # )
    # ------------------------------------------------------------------

    CITRA_DEFINITION = _skill_definition(
        name="skill",
        skill_parameter="name",
        description=(
            "Load a specialized skill when the current task matches one "
            "of the skills listed in the system prompt. The skill's full "
            "instructions are added to the current conversation."
        ),
    )

    # ------------------------------------------------------------------
    # Claude Code
    #
    # Skill(
    #     skill,
    #     args?,
    # )
    #
    # Claude Code publicly documents Skill as the model-facing tool.
    # Current tool-input/hook behavior uses skill + optional args.
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Gemini CLI
    #
    # activate_skill(
    #     name,
    # )
    #
    # Gemini dynamically constrains `name` to discovered skills with an
    # enum. Citra can add that later if its skill registry exposes names
    # during definition construction.
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Qwen Code
    #
    # skill(
    #     skill,
    #     args?,
    # )
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Kimi Code
    #
    # Skill(
    #     skill,
    #     args?,
    # )
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # ZCode / GLM
    #
    # ZCode definitely exposes an Agent-side Skill capability, but its
    # public documentation does not currently publish the complete JSON
    # tool schema.
    #
    # Preserve the observed capitalized Skill prior without inventing
    # unsupported argument semantics.
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # OpenCode
    #
    # Current source:
    #
    # skill(
    #     name,
    # )
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Codex
    #
    # Codex itself does not expose an equivalent model-callable Skill
    # function. Its native runtime normally tells the model to open the
    # applicable SKILL.md or injects a skill through an app-server input.
    #
    # Since Citra does provide a dedicated skill loader, retain Citra's
    # simple skill(name) contract rather than fabricating a Codex API.
    # ------------------------------------------------------------------


    @classmethod
    @override
    def definition_for_context(
        cls,
        context: ExecutionContext,
    ) -> ChatCompletionTool:
        """Return the tool's model-independent definition."""
        del context
        return cls.CITRA_DEFINITION

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        """Initialize the instance."""
        super().__init__(
            context=context,
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _skill_name(
        arguments: dict[str, Any],
    ) -> str:
        """Handle skill name."""
        value = arguments.get(
            "name",
            arguments.get("skill"),
        )

        if value is None:
            raise ValueError(
                "Skill invocation contained no skill name."
            )

        name = str(value).strip()

        if not name:
            raise ValueError(
                "Skill name cannot be empty."
            )

        return name

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Execute the execute operation."""
        name = self._skill_name(
            arguments
        )

        result = self.context.skills.get_skill(
            name,
            self.context,
        )

        args = arguments.get(
            "args"
        )

        if args is None:
            return result

        args_text = str(
            args
        ).strip()

        if not args_text:
            return result

        # Compatibility behavior until the skill registry gets a native
        # argument-expansion API supporting $ARGUMENTS / $0 / etc.
        return (
            result
            + "\n\n"
            + f"ARGUMENTS: {args_text}"
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Handle format call log."""
        name = self._skill_name(
            arguments
        )

        args = arguments.get(
            "args"
        )

        if args:
            return (
                f"skill={name} | "
                f"args={self._truncate(str(args))}"
            )

        return f"skill={name}"

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        """Handle format result log."""
        text = str(
            result
        )

        lines = text.splitlines()

        return (
            f"{len(lines)} lines | "
            f"{len(text)} chars"
        )

    @staticmethod
    def _truncate(
        value: str,
        limit: int = 120,
    ) -> str:
        """Handle truncate."""
        if len(value) <= limit:
            return value

        return (
            value[:limit]
            + "..."
        )
