from typing import Any, override

from ...context import ExecutionContext
from ..tool import Tool, ToolDefinition
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

    CLAUDE_CODE_DEFINITION = _skill_definition(
        name="Skill",
        skill_parameter="skill",
        include_args=True,
        description=(
            "Execute a skill within the main conversation. Use a skill "
            "when the task matches one of the available skills. The skill "
            "name must exactly match an available skill."
        ),
    )

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

    GEMINI_CLI_DEFINITION = _skill_definition(
        name="activate_skill",
        skill_parameter="name",
        description=(
            "Activate specialized procedural expertise and resources "
            "from an available skill when it is relevant to the task."
        ),
    )

    # ------------------------------------------------------------------
    # Qwen Code
    #
    # skill(
    #     skill,
    #     args?,
    # )
    # ------------------------------------------------------------------

    QWEN_CODE_DEFINITION = _skill_definition(
        name="skill",
        skill_parameter="skill",
        include_args=True,
        description=(
            "Execute a skill within the main conversation. Use this when "
            "the current task matches one of the available skills."
        ),
    )

    # ------------------------------------------------------------------
    # Kimi Code
    #
    # Skill(
    #     skill,
    #     args?,
    # )
    # ------------------------------------------------------------------

    KIMI_CODE_DEFINITION = _skill_definition(
        name="Skill",
        skill_parameter="skill",
        include_args=True,
        description=(
            "Invoke a registered Skill whose instructions are relevant "
            "to the current task. Optional argument text may be passed "
            "to the Skill."
        ),
    )

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

    ZCODE_DEFINITION = _skill_definition(
        name="Skill",
        skill_parameter="name",
        description=(
            "Load and apply one of the available skills to the current "
            "task. The name must exactly identify an available skill."
        ),
    )

    # ------------------------------------------------------------------
    # OpenCode
    #
    # Current source:
    #
    # skill(
    #     name,
    # )
    # ------------------------------------------------------------------

    OPENCODE_DEFINITION = _skill_definition(
        name="skill",
        skill_parameter="name",
        description=(
            "Load a specialized skill when the task matches one of the "
            "skills listed in available_skills."
        ),
    )

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

    CODEX_DEFINITION = _skill_definition(
        name="skill",
        skill_parameter="name",
        description=(
            "Load the full instructions for an available skill before "
            "performing the workflow it describes."
        ),
    )

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        """Handle definitions for context."""
        del context

        return (
            ToolDefinition(
                definition=cls.CLAUDE_CODE_DEFINITION,
                model_family_matchers=(
                    "claude",
                ),
            ),
            ToolDefinition(
                definition=cls.GEMINI_CLI_DEFINITION,
                model_family_matchers=(
                    "gemini",
                ),
            ),
            ToolDefinition(
                definition=cls.QWEN_CODE_DEFINITION,
                model_family_matchers=(
                    "qwen",
                ),
            ),
            ToolDefinition(
                definition=cls.KIMI_CODE_DEFINITION,
                model_family_matchers=(
                    "kimi",
                    "moonshot",
                ),
            ),
            ToolDefinition(
                definition=cls.ZCODE_DEFINITION,
                model_family_matchers=(
                    "glm",
                ),
            ),
            ToolDefinition(
                definition=cls.CODEX_DEFINITION,
                model_family_matchers=(
                    "gpt",
                    "codex",
                ),
            ),
            ToolDefinition(
                definition=cls.CITRA_DEFINITION,
            ),
        )

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