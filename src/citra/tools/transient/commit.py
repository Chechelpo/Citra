from typing import Any, override

from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..tool import Tool


class Commit(Tool):
    """Stage agent-workspace changes and apply them to the source."""

    ACTIONS = (
        "status",
        "diff",
        "stage",
        "stage_patch",
        "unstage",
        "apply",
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="commit",
            description=(
                "Inspect and stage changes in the isolated agent workspace, "
                "then apply only staged file updates to @source. This tool "
                "never creates a Git commit and never changes the source "
                "repository's index or history. Actions: status, diff, stage, "
                "stage_patch, unstage, and apply. Apply performs a conflict "
                "check against the content originally materialized this turn."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="action",
                        schema=JsonSchema.string(
                            description="Commit workflow action.",
                            enum=ACTIONS,
                        ),
                    ),
                    JsonProperty(
                        name="paths",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Agent-workspace-relative paths or pathspecs. "
                                "Required by stage and unstage; optional for "
                                "diff."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="staged",
                        schema=JsonSchema.boolean(
                            description=(
                                "For diff, show staged changes instead of "
                                "unstaged changes. Defaults to false."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="patch",
                        schema=JsonSchema.string(
                            description=(
                                "Unified diff to add partially to the private "
                                "staging index. Required by stage_patch."
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
        action: str = arguments["action"]
        changes = self.context.workspace.changes

        if action == "status":
            self._reject_unused(
                arguments,
                "paths",
                "staged",
                "patch",
            )
            return changes.status()

        if action == "diff":
            self._reject_unused(
                arguments,
                "patch",
            )
            return changes.diff(
                staged=arguments.get(
                    "staged",
                    False,
                ),
                paths=arguments.get(
                    "paths",
                    (),
                ),
            )

        if action == "stage":
            self._reject_unused(
                arguments,
                "staged",
                "patch",
            )
            return changes.stage(
                self._required_paths(arguments)
            )

        if action == "stage_patch":
            self._reject_unused(
                arguments,
                "paths",
                "staged",
            )
            patch = arguments.get("patch")

            if patch is None:
                raise ValueError(
                    "'patch' is required for action 'stage_patch'."
                )

            return changes.stage_patch(
                patch
            )

        if action == "unstage":
            self._reject_unused(
                arguments,
                "staged",
                "patch",
            )
            return changes.unstage(
                self._required_paths(arguments)
            )

        if action == "apply":
            self._reject_unused(
                arguments,
                "paths",
                "staged",
                "patch",
            )
            return changes.apply()

        raise ValueError(
            f"Unsupported commit action: {action}"
        )

    @staticmethod
    def _required_paths(
        arguments: dict[str, Any],
    ) -> list[str]:
        paths = arguments.get("paths")

        if not paths:
            raise ValueError(
                "'paths' is required for this commit action."
            )

        return paths

    @staticmethod
    def _reject_unused(
        arguments: dict[str, Any],
        *names: str,
    ) -> None:
        supplied = [
            name
            for name in names
            if name in arguments
        ]

        if supplied:
            raise ValueError(
                "Arguments not valid for this action: "
                + ", ".join(supplied)
            )
