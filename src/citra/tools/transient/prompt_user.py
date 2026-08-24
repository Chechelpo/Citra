# src/citra/tools/prompt_user.py

"""
Tool for prompting users when things are unclear/undefined or asking
for permission to do dangerous tasks.

Requested tool modalities:

  1. plain-text — the model asks an open-ended question that the user
     answers in the terminal.

  2. option-list — the model provides a list of options through the
     terminal.  The user then selects the number corresponding to the
     correct answer, or optionally switches to a plain-text answer even
     if it was initially option-list.

Regardless of modality, if the user is inactive for a configurable
period (default 30s) a "user-unavailable" message is returned to the
model, and it is asked to pick the best answer out of the query it has
made.  The timeout is an *inactivity* timeout: any buffer modification
resets it, so an actively typing user never times out.
"""

from __future__ import annotations

from typing import Any, override

from ...agent.interactions import UserInteractionBroker
from ...context import ExecutionContext
from ..tool import Tool
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ...utils.terminal import (
    BLUE,
    BOLD,
    CYAN,
    DIM,
    RESET,
    YELLOW,
    terminal_bell,
)
from ...utils.terminal_input import terminal_input


USER_UNAVAILABLE_MESSAGE = (
    "user-unavailable: no response was received within the timeout "
    "period. The user may be away. Pick the best answer yourself based "
    "on the question you asked."
)


class PromptUser(Tool):
    INVALIDATES_TOOL_CACHE = False

    """
    Prompts the terminal user for input.

    Two modalities are supported, selected by whether the model supplies
    ``options``:

    * **plain-text** (no ``options``): an open-ended question is printed
      and the user's free-form answer is returned.

    * **option-list** (``options`` provided): a numbered list is printed.
      The user may either type the number of the desired option or type a
      free-form answer to override the list.

    A timeout (default 30 seconds) applies to both modalities.  The
    timeout is an *inactivity* timeout: every keystroke / buffer edit
    resets it.  When the user stays idle for the full interval, a
    ``user-unavailable`` message is returned so the model can proceed
    autonomously.
    """

    DEFAULT_TIMEOUT_SECONDS = 30

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="prompt_user",
            description=(
                "Prompt the terminal user for input. Use this when "
                "something is unclear or undefined, or when asking for "
                "permission to perform a potentially dangerous task. "
                "Two modalities are supported: 'plain-text' (open-ended "
                "question) when no options are given, and 'option-list' "
                "(numbered selection) when options are provided. In "
                "option-list mode the user may either type the number of "
                "an option or type a free-form answer. If the user stays "
                "inactive for the timeout (default 30s), a "
                "'user-unavailable' message is returned and the model "
                "should pick the best answer itself. The timeout is an "
                "inactivity timeout: any keystroke resets it."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="question",
                        schema=JsonSchema.string(
                            description=(
                                "The question to present to the user."
                            ),
                        ),
                    ),
                    JsonProperty(
                        name="options",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Optional list of predefined options. "
                                "When provided, the tool operates in "
                                "option-list mode: the user selects a "
                                "number or types a free-form answer. "
                                "When omitted, the tool operates in "
                                "plain-text mode."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="timeout",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum number of seconds of user "
                                "inactivity to wait before proceeding. "
                                "Defaults to 30. The timer is reset on "
                                "every keystroke, so an active user never "
                                "times out. If the user stays idle for the "
                                "full interval, a 'user-unavailable' "
                                "message is returned."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    def __init__(self, context: ExecutionContext):
        super().__init__(
            context=context,
            definition=self.DEFINITION,
        )

    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        question: str = arguments["question"].strip()

        if not question:
            raise ValueError("'question' cannot be empty.")

        options: list[str] | None = arguments.get("options")
        timeout: int = arguments.get(
            "timeout",
            self.DEFAULT_TIMEOUT_SECONDS,
        )

        if timeout <= 0:
            raise ValueError("'timeout' must be greater than zero.")

        if options is not None:
            cleaned = [opt.strip() for opt in options]

            if not cleaned:
                raise ValueError(
                    "'options' must contain at least one option."
                )

            # A blank-only option carries no usable information; reject
            # it consistently rather than rendering an empty menu entry.
            if any(not opt for opt in cleaned):
                raise ValueError(
                    "'options' must not contain blank entries."
                )

            options = cleaned

        broker = self.context.user_interactions

        if isinstance(broker, UserInteractionBroker):
            answer = broker.ask(
                question,
                tuple(options or ()),
                timeout=timeout,
            )
        else:
            # Non-REPL callers retain the direct behavior; the interactive
            # application uses a broker so only the foreground thread ever
            # owns prompt_toolkit.
            config = getattr(self.context, "config", None)
            notifications = getattr(config, "notifications", None)
            if notifications is None or getattr(
                notifications,
                "prompt_bell",
                True,
            ):
                terminal_bell()
            print()
            print(f"{CYAN}⏺{RESET} {BOLD}{question}{RESET}")

            if options:
                for index, option in enumerate(options, start=1):
                    print(f"  {DIM}{index}.{RESET} {option}")
                print(
                    f"\n{DIM}Type a number to select an option, "
                    f"or type your own answer.{RESET}"
                )
            else:
                print(f"{DIM}(open-ended question){RESET}")

            answer = terminal_input.prompt_with_idle_timeout(
                timeout=timeout,
                message=f"{BOLD}{BLUE}❯{RESET} ",
            )

        if answer is None:
            if not isinstance(broker, UserInteractionBroker):
                print(
                    f"{YELLOW}⏺ (no response within "
                    f"{timeout}s — proceeding as user-unavailable)"
                    f"{RESET}"
                )
            return USER_UNAVAILABLE_MESSAGE

        answer = answer.strip()

        if not answer:
            return "(empty response)"

        # In option-list mode, resolve a numeric selection to the
        # corresponding option text.
        if options:
            resolved = self._resolve_option(
                answer,
                options,
            )

            if resolved is not None:
                return resolved

        return answer

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        question = str(arguments.get("question", ""))
        options = arguments.get("options")

        modality = "option-list" if options else "plain-text"
        parts = [f"mode={modality}", f"q={self._truncate(question)}"]

        if options:
            parts.append(f"options={len(options)}")

        return " | ".join(parts)

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        text = str(result)

        if text == USER_UNAVAILABLE_MESSAGE:
            return "user-unavailable"

        if text == "(empty response)":
            return "empty response"

        return self._truncate(text)

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= 120:
            return value
        return value[:120] + "..."

    @staticmethod
    def _resolve_option(
        answer: str,
        options: list[str],
    ) -> str | None:
        """
        If *answer* is a 1-based index into *options*, return the
        matching option text.  Otherwise return ``None`` so the raw
        free-form answer is used.
        """
        try:
            index = int(answer)
        except ValueError:
            return None

        if 1 <= index <= len(options):
            return options[index - 1]

        return None
