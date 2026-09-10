"""
Tool for prompting the user for clarification, decisions, or approval.
"""
from __future__ import annotations

import json
from typing import Any, override

from ...agent.interactions import UserInteractionBroker
from ...cli.input import terminal_input
from ...cli.rendering import render_notice, render_question
from ...context import ExecutionContext
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ...utils.terminal import terminal_bell
from ..capabilities import ToolCapabilities
from ..tool import Tool

USER_UNAVAILABLE_MESSAGE = 'user-unavailable: no response was received within the timeout period. The user may be away. Pick the best answer yourself based on the question you asked.'
OPTION_SCHEMA = JsonSchema.object(properties=(JsonProperty(name='label', schema=JsonSchema.string(description='Short user-facing option label.')), JsonProperty(name='description', schema=JsonSchema.string(description='Short explanation of the option or its trade-offs.'))), additional_properties=False)
CLAUDE_QUESTION_SCHEMA = JsonSchema.object(properties=(JsonProperty(name='question', schema=JsonSchema.string(description='Complete question to ask the user.')), JsonProperty(name='header', schema=JsonSchema.string(description='Short UI label for the question.')), JsonProperty(name='options', schema=JsonSchema.array(OPTION_SCHEMA, description='Available choices.')), JsonProperty(name='multiSelect', schema=JsonSchema.boolean(description='Allow selecting multiple choices.'))), additional_properties=False)
KIMI_QUESTION_SCHEMA = JsonSchema.object(properties=(JsonProperty(name='question', schema=JsonSchema.string(description='Complete question to ask the user.')), JsonProperty(name='header', schema=JsonSchema.string(description='Short UI label.'), required=False), JsonProperty(name='options', schema=JsonSchema.array(OPTION_SCHEMA, description='Available choices.')), JsonProperty(name='multi_select', schema=JsonSchema.boolean(description='Allow selecting multiple choices.'), required=False)), additional_properties=False)
GEMINI_QUESTION_SCHEMA = JsonSchema.object(properties=(JsonProperty(name='question', schema=JsonSchema.string(description='Complete question to ask.')), JsonProperty(name='header', schema=JsonSchema.string(description='Short UI label.')), JsonProperty(name='type', schema=JsonSchema.string(description="Question type: 'choice', 'text', or 'yesno'. Defaults to 'choice'."), required=False), JsonProperty(name='options', schema=JsonSchema.array(OPTION_SCHEMA, description='Choices for a choice question.'), required=False), JsonProperty(name='multiSelect', schema=JsonSchema.boolean(description='Allow multiple selections for a choice question.'), required=False), JsonProperty(name='placeholder', schema=JsonSchema.string(description='Placeholder for free-form text input.'), required=False)), additional_properties=False)
OPENCODE_QUESTION_SCHEMA = JsonSchema.object(properties=(JsonProperty(name='question', schema=JsonSchema.string(description='Complete question.')), JsonProperty(name='header', schema=JsonSchema.string(description='Very short question label.')), JsonProperty(name='options', schema=JsonSchema.array(OPTION_SCHEMA, description='Available choices.')), JsonProperty(name='multiple', schema=JsonSchema.boolean(description='Allow selecting multiple choices.'), required=False)), additional_properties=False)
CODEX_QUESTION_SCHEMA = JsonSchema.object(properties=(JsonProperty(name='id', schema=JsonSchema.string(description='Stable snake_case identifier used to map the answer.')), JsonProperty(name='header', schema=JsonSchema.string(description='Short UI header.')), JsonProperty(name='question', schema=JsonSchema.string(description='Single-sentence question shown to the user.')), JsonProperty(name='options', schema=JsonSchema.array(OPTION_SCHEMA, description='Mutually exclusive choices. Do not add an Other option; custom input is available separately.'))), additional_properties=False)
ZCODE_OPTION_SCHEMA = JsonSchema.object(properties=(JsonProperty(name='label', schema=JsonSchema.string(description='User-facing option label.')), JsonProperty(name='value', schema=JsonSchema.string(description='Option value.'), required=False), JsonProperty(name='description', schema=JsonSchema.string(description='Optional explanation of the choice.'), required=False)), additional_properties=False)
ZCODE_QUESTION_SCHEMA = JsonSchema.object(properties=(JsonProperty(name='question', schema=JsonSchema.string(description='Question to ask the user.')), JsonProperty(name='header', schema=JsonSchema.string(description='Optional short UI label.'), required=False), JsonProperty(name='options', schema=JsonSchema.array(ZCODE_OPTION_SCHEMA, description='Available choices.'), required=False), JsonProperty(name='multiSelect', schema=JsonSchema.boolean(description='Allow selecting multiple choices.'), required=False)), additional_properties=False)

class PromptUser(Tool):
    """Represent PromptUser."""
    TOOL_ID = 'prompt_user'
    CAPABILITIES = ToolCapabilities()
    INVALIDATES_TOOL_CACHE = False
    DEFAULT_TIMEOUT_SECONDS = 270
    CITRA_DEFINITION = ChatCompletionTool(function=FunctionDefinition(name='prompt_user', description='Prompt the user for clarification, a decision, or permission. Provide options for a choice question or omit them for free-form input. If the user is unavailable, continue using your best judgment.', parameters=JsonSchema.object(properties=(JsonProperty(name='question', schema=JsonSchema.string(description='Question to present to the user.')), JsonProperty(name='options', schema=JsonSchema.array(JsonSchema.string(), description='Optional predefined choices.'), required=False), JsonProperty(name='timeout', schema=JsonSchema.integer(description='User inactivity timeout in seconds. Defaults to 30.'), required=False)), additional_properties=False)))

    @classmethod
    @override
    def definition_for_context(
        cls,
        context: ExecutionContext,
    ) -> ChatCompletionTool:
        """Return the tool's model-independent definition."""
        del context
        return cls.CITRA_DEFINITION

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize the instance."""
        super().__init__(context=context)

    def _normalize_questions(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Handle normalize questions."""
        if 'question' in arguments:
            options = [
                str(option).strip()
                for option in arguments.get('options') or []
            ]
            if any(not option for option in options):
                raise ValueError("Options cannot be empty.")
            return [{'id': '0', 'question': arguments['question'], 'options': options, 'multiple': False}]
        raw_questions = arguments.get('questions')
        if not isinstance(raw_questions, list):
            raise ValueError("'questions' must be an array.")
        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_questions):
            if not isinstance(raw, dict):
                raise ValueError('Each question must be an object.')
            question = str(raw.get('question', '')).strip()
            if not question:
                raise ValueError('Question text cannot be empty.')
            question_type = raw.get('type', 'choice')
            options_raw = raw.get('options') or []
            options: list[str] = []
            for option in options_raw:
                if isinstance(option, str):
                    label = option.strip()
                elif isinstance(option, dict):
                    label = str(option.get('label', option.get('value', ''))).strip()
                    description = str(option.get('description', '')).strip()
                    if label and description:
                        label = f'{label} — {description}'
                else:
                    continue
                if label:
                    options.append(label)
            if question_type == 'yesno':
                options = ['Yes', 'No']
            elif question_type == 'text':
                options = []
            multiple = bool(raw.get('multiSelect', raw.get('multi_select', raw.get('multiple', False))))
            normalized.append({'id': str(raw.get('id', index)), 'question': question, 'options': options, 'multiple': multiple})
        if not normalized:
            raise ValueError('At least one question is required.')
        return normalized

    def _ask_one(self, *, question: str, options: list[str], multiple: bool, timeout: int) -> str:
        """Handle ask one."""
        question = question.strip()
        if not question:
            raise ValueError("'question' cannot be empty.")
        if multiple and options:
            question = question + '\n' + 'You may select multiple choices; use a custom answer to provide multiple selections if the interface only permits one numbered choice.'
        broker = self.context.user_interactions
        if isinstance(broker, UserInteractionBroker):
            answer = broker.ask(question, tuple(options), timeout=timeout)
        else:
            if self.context.config.notifications.prompt_bell:
                terminal_bell()
            render_question(question, options)
            answer = terminal_input.prompt_with_idle_timeout(timeout=timeout, message='› ')
        if answer is None:
            if not isinstance(broker, UserInteractionBroker):
                render_notice(
                    f'No response within {timeout}s; continuing without user input.',
                    level='warning',
                )
            return USER_UNAVAILABLE_MESSAGE
        answer = answer.strip()
        if not answer:
            return '(empty response)'
        if options:
            resolved = self._resolve_option(answer, options)
            if resolved is not None:
                return resolved
        return answer

    @override
    def _execute(self, arguments: dict[str, Any]) -> str:
        """Execute the execute operation."""
        timeout = int(arguments.get('timeout', self.DEFAULT_TIMEOUT_SECONDS))
        if timeout <= 0:
            raise ValueError("'timeout' must be greater than zero.")
        questions = self._normalize_questions(arguments)
        answers: dict[str, str] = {}
        for question in questions:
            answer = self._ask_one(question=question['question'], options=question['options'], multiple=question['multiple'], timeout=timeout)
            answers[question['id']] = answer
            if answer == USER_UNAVAILABLE_MESSAGE:
                break
        if 'question' in arguments and len(answers) == 1:
            return next(iter(answers.values()))
        return json.dumps({'answers': answers}, ensure_ascii=False)

    @override
    def format_call_log(self, arguments: dict[str, Any]) -> str:
        """Handle format call log."""
        if 'questions' in arguments:
            questions = arguments.get('questions')
            count = len(questions) if isinstance(questions, list) else 0
            return f'questions={count}'
        question = str(arguments.get('question', ''))
        options = arguments.get('options')
        modality = 'option-list' if options else 'plain-text'
        parts = [f'mode={modality}', f'q={self._truncate(question)}']
        if options:
            parts.append(f'options={len(options)}')
        return ' | '.join(parts)

    @override
    def format_result_log(self, result: Any) -> str:
        """Handle format result log."""
        text = str(result)
        if USER_UNAVAILABLE_MESSAGE in text:
            return 'user-unavailable'
        if text == '(empty response)':
            return 'empty response'
        return self._truncate(text)

    @staticmethod
    def _truncate(value: str) -> str:
        """Handle truncate."""
        if len(value) <= 120:
            return value
        return value[:120] + '...'

    @staticmethod
    def _resolve_option(answer: str, options: list[str]) -> str | None:
        """Handle resolve option."""
        try:
            index = int(answer)
        except ValueError:
            return None
        if 1 <= index <= len(options):
            return options[index - 1]
        return None
