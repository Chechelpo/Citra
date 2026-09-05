"""
Tool for prompting the user for clarification, decisions, or approval.
"""
from __future__ import annotations
import json
from typing import Any, override
from ...agent.interactions import UserInteractionBroker
from ...context import ExecutionContext
from ..capabilities import ToolCapabilities
from ..tool import Tool, ToolDefinition
from ...utils.json_schema import ChatCompletionTool, FunctionDefinition, JsonProperty, JsonSchema
from ...utils.terminal import BLUE, BOLD, CYAN, DIM, RESET, YELLOW, terminal_bell
from ...utils.terminal_input import terminal_input
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
    CLAUDE_CODE_DEFINITION = ChatCompletionTool(function=FunctionDefinition(name='AskUserQuestion', description='Ask the user questions during execution to clarify requirements, gather preferences, or make decisions. Users can provide custom input in addition to the provided choices.', parameters=JsonSchema.object(properties=(JsonProperty(name='questions', schema=JsonSchema.array(CLAUDE_QUESTION_SCHEMA, description='Questions to ask the user.')),), additional_properties=False)))
    GEMINI_CLI_DEFINITION = ChatCompletionTool(function=FunctionDefinition(name='ask_user', description='Ask the user one or more questions to gather preferences, clarify requirements, or make decisions. Supports choices, free-form text, and yes/no questions.', parameters=JsonSchema.object(properties=(JsonProperty(name='questions', schema=JsonSchema.array(GEMINI_QUESTION_SCHEMA, description='Questions to ask.')),), additional_properties=False)))
    QWEN_CODE_DEFINITION = ChatCompletionTool(function=FunctionDefinition(name='ask_user_question', description='Ask the user structured questions when their input materially affects the next action.', parameters=JsonSchema.object(properties=(JsonProperty(name='questions', schema=JsonSchema.array(CLAUDE_QUESTION_SCHEMA, description='Questions to ask.')),), additional_properties=False)))
    KIMI_CODE_DEFINITION = ChatCompletionTool(function=FunctionDefinition(name='AskUserQuestion', description="Present structured questions and options to the user to collect preferences, decisions, or requirements. Use only when the user's choice genuinely affects subsequent actions.", parameters=JsonSchema.object(properties=(JsonProperty(name='questions', schema=JsonSchema.array(KIMI_QUESTION_SCHEMA, description='Questions to ask.')),), additional_properties=False)))
    CODEX_DEFINITION = ChatCompletionTool(function=FunctionDefinition(name='request_user_input', description='Request user input for questions that materially affect the implementation. Prefer a single focused question and provide concrete mutually exclusive options.', parameters=JsonSchema.object(properties=(JsonProperty(name='questions', schema=JsonSchema.array(CODEX_QUESTION_SCHEMA, description='Questions to show the user.')),), additional_properties=False)))
    ZCODE_DEFINITION = ChatCompletionTool(function=FunctionDefinition(name='askUserQuestion', description='Ask the user one or more questions during execution. Questions may provide suggested answers and may allow multiple selections.', parameters=JsonSchema.object(properties=(JsonProperty(name='questions', schema=JsonSchema.array(ZCODE_QUESTION_SCHEMA, description='Questions to ask.')),), additional_properties=False)))
    OPENCODE_DEFINITION = ChatCompletionTool(function=FunctionDefinition(name='question', description='Ask the user questions during execution to gather preferences, clarify ambiguous instructions, or obtain implementation decisions.', parameters=JsonSchema.object(properties=(JsonProperty(name='questions', schema=JsonSchema.array(OPENCODE_QUESTION_SCHEMA, description='Questions to ask.')),), additional_properties=False)))

    @classmethod
    @override
    def definitions_for_context(cls, context: ExecutionContext) -> tuple[ToolDefinition, ...]:
        """Handle definitions for context."""
        del context
        return (ToolDefinition(definition=cls.CLAUDE_CODE_DEFINITION, model_family_matchers=('claude',)), ToolDefinition(definition=cls.GEMINI_CLI_DEFINITION, model_family_matchers=('gemini',)), ToolDefinition(definition=cls.QWEN_CODE_DEFINITION, model_family_matchers=('qwen',)), ToolDefinition(definition=cls.KIMI_CODE_DEFINITION, model_family_matchers=('kimi', 'moonshot')), ToolDefinition(definition=cls.ZCODE_DEFINITION, model_family_matchers=('glm',)), ToolDefinition(definition=cls.CODEX_DEFINITION, model_family_matchers=('gpt', 'codex')), ToolDefinition(definition=cls.CITRA_DEFINITION))

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize the instance."""
        super().__init__(context=context)

    def _normalize_questions(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Handle normalize questions."""
        if 'question' in arguments:
            return [{'id': '0', 'question': arguments['question'], 'options': arguments.get('options') or [], 'multiple': False}]
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
            print()
            print(f'{CYAN}⏺{RESET} {BOLD}{question}{RESET}')
            if options:
                for index, option in enumerate(options, start=1):
                    print(f'  {DIM}{index}.{RESET} {option}')
                if multiple:
                    hint = 'Type a choice, or type your own answer containing multiple selections.'
                else:
                    hint = 'Type a number to select an option, or type your own answer.'
                print(f'\n{DIM}{hint}{RESET}')
            else:
                print(f'{DIM}(open-ended question){RESET}')
            answer = terminal_input.prompt_with_idle_timeout(timeout=timeout, message=f'{BOLD}{BLUE}❯{RESET} ')
        if answer is None:
            if not isinstance(broker, UserInteractionBroker):
                print(f'{YELLOW}⏺ (no response within {timeout}s — proceeding as user-unavailable){RESET}')
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
