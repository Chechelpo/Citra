from abc import ABC, abstractmethod
from typing import Any, final
import logging

from jsonschema import Draft202012Validator

from ..context import ExecutionContext
from ..utils.json_schema import ChatCompletionTool


logger = logging.getLogger(__name__)

_MAX_LOG_RESULT_LENGTH = 1000


class InvalidToolArguments(ValueError):
    pass


class Tool(ABC):
    def __init__(
        self,
        context: ExecutionContext,
        definition: ChatCompletionTool,
    ):
        self.__context = context
        self.__definition = definition

        Draft202012Validator.check_schema(
            definition.function.parameters.to_dict()
        )

    @property
    def context(self) -> ExecutionContext:
        return self.__context

    @final
    def rebind_context(
        self,
        context: ExecutionContext,
    ) -> None:
        """Bind a reused session tool to the current agent-turn context."""
        self.__context = context

    @property
    def id(self) -> str:
        return self.__definition.function.name

    @property
    def description(self) -> str:
        return self.__definition.function.description

    def get_as_tool(self) -> dict[str, Any]:
        """
        Return the OpenAI-compatible tool definition
        exposed to the model.
        """
        return self.__definition.to_dict()

    def validate_arguments(
        self,
        arguments: dict[str, Any],
    ) -> None:
        validator = Draft202012Validator(
            self.__definition.function.parameters.to_dict()
        )

        errors = list(
            validator.iter_errors(arguments)
        )

        if not errors:
            return

        messages: list[str] = []

        for error in errors:
            path = ".".join(
                str(part)
                for part in error.absolute_path
            )

            if path:
                messages.append(
                    f"{path}: {error.message}"
                )
            else:
                messages.append(error.message)

        raise InvalidToolArguments(
            f"Invalid arguments for tool '{self.id}': "
            + "; ".join(messages)
        )

    @final
    def execute(
        self,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Validate model-provided arguments, execute the tool,
        and log the call.
        """
        self.validate_arguments(arguments)

        try:
            result = self._execute(arguments)
        except Exception as error:
            logger.exception(
                "[%s] %s -> ERROR: %s",
                self.id,
                arguments,
                error,
            )
            raise

        logger.info(
            "[%s] %s -> %s",
            self.id,
            arguments,
            self._truncate_log_value(result),
        )

        return result

    @staticmethod
    def _truncate_log_value(
        value: Any,
        max_length: int = _MAX_LOG_RESULT_LENGTH,
    ) -> str:
        text = str(value)

        if len(text) <= max_length:
            return text

        truncated_chars = len(text) - max_length

        return (
            text[:max_length]
            + f"... <truncated {truncated_chars} chars>"
        )

    @abstractmethod
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> Any:
        ...
