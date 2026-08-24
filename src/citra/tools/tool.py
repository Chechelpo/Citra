from abc import ABC, abstractmethod
from time import perf_counter
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
    # Tool-result cache policy. Cacheable tools may reuse an identical
    # result within the current agent turn. Tools are assumed to mutate
    # observable state unless they explicitly opt out; stale cache hits
    # are worse than unnecessary invalidation.
    CACHEABLE = False
    INVALIDATES_TOOL_CACHE = True
    MAX_OUTPUT_TOKENS: int | None = 2_000

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
        self.__context = context

    @property
    def id(self) -> str:
        return self.__definition.function.name

    @property
    def description(self) -> str:
        return self.__definition.function.description

    def get_as_tool(self) -> dict[str, Any]:
        return self.__definition.to_dict()

    def is_cacheable(
        self,
        arguments: dict[str, Any],
    ) -> bool:
        """Return whether an identical call may reuse a prior result."""
        del arguments
        return self.CACHEABLE

    def invalidates_tool_cache(
        self,
        arguments: dict[str, Any],
    ) -> bool:
        """Return whether this call may change state seen by cached tools."""
        del arguments
        return self.INVALIDATES_TOOL_CACHE

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
        self.validate_arguments(arguments)

        call_log = self.format_call_log(
            arguments
        )

        logger.info(
            "[%s] START %s",
            self.id,
            call_log,
        )

        started = perf_counter()

        try:
            result = self._execute(
                arguments
            )
        except Exception as error:
            elapsed = perf_counter() - started

            logger.exception(
                "[%s] ERROR after %.3fs | %s | %s",
                self.id,
                elapsed,
                call_log,
                error,
            )

            raise

        elapsed = perf_counter() - started

        result_log = self.format_result_log(
            result
        )

        logger.info(
            "[%s] DONE in %.3fs | %s",
            self.id,
            elapsed,
            self._truncate_log_value(
                result_log
            ),
        )

        return result

    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """
        Human-friendly description of the invocation.

        Tools can override this when the raw argument dict isn't useful.
        """
        return str(arguments)

    def format_result_log(
        self,
        result: Any,
    ) -> str:
        """
        Human-friendly description of the result.

        Tools should override this for structured/large results.
        """
        return str(result)

    @staticmethod
    def _truncate_log_value(
        value: Any,
        max_length: int = _MAX_LOG_RESULT_LENGTH,
    ) -> str:
        text = str(value)

        if len(text) <= max_length:
            return text

        truncated_chars = (
            len(text) - max_length
        )

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