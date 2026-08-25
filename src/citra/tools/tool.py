from citra.utils.model_tokenizer import tokenize
from abc import ABC, abstractmethod
from hashlib import sha256
from time import perf_counter
from typing import Any, final
import logging
import json

from jsonschema import Draft202012Validator

from ..context import ExecutionContext
from ..utils.json_schema import ChatCompletionTool


logger = logging.getLogger(__name__)

_MAX_LOG_RESULT_LENGTH = 1000


class InvalidToolArguments(ValueError):
    pass


class Tool(ABC):
    HISTORY_ARGUMENT_COMPACT_THRESHOLD_TOKENS = 128
    HISTORY_ARGUMENT_DIGEST_LENGTH = 12

    # Tool-result cache policy. Cacheable tools may reuse an identical
    # result within the current agent turn. Tools are assumed to mutate
    # observable state unless they explicitly opt out; stale cache hits
    # are worse than unnecessary invalidation.
    CACHEABLE = False
    INVALIDATES_TOOL_CACHE = True
    MAX_OUTPUT_TOKENS: int | None = 4_000

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

    def compact_history_arguments(
        self,
        arguments: dict[str, Any],
        result: Any,
    ) -> dict[str, Any] | None:
        """
        Return compacted arguments to retain in model-facing history.

        This hook is called only after the tool has executed. Returning
        ``None`` preserves the model's original argument JSON exactly.
        Tools that persist large payloads elsewhere may return a schema-valid
        replacement mapping so the next model request never pays to replay
        those payloads.

        Implementations must not mutate ``arguments`` in place. They should
        normally compact only after a successful operation; failed calls keep
        their exact inputs so the model can reason about the failure.
        """
        del arguments, result
        return None

    def _compact_history_string_arguments(
        self,
        arguments: dict[str, Any],
        *names: str,
        min_token_savings: int = 64,
    ) -> dict[str, Any] | None:
        """
        Compact large string arguments only when doing so produces a meaningful
        reduction in their actual model-facing history cost.

        Cost is measured after both levels of JSON serialization:
        1. the tool arguments object -> function.arguments JSON string
        2. that JSON string -> model-facing tool-call JSON

        This is more accurate than character length or tokenizing the raw value,
        especially for source code containing quotes, newlines, or backslashes.

        Returns a new argument mapping when at least one field is worth
        compacting, otherwise None. Never mutates `arguments`.
        """
        model_id = self.context.config.model().id

        def history_tokens(
            candidate: dict[str, Any],
        ) -> int:
            # This is the representation stored in:
            #
            #   tool_call["function"]["arguments"]
            #
            arguments_json = json.dumps(
                candidate,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            # Model history sees `arguments_json` as a JSON string nested inside
            # the tool-call object, so serialize that level too. Include the
            # stable function envelope to account for tokenizer boundary effects.
            history_fragment = json.dumps(
                {
                    "type": "function",
                    "function": {
                        "name": self.id,
                        "arguments": arguments_json,
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )

            return tokenize(
                model_id,
                history_fragment,
            )

        compacted = dict(arguments)
        current_tokens = history_tokens(
            compacted
        )
        changed = False

        for name in names:
            value = compacted.get(name)

            if not isinstance(value, str):
                continue

            digest = sha256(
                value.encode("utf-8")
            ).hexdigest()[
                :self.HISTORY_ARGUMENT_DIGEST_LENGTH
            ]

            marker = (
                f"<citra: compacted {self.id}.{name}; "
                f"{len(value)} chars; "
                f"sha256={digest}>"
            )

            candidate = dict(compacted)
            candidate[name] = marker

            candidate_tokens = history_tokens(
                candidate
            )

            savings = (
                current_tokens
                - candidate_tokens
            )

            if savings < min_token_savings:
                continue

            compacted = candidate
            current_tokens = candidate_tokens
            changed = True

        return (
            compacted
            if changed
            else None
        )

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