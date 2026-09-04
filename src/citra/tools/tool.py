from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from hashlib import sha256
from time import perf_counter
from typing import Any, ClassVar, final
import json
import logging

from citra.utils.model_tokenizer import tokenize
from jsonschema import Draft202012Validator

from ..context import ExecutionContext
from ..utils.json_schema import ChatCompletionTool
from .capabilities import InvalidToolCapabilities, ToolCapabilities


_MAX_LOG_RESULT_LENGTH = 1000


class InvalidToolArguments(ValueError):
    """Represent InvalidToolArguments."""
    pass


class InvalidToolDefinition(ValueError):
    """Represent InvalidToolDefinition."""
    pass


@dataclass(frozen=True)
class ToolDefinition:
    """
    A model-facing representation of a tool.

    An empty `model_family_matchers` tuple acts as the fallback definition.
    More-specific model matchers take precedence over less-specific ones.
    `primary` resolves ambiguity between definitions with equal specificity.
    """
    definition: ChatCompletionTool
    model_family_matchers: tuple[str, ...] = ()
    primary: bool = False

    def __post_init__(self) -> None:
        """Validate and initialize the instance after construction."""
        if any(
            not matcher.strip()
            for matcher in self.model_family_matchers
        ):
            raise ValueError(
                "Tool model-family matchers cannot be empty strings."
            )

    def match_score(
        self,
        model_id: str,
    ) -> int | None:
        """
        Return match specificity.

        - None: does not match
        - 0: fallback definition
        - >0: length of the most-specific matching family string
        """
        if not self.model_family_matchers:
            return 0

        normalized_model_id = model_id.casefold()

        matches = [
            len(matcher)
            for matcher in self.model_family_matchers
            if matcher.casefold() in normalized_model_id
        ]

        if not matches:
            return None

        return max(matches)

    def with_name(
        self,
        name: str,
        *,
        model_family_matchers: tuple[str, ...] | None = None,
        primary: bool | None = None,
    ) -> ToolDefinition:
        """
        Create another model-facing definition with the same schema but a
        different function name.
        """
        return ToolDefinition(
            definition=replace(
                self.definition,
                function=replace(
                    self.definition.function,
                    name=name,
                ),
            ),
            model_family_matchers=(
                self.model_family_matchers
                if model_family_matchers is None
                else model_family_matchers
            ),
            primary=(
                self.primary
                if primary is None
                else primary
            ),
        )


class Tool(ABC):
    """Base for schema-selected, capability-aware, lifecycle-logged tools."""
    HISTORY_ARGUMENT_COMPACT_THRESHOLD_TOKENS : ClassVar[int] = 128
    HISTORY_ARGUMENT_DIGEST_LENGTH: ClassVar[int] = 12

    # Stable Citra-internal identity.
    # This does NOT change when the model-facing function name changes.
    TOOL_ID: ClassVar[str]

    # Concrete action-dispatching tools override this immutable declaration.
    CAPABILITIES: ClassVar[ToolCapabilities] = ToolCapabilities()

    _DESCRIPTION : ClassVar[str] = ""

    # Tool-result cache policy.
    CACHEABLE : ClassVar[bool] = False
    INVALIDATES_TOOL_CACHE: ClassVar[bool] = True
    MAX_OUTPUT_TOKENS: ClassVar[int | None] = 16_000

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        """Initialize the instance."""
        self.__context = context
        self.__capabilities = self._resolve_capabilities(None)
        self.__definition = self._resolve_definition(
            context
        )
        self._logger().debug(
            "Initialized tool '%s' with %d enabled capabilities",
            self.id,
            len(self.__capabilities.enabled_actions),
            extra={"origin": type(self).__module__},
        )

    # -------------------------------------------------------------------------
    # Definition
    # -------------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        """
        Return the model-facing definitions supported in this execution context.

        Runtime configuration belongs here.

        For example, Bash may omit the `network` and `reason` parameters when
        network access is globally enabled.

        Model-specific names or schemas should be returned as additional
        ToolDefinition entries.
        """
        ...
        
    def definitions_for_instance(
        self,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        """Handle definitions for instance."""
        return type(self).definitions_for_context(
            context
        )

    def _resolve_definition(
        self,
        context: ExecutionContext,
    ) -> ChatCompletionTool:
        """Handle resolve definition."""
        definitions = self.definitions_for_instance(
            context
        )

        definition = self._select_definition(
            context,
            definitions,
        )
        return self.__capabilities.apply_to_definition(definition)

    @classmethod
    @final
    def resolve_definition_for_context(
        cls,
        context: ExecutionContext,
        capabilities: ToolCapabilities | None = None,
    ) -> ChatCompletionTool:
        """Resolve the public definition under an optional action restriction."""

        definition = cls._select_definition(
            context,
            cls.definitions_for_context(context),
        )
        return cls._resolve_capabilities(capabilities).apply_to_definition(
            definition
        )

    @classmethod
    def _resolve_capabilities(
        cls,
        capabilities: ToolCapabilities | None,
    ) -> ToolCapabilities:
        """Bind a caller restriction to the class capability declaration."""
        declaration = cls.CAPABILITIES
        if not isinstance(declaration, ToolCapabilities):
            raise TypeError(
                f"Tool '{cls.TOOL_ID}' CAPABILITIES must be ToolCapabilities."
            )
        option = declaration if capabilities is None else capabilities
        return option.bind(declaration)

    @classmethod
    @final
    def configure_capabilities(
        cls,
        capabilities: ToolCapabilities | None = None,
    ) -> ToolCapabilities:
        """Return class capabilities with an optional restriction applied."""
        return cls._resolve_capabilities(capabilities)

    @classmethod
    def _select_definition(
        cls,
        context: ExecutionContext,
        definitions: tuple[ToolDefinition, ...],
    ) -> ChatCompletionTool:
        """Handle select definition."""
        model_id = context.config.model().id

        if not definitions:
            raise InvalidToolDefinition(
                f"Tool '{cls.TOOL_ID}' produced no definitions."
            )

        matched: list[
            tuple[int, ToolDefinition]
        ] = []

        for tool_definition in definitions:
            definition = tool_definition.definition

            Draft202012Validator.check_schema(
                definition.function.parameters.to_dict()
            )

            score = tool_definition.match_score(
                model_id
            )

            if score is not None:
                matched.append(
                    (
                        score,
                        tool_definition,
                    )
                )

        if not matched:
            raise InvalidToolDefinition(
                f"Tool '{cls.TOOL_ID}' has no definition "
                f"for model '{model_id}'."
            )

        best_score = max(
            score
            for score, _ in matched
        )

        candidates = [
            tool_definition
            for score, tool_definition in matched
            if score == best_score
        ]

        if len(candidates) == 1:
            return candidates[0].definition

        primary = [
            candidate
            for candidate in candidates
            if candidate.primary
        ]

        if len(primary) == 1:
            return primary[0].definition

        names = [
            candidate.definition.function.name
            for candidate in candidates
        ]

        if len(primary) > 1:
            raise InvalidToolDefinition(
                f"Tool '{cls.TOOL_ID}' has multiple primary definitions "
                f"for model '{model_id}': {names}"
            )

        raise InvalidToolDefinition(
            f"Tool '{cls.TOOL_ID}' has ambiguous definitions "
            f"for model '{model_id}': {names}"
        )

    @property
    def definition(self) -> ChatCompletionTool:
        """
        Exact definition exposed to the currently bound model.
        """
        return self.__definition

    @property
    def capabilities(self) -> ToolCapabilities:
        """Return the supported and enabled actions bound to this instance."""
        return self.__capabilities

    @property
    def id(self) -> str:
        """
        Stable internal Citra tool identifier.
        """
        return self.TOOL_ID

    @property
    def model_name(self) -> str:
        """
        Function name the current model actually sees.
        """
        return self.__definition.function.name

    @property
    def description(self) -> str:
        """Handle description."""
        return self.__definition.function.description

    def get_as_tool(self) -> dict[str, Any]:
        """Return get as tool."""
        return self.__definition.to_dict()

    def accepts_model_name(
        self,
        name: str,
    ) -> bool:
        """
        Useful when dispatching a model-emitted tool call.
        """
        return name == self.model_name

    # -------------------------------------------------------------------------
    # Context
    # -------------------------------------------------------------------------

    @property
    def context(self) -> ExecutionContext:
        """Handle context."""
        return self.__context

    @classmethod
    @final
    def registration_summary(cls) -> str:
        """Static registry summary, when a tool declares one."""

        return cls._DESCRIPTION

    @final
    def rebind_context(
        self,
        context: ExecutionContext,
    ) -> None:
        # Resolve first so a bad context cannot leave the tool half-rebound.
        """Handle rebind context."""
        definition = self._resolve_definition(
            context
        )

        self.__context = context
        self.__definition = definition
        self._logger().debug(
            "Rebound tool '%s' to execution context",
            self.id,
            extra={"origin": type(self).__module__},
        )

    @final
    def rebind_capabilities(
        self,
        capabilities: ToolCapabilities | None,
    ) -> None:
        """Apply a ToolSet action restriction without reconstructing the tool."""
        resolved = self._resolve_capabilities(capabilities)
        previous = self.__capabilities
        self.__capabilities = resolved
        try:
            definition = self._resolve_definition(self.__context)
        except Exception:
            self.__capabilities = previous
            raise
        self.__definition = definition
        self._logger().debug(
            "Rebound tool '%s' capabilities",
            self.id,
            extra={
                "origin": type(self).__module__,
                "enabled_actions": resolved.enabled_actions,
            },
        )

    # -------------------------------------------------------------------------
    # Cache policy
    # -------------------------------------------------------------------------

    
    def is_cacheable(
        self,
        arguments: dict[str, Any],
    ) -> bool:
        """Return whether is cacheable."""
        del arguments
        return self.CACHEABLE

    def invalidates_tool_cache(
        self,
        arguments: dict[str, Any],
    ) -> bool:
        """Handle invalidates tool cache."""
        del arguments
        return self.INVALIDATES_TOOL_CACHE

    # -------------------------------------------------------------------------
    # History compaction
    # -------------------------------------------------------------------------

    def compact_history_arguments(
        self,
        arguments: dict[str, Any],
        result: Any,
    ) -> dict[str, Any] | None:
        """Handle compact history arguments."""
        del arguments, result
        return None

    def _compact_history_string_arguments(
        self,
        arguments: dict[str, Any],
        *names: str,
        min_token_savings: int = 128,
    ) -> dict[str, Any] | None:
        """Handle compact history string arguments."""
        model_id = self.context.config.model().id

        def history_tokens(
            candidate: dict[str, Any],
        ) -> int:
            """Handle history tokens."""
            arguments_json = json.dumps(
                candidate,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            history_fragment = json.dumps(
                {
                    "type": "function",
                    "function": {
                        # Important: history contains the name the model saw,
                        # not Citra's canonical TOOL_ID.
                        "name": self.model_name,
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

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_arguments(
        self,
        arguments: dict[str, Any],
    ) -> None:
        """Handle validate arguments."""
        try:
            self.__capabilities.validate_arguments(arguments)
        except InvalidToolCapabilities as error:
            self._logger().warning(
                "Rejected disabled action for tool '%s': %s",
                self.id,
                error,
                extra={"origin": type(self).__module__},
            )
            raise InvalidToolArguments(
                f"Invalid arguments for tool '{self.model_name}': {error}"
            ) from error

        validator = Draft202012Validator(
            self.definition.function.parameters.to_dict()
        )

        errors = sorted(
            validator.iter_errors(arguments),
            key=lambda error: tuple(
                str(part)
                for part in error.absolute_path
            ),
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
                messages.append(
                    error.message
                )

        self._logger().warning(
            "Rejected invalid arguments for tool '%s': %s",
            self.id,
            "; ".join(messages),
            extra={"origin": type(self).__module__},
        )
        raise InvalidToolArguments(
            f"Invalid arguments for tool "
            f"'{self.model_name}': "
            + "; ".join(messages)
        )

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    @final
    def execute(
        self,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute the execute operation."""
        self.validate_arguments(
            arguments
        )

        call_log = self.format_call_log(
            arguments
        )

        operation_logger = self._logger()
        operation_logger.info(
            "[%s] START %s",
            self.id,
            call_log,
            extra={"origin": type(self).__module__},
        )

        started = perf_counter()

        try:
            result = self._execute(
                arguments
            )
        except Exception as error:
            elapsed = (
                perf_counter()
                - started
            )

            operation_logger.exception(
                "[%s] ERROR after %.3fs | %s | %s",
                self.id,
                elapsed,
                call_log,
                error,
                extra={"origin": type(self).__module__},
            )

            raise

        elapsed = (
            perf_counter()
            - started
        )

        result_log = self.format_result_log(
            result
        )

        operation_logger.info(
            "[%s] DONE in %.3fs | %s",
            self.id,
            elapsed,
            self._truncate_log_value(
                result_log
            ),
            extra={"origin": type(self).__module__},
        )

        return result

    @abstractmethod
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute the execute operation."""
        ...

    def compact_if_over_budget(
        self,
        arguments: dict[str, Any],
        tool_result: str,
    ) -> str:
        """
        Truncate the tool result if it exceeds this tool's token budget.

        Assumes:
            tokenize(text, tokenamount) -> int

        returns the token count of `text`.
        """
        del arguments

        token_budget = self.MAX_OUTPUT_TOKENS

        if token_budget is None:
            return tool_result

        if tokenize(model_id=self.context.model_config().id, text=tool_result) <= token_budget:
            return tool_result

        suffix = "\n... <truncated to fit tool token budget>"

        # Find the largest character prefix which, including the truncation
        # marker, remains within the token budget.
        low = 0
        high = len(tool_result)

        while low < high:
            mid = (low + high + 1) // 2

            candidate: str = (
                tool_result[:mid]
                + suffix
            )

            if tokenize(model_id=self.context.model_config().id, text=candidate) <= token_budget:
                low = mid
            else:
                high = mid - 1

        return (
            tool_result[:low]
            + suffix
        )
    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        """Handle format call log."""
        return str(arguments)

    def _logger(self) -> logging.Logger:
        """Return a logger named for the concrete tool's source module."""
        module = type(self).__module__
        logger_name = (
            module
            if module == "citra" or module.startswith("citra.")
            else f"citra.tools.external.{module}"
        )
        return logging.getLogger(logger_name)

    def format_result_log(
        self,
        result: Any,
    ) -> str:
        """Handle format result log."""
        return str(result)

    @staticmethod
    def _truncate_log_value(
        value: Any,
        max_length: int = _MAX_LOG_RESULT_LENGTH,
    ) -> str:
        """Handle truncate log value."""
        text = str(value)

        if len(text) <= max_length:
            return text

        truncated_chars = (
            len(text)
            - max_length
        )

        return (
            text[:max_length]
            + f"... <truncated {truncated_chars} chars>"
        )
