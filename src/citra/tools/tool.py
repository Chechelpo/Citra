from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, make_dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any, ClassVar, final, Self, TypeVar, Generic, cast
import json
import logging
import sys

from citra.utils.model_tokenizer import tokenize
from jsonschema import Draft202012Validator

from ..context import ExecutionContext
from ..utils.json_schema import ChatCompletionTool, JsonSchema, JsonType
from .capabilities import InvalidToolCapabilities, ToolCapabilities


_MAX_LOG_RESULT_LENGTH = 1000


class InvalidToolArguments(ValueError):
    """Represent InvalidToolArguments."""
    pass


class InvalidToolDefinition(ValueError):
    """Represent InvalidToolDefinition."""
    pass

@dataclass(frozen=True)
class ToolArguments(Mapping[str, Any]):
    """Immutable, tool-owned arguments stored at the model boundary."""

    _TOOL_TYPE: ClassVar[type[Tool[Any]] | None] = None

    @classmethod
    def from_dict(
        cls,
        arguments: dict[str, Any],
    ) -> Self:
        if not is_dataclass(cls):
            raise TypeError(
                f"{cls.__name__} must be decorated with @dataclass."
            )

        try:
            return cls(**arguments)
        except (TypeError, ValueError) as error:
            raise InvalidToolArguments(
                f"Could not parse {cls.__name__}: {error}"
            ) from error

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary for JSON and sandbox boundaries."""
        return {
            name: value
            for name, value in asdict(self).items()
            if value is not None
        }

    def __getitem__(self, name: str) -> Any:
        if name not in {field.name for field in fields(self)}:
            raise KeyError(name)
        value = getattr(self, name)
        if value is None:
            raise KeyError(name)
        return value

    def __iter__(self) -> Iterator[str]:
        return (
            field.name
            for field in fields(self)
            if getattr(self, field.name) is not None
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)

    @classmethod
    def tool_type(
        cls,
    ) -> type[Tool[Self]]:
        tool_type = cls.__dict__.get("_TOOL_TYPE")
        if tool_type is None:
            raise TypeError(f"{cls.__name__} is not bound to a Tool.")
        return cast(type[Tool[Self]], tool_type)

    @classmethod
    def _bind_tool(
        cls,
        tool_type: type[Tool[Self]],
    ) -> None:
        existing = cls.__dict__.get("_TOOL_TYPE")
        if existing is not None and existing is not tool_type:
            raise TypeError(
                f"{cls.__name__} is already bound to {existing.__name__}; "
                f"cannot bind it to {tool_type.__name__}."
            )
        cls._TOOL_TYPE = tool_type


@dataclass(frozen=True)
class UnboundToolArguments(ToolArguments):
    """Typed fallback used when response parsing has no tool registry."""

    data: Mapping[str, Any]

    @classmethod
    def from_dict(cls, arguments: dict[str, Any]) -> Self:
        return cls(data=dict(arguments))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

    def __getitem__(self, name: str) -> Any:
        return self.data[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

ArgumentsT = TypeVar(
    "ArgumentsT",
    bound=ToolArguments,
)


def _arguments_type_from_definition(
    tool_type: type[Tool[Any]],
    definition: ChatCompletionTool,
) -> type[ToolArguments]:
    """Build the immutable argument record declared by a tool's JSON schema."""
    properties = definition.function.parameters.properties
    required = [property for property in properties if property.required]
    optional = [property for property in properties if not property.required]
    argument_fields: list[tuple[Any, ...]] = [
        (property.name, _python_type(property.schema))
        for property in required
    ]
    argument_fields.extend(
        (property.name, _python_type(property.schema) | None, None)
        for property in optional
    )
    name = f"{tool_type.__name__}Arguments"
    arguments_type = make_dataclass(
        name,
        argument_fields,
        bases=(ToolArguments,),
        frozen=True,
        slots=True,
        namespace={"__module__": tool_type.__module__},
    )
    arguments_type.__module__ = tool_type.__module__
    setattr(sys.modules[tool_type.__module__], name, arguments_type)
    return arguments_type


def _python_type(schema: JsonSchema) -> Any:
    """Translate the supported JSON-schema vocabulary into Python types."""
    if schema.type is JsonType.STRING:
        return str
    if schema.type is JsonType.INTEGER:
        return int
    if schema.type is JsonType.NUMBER:
        return float
    if schema.type is JsonType.BOOLEAN:
        return bool
    if schema.type is JsonType.ARRAY:
        assert schema.items is not None
        return list[_python_type(schema.items)]
    if schema.type is JsonType.OBJECT:
        return dict[str, Any]
    return Any

class Tool(ABC, Generic[ArgumentsT]):
    """Base for schema-selected, capability-aware, lifecycle-logged tools."""
    HISTORY_ARGUMENT_COMPACT_THRESHOLD_TOKENS : ClassVar[int] = 128
    HISTORY_ARGUMENT_DIGEST_LENGTH: ClassVar[int] = 12
    ARGUMENTS_TYPE: ClassVar[type[ToolArguments]]
    DEFINITION: ClassVar[ChatCompletionTool]

    # Stable Citra-internal identity.
    # This does NOT change when the model-facing function name changes.
    TOOL_ID: ClassVar[str]

    # Concrete action-dispatching tools override this immutable declaration.
    CAPABILITIES: ClassVar[ToolCapabilities] = ToolCapabilities()

    _DESCRIPTION : ClassVar[str] = ""
    _USE_DESCRIPTION : ClassVar[str | None] = None
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

    def __init_subclass__(
        cls,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)

        argument_type = cls.__dict__.get("ARGUMENTS_TYPE")

        if argument_type is None:
            definition = cls.__dict__.get("DEFINITION") or cls.__dict__.get(
                "CITRA_DEFINITION"
            )
            if isinstance(definition, ChatCompletionTool):
                argument_type = _arguments_type_from_definition(cls, definition)
                cls.ARGUMENTS_TYPE = argument_type

        # Allows abstract/intermediate Tool classes that don't
        # declare their own argument type.
        if argument_type is None:
            return

        if (
            not isinstance(argument_type, type)
            or not issubclass(argument_type, ToolArguments)
        ):
            raise TypeError(
                f"{cls.__name__}.ARGUMENTS_TYPE must be "
                f"a ToolArguments subclass."
            )

        argument_type._bind_tool(cast(type[Tool[ToolArguments]], cls))
        cls.Arguments = argument_type

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Arguments
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @classmethod
    def arguments_type(
        cls,
    ) -> type[ArgumentsT]:
        return cast(
            type[ArgumentsT],
            cls.ARGUMENTS_TYPE,
        )

    # -------------------------------------------------------------------------
    # Definition
    # -------------------------------------------------------------------------
    @classmethod
    def get_prompt_info(cls) -> str | None:
        return cls._USE_DESCRIPTION

    @classmethod
    def definition_for_context(
        cls,
        context: ExecutionContext,
    ) -> ChatCompletionTool:
        """Return this tool's single definition, adjusted for runtime context."""
        del context
        return cls.DEFINITION
        
    def definition_for_instance(
        self,
        context: ExecutionContext,
    ) -> ChatCompletionTool:
        """Return the definition when it depends on instance state."""
        return type(self).definition_for_context(context)

    def _resolve_definition(
        self,
        context: ExecutionContext,
    ) -> ChatCompletionTool:
        """Handle resolve definition."""
        definition = self.definition_for_instance(context)
        self._validate_definition(definition)
        return self.__capabilities.apply_to_definition(definition)

    @classmethod
    @final
    def resolve_definition_for_context(
        cls,
        context: ExecutionContext,
        capabilities: ToolCapabilities | None = None,
    ) -> ChatCompletionTool:
        """Resolve the public definition under an optional action restriction."""

        definition = cls.definition_for_context(context)
        cls._validate_definition(definition)
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
    def _validate_definition(
        cls,
        definition: ChatCompletionTool,
    ) -> None:
        """Validate the one model-independent definition."""
        if not isinstance(definition, ChatCompletionTool):
            raise InvalidToolDefinition(
                f"Tool '{cls.TOOL_ID}' did not produce a ChatCompletionTool."
            )
        Draft202012Validator.check_schema(
            definition.function.parameters.to_dict()
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
        arguments: ArgumentsT,
    ) -> bool:
        """Return whether is cacheable."""
        del arguments
        return self.CACHEABLE

    def invalidates_tool_cache(
        self,
        arguments: ArgumentsT,
    ) -> bool:
        """Handle invalidates tool cache."""
        del arguments
        return self.INVALIDATES_TOOL_CACHE

    # -------------------------------------------------------------------------
    # History compaction
    # -------------------------------------------------------------------------

    def compact_history_arguments(
        self,
        arguments: ArgumentsT,
        result: Any,
    ) -> ArgumentsT | None:
        """Handle compact history arguments."""
        del arguments, result
        return None

    def _compact_history_string_arguments(
        self,
        arguments: ArgumentsT,
        *names: str,
        min_token_savings: int = 128,
    ) -> ArgumentsT | None:
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

        compacted = arguments.to_dict()

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

        return self.arguments_type().from_dict(compacted) if changed else None

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_arguments(
        self,
        arguments: Mapping[str, Any],
    ) -> None:
        """Handle validate arguments."""
        argument_dict = dict(arguments)
        try:
            self.__capabilities.validate_arguments(argument_dict)
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
            validator.iter_errors(argument_dict),
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

    def parse_arguments(self, raw_arguments: str) -> ArgumentsT:
        """Parse this tool's model-emitted argument payload."""
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as error:
            raise InvalidToolArguments(
                f"Invalid JSON arguments for tool '{self.model_name}': {error}"
            ) from error
        if not isinstance(arguments, dict):
            raise InvalidToolArguments(
                f"Arguments for tool '{self.model_name}' must be a JSON object."
            )
        self.validate_argument_dict(arguments)
        return self.arguments_type().from_dict(arguments)

    def validate_argument_dict(self, arguments: dict[str, Any]) -> None:
        """Validate decoded arguments before constructing their typed record."""
        validator = Draft202012Validator(
            self.definition.function.parameters.to_dict()
        )
        errors = sorted(
            validator.iter_errors(arguments),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if errors:
            messages = [
                f"{'.'.join(str(part) for part in error.absolute_path)}: "
                f"{error.message}"
                if error.absolute_path
                else error.message
                for error in errors
            ]
            raise InvalidToolArguments(
                f"Invalid arguments for tool '{self.model_name}': "
                + "; ".join(messages)
            )

    @final
    def execute(
        self,
        arguments: ArgumentsT | dict[str, Any],
    ) -> Any:
        """Execute the execute operation."""
        if not isinstance(arguments, self.arguments_type()):
            self.validate_argument_dict(dict(arguments))
            arguments = self.arguments_type().from_dict(dict(arguments))
        self.validate_arguments(arguments)

        call_log = self.format_call_log(
            arguments.to_dict()
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
        arguments: ArgumentsT,
    ) -> Any:
        """Execute the execute operation."""
        ...

    def compact_if_over_budget(
        self,
        arguments: ArgumentsT,
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
