"""Declarative action capabilities for model-facing tools."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from citra.logging import Logger
from citra.utils.json_schema import ChatCompletionTool, JsonProperty, JsonType


_logger = Logger(__name__)


class InvalidToolCapabilities(ValueError):
    """Report an invalid capability declaration or restriction."""


@dataclass(frozen=True)
class ToolCapabilities:
    """Declare supported actions and optionally include or exclude a subset.

    Tool classes use ``actions`` (plus optional model-facing aliases) to
    declare the complete action surface they implement. A ``ToolSet`` entry
    may instead provide ``include`` or ``exclude`` to narrow that surface.
    The registry binds the restriction to the class declaration before the
    tool is exposed or instantiated.
    """

    actions: tuple[str, ...] = ()
    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] | None = None
    action_arguments: tuple[str, ...] = ("action",)
    aliases: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate one immutable capability declaration or restriction."""
        self._validate_string_tuple("actions", self.actions)
        self._validate_optional_string_tuple("include", self.include)
        self._validate_optional_string_tuple("exclude", self.exclude)
        self._validate_string_tuple("action_arguments", self.action_arguments)

        if self.actions and not self.action_arguments:
            raise InvalidToolCapabilities(
                "Action capability declarations require an action argument name."
            )

        if self.include is not None and self.exclude is not None:
            _logger.error(
                "Tool capability restriction used inclusion and exclusion",
                include=self.include,
                exclude=self.exclude,
            )
            raise InvalidToolCapabilities(
                "Tool capabilities must use inclusion or exclusion, not both."
            )

        if not isinstance(self.aliases, tuple):
            raise TypeError("aliases must be a tuple")

        alias_names: list[str] = []
        for alias in self.aliases:
            if (
                not isinstance(alias, tuple)
                or len(alias) != 2
                or not all(isinstance(value, str) and value for value in alias)
            ):
                raise TypeError(
                    "aliases must contain non-empty (model_action, action) tuples"
                )
            alias_names.append(alias[0])

        self._raise_for_duplicates("aliases", tuple(alias_names))

        if self.actions:
            supported = set(self.actions)
            invalid_targets = tuple(
                canonical
                for _model_action, canonical in self.aliases
                if canonical not in supported
            )
            if invalid_targets:
                raise InvalidToolCapabilities(
                    "Capability aliases target unsupported actions: "
                    + ", ".join(invalid_targets)
                )
            self._validate_selection(supported)

        _logger.trace(
            "Validated tool capabilities",
            supported=len(self.actions),
            restricted=self.is_restricted,
        )

    @property
    def is_restricted(self) -> bool:
        """Return whether this object narrows the declared action set."""
        return self.include is not None or self.exclude is not None

    @property
    def enabled_actions(self) -> tuple[str, ...]:
        """Return enabled canonical actions in declaration order."""
        if self.include is not None:
            included = set(self.include)
            return tuple(action for action in self.actions if action in included)

        excluded = set(self.exclude or ())
        return tuple(action for action in self.actions if action not in excluded)

    def bind(self, declaration: ToolCapabilities) -> ToolCapabilities:
        """Bind an include/exclude restriction to a tool declaration."""
        if not isinstance(declaration, ToolCapabilities):
            raise TypeError("declaration must be ToolCapabilities")
        if declaration.is_restricted:
            raise InvalidToolCapabilities(
                "Tool class capability declarations cannot be restricted."
            )

        if self.actions and self.actions != declaration.actions:
            raise InvalidToolCapabilities(
                "A ToolSet capability option cannot redeclare supported actions."
            )
        if self.aliases and self.aliases != declaration.aliases:
            raise InvalidToolCapabilities(
                "A ToolSet capability option cannot redeclare action aliases."
            )
        if (
            self.action_arguments != ("action",)
            and self.action_arguments != declaration.action_arguments
        ):
            raise InvalidToolCapabilities(
                "A ToolSet capability option cannot redeclare action arguments."
            )

        bound = ToolCapabilities(
            actions=declaration.actions,
            include=self.include,
            exclude=self.exclude,
            action_arguments=declaration.action_arguments,
            aliases=declaration.aliases,
        )

        if bound.is_restricted and not declaration.actions:
            raise InvalidToolCapabilities(
                "This tool declares no selectable action capabilities."
            )
        if bound.is_restricted and not bound.enabled_actions:
            raise InvalidToolCapabilities(
                "A capability restriction must leave at least one action enabled."
            )

        _logger.debug(
            "Bound tool capability restriction",
            enabled=bound.enabled_actions,
            excluded=bound.exclude or (),
        )
        return bound

    def apply_to_definition(
        self,
        definition: ChatCompletionTool,
    ) -> ChatCompletionTool:
        """Restrict a model-facing action enum when this object is narrowed."""
        if not self.is_restricted:
            return definition

        parameters = definition.function.parameters
        properties: list[JsonProperty] = []
        found_action_argument = False

        for prop in parameters.properties:
            if prop.name not in self.action_arguments:
                properties.append(prop)
                continue

            found_action_argument = True
            if prop.schema.type is not JsonType.STRING:
                raise InvalidToolCapabilities(
                    f"Capability argument '{prop.name}' must use a string schema."
                )
            enum = self._restricted_model_actions(prop.schema.enum)
            if not enum:
                raise InvalidToolCapabilities(
                    f"Capability restriction leaves '{prop.name}' with no actions."
                )
            properties.append(
                replace(
                    prop,
                    schema=replace(prop.schema, enum=enum),
                )
            )

        if not found_action_argument:
            raise InvalidToolCapabilities(
                "Tool capability declaration does not match an action argument "
                f"in model-facing tool '{definition.function.name}'."
            )

        restricted = replace(
            definition,
            function=replace(
                definition.function,
                parameters=replace(parameters, properties=tuple(properties)),
            ),
        )
        _logger.debug(
            "Applied capabilities to model-facing definition",
            tool=definition.function.name,
            enabled=self.enabled_actions,
        )
        return restricted

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Reject an action disabled by this capability restriction."""
        if not self.is_restricted:
            return

        selected: object | None = None
        for name in self.action_arguments:
            if name in arguments:
                selected = arguments[name]
                break

        if not isinstance(selected, str):
            return

        canonical = self.canonical_action(selected)
        if canonical in self.enabled_actions:
            return

        _logger.warn(
            "Rejected disabled tool action",
            action=selected,
            canonical_action=canonical,
            enabled=self.enabled_actions,
        )
        raise InvalidToolCapabilities(
            f"Action '{selected}' is disabled; enabled actions: "
            + ", ".join(self.enabled_actions)
        )

    def canonical_action(self, action: str) -> str:
        """Translate one model-facing action alias to its canonical name."""
        for model_action, canonical in self.aliases:
            if action == model_action:
                return canonical
        return action

    def _restricted_model_actions(
        self,
        model_actions: tuple[Any, ...],
    ) -> tuple[str, ...]:
        """Filter an existing enum while retaining its model-facing spelling."""
        enabled = set(self.enabled_actions)
        if model_actions:
            return tuple(
                action
                for action in model_actions
                if isinstance(action, str)
                and self.canonical_action(action) in enabled
            )
        return self.enabled_actions

    def _validate_selection(self, supported: set[str]) -> None:
        """Validate configured includes or excludes against supported actions."""
        selected = self.include if self.include is not None else self.exclude
        if selected is None:
            return
        unknown = tuple(action for action in selected if action not in supported)
        if unknown:
            _logger.error(
                "Tool capability restriction named unsupported actions",
                actions=unknown,
            )
            raise InvalidToolCapabilities(
                "Unsupported tool capability actions: " + ", ".join(unknown)
            )

    @classmethod
    def _validate_optional_string_tuple(
        cls,
        name: str,
        values: tuple[str, ...] | None,
    ) -> None:
        """Validate an optional tuple of unique non-empty strings."""
        if values is None:
            return
        cls._validate_string_tuple(name, values)

    @classmethod
    def _validate_string_tuple(cls, name: str, values: tuple[str, ...]) -> None:
        """Validate a tuple of unique non-empty strings."""
        if not isinstance(values, tuple):
            raise TypeError(f"{name} must be a tuple")
        if any(not isinstance(value, str) or not value for value in values):
            raise TypeError(f"{name} must contain non-empty strings")
        cls._raise_for_duplicates(name, values)

    @staticmethod
    def _raise_for_duplicates(name: str, values: tuple[str, ...]) -> None:
        """Reject duplicate values in a capability tuple."""
        duplicates = tuple(
            value
            for index, value in enumerate(values)
            if value in values[:index]
        )
        if duplicates:
            raise InvalidToolCapabilities(
                f"{name} contains duplicate values: " + ", ".join(duplicates)
            )


__all__ = ["InvalidToolCapabilities", "ToolCapabilities"]
