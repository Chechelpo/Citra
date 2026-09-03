from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


JsonPrimitive = str | int | float | bool | None


class JsonType(str, Enum):
    """Represent JsonType."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    NULL = "null"


@dataclass(frozen=True)
class JsonProperty:
    """Represent JsonProperty."""
    name: str
    schema: JsonSchema
    required: bool = True


@dataclass(frozen=True)
class JsonSchema:
    """Represent JsonSchema."""
    type: JsonType

    description: str | None = None

    # Object-specific
    properties: tuple[JsonProperty, ...] = ()
    additional_properties: bool = False

    # Array-specific
    items: JsonSchema | None = None

    # Generic restrictions
    enum: tuple[JsonPrimitive, ...] = ()

    def __post_init__(self) -> None:
        """Validate and initialize the instance after construction."""
        if self.type is JsonType.ARRAY and self.items is None:
            raise ValueError("Array schemas must define 'items'.")

        if self.type is not JsonType.ARRAY and self.items is not None:
            raise ValueError(
                "'items' can only be defined for array schemas."
            )

        if self.type is not JsonType.OBJECT and self.properties:
            raise ValueError(
                "'properties' can only be defined for object schemas."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert the value to dict."""
        result: dict[str, Any] = {
            "type": self.type.value,
        }

        if self.description is not None:
            result["description"] = self.description

        if self.enum:
            result["enum"] = list(self.enum)

        if self.type is JsonType.OBJECT:
            result["properties"] = {
                prop.name: prop.schema.to_dict()
                for prop in self.properties
            }

            required = tuple(
                prop.name
                for prop in self.properties
                if prop.required
            )

            if required:
                result["required"] = list(required)

            result["additionalProperties"] = self.additional_properties

        elif self.type is JsonType.ARRAY:
            assert self.items is not None
            result["items"] = self.items.to_dict()

        return result

    @classmethod
    def string(
        cls,
        *,
        description: str | None = None,
        enum: tuple[str, ...] = (),
    ) -> JsonSchema:
        """Handle string."""
        return cls(
            type=JsonType.STRING,
            description=description,
            enum=enum,
        )

    @classmethod
    def integer(
        cls,
        *,
        description: str | None = None,
        enum: tuple[int, ...] = (),
    ) -> JsonSchema:
        """Handle integer."""
        return cls(
            type=JsonType.INTEGER,
            description=description,
            enum=enum,
        )

    @classmethod
    def number(
        cls,
        *,
        description: str | None = None,
    ) -> JsonSchema:
        """Handle number."""
        return cls(
            type=JsonType.NUMBER,
            description=description,
        )

    @classmethod
    def boolean(
        cls,
        *,
        description: str | None = None,
    ) -> JsonSchema:
        """Handle boolean."""
        return cls(
            type=JsonType.BOOLEAN,
            description=description,
        )

    @classmethod
    def array(
        cls,
        items: JsonSchema,
        *,
        description: str | None = None,
    ) -> JsonSchema:
        """Handle array."""
        return cls(
            type=JsonType.ARRAY,
            description=description,
            items=items,
        )

    @classmethod
    def object(
        cls,
        properties: tuple[JsonProperty, ...] = (),
        *,
        description: str | None = None,
        additional_properties: bool = False,
    ) -> JsonSchema:
        """Handle object."""
        return cls(
            type=JsonType.OBJECT,
            description=description,
            properties=properties,
            additional_properties=additional_properties,
        )


@dataclass(frozen=True)
class FunctionDefinition:
    """Represent FunctionDefinition."""
    name: str
    description: str
    parameters: JsonSchema
    strict: bool = False

    def __post_init__(self) -> None:
        """Validate and initialize the instance after construction."""
        if self.parameters.type is not JsonType.OBJECT:
            raise ValueError(
                "Function parameters must be an object schema."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert the value to dict."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_dict(),
            "strict": self.strict,
        }


@dataclass(frozen=True)
class ChatCompletionTool:
    """Represent ChatCompletionTool."""
    function: FunctionDefinition

    type: str = field(
        default="function",
        init=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the value to dict."""
        return {
            "type": self.type,
            "function": self.function.to_dict(),
        }