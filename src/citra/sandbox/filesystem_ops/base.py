from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar, cast

from citra.utils.model_tokenizer import tokenize


OutputT = TypeVar("OutputT", bound="FilesystemOutput")


class FilesystemOutput(ABC):
    """Structured result returned by one filesystem operation.

    Budgeting intentionally lives here rather than in the filesystem client or
    operation inputs. The caller chooses the token budget when it renders the
    result for a model.
    """

    @classmethod
    @abstractmethod
    def from_payload(cls, payload: Any) -> "FilesystemOutput":
        """Parse the worker's structured JSON payload or raise ValueError."""

    @abstractmethod
    def to_payload(self) -> Any:
        """Return a JSON-serializable worker payload."""

    @abstractmethod
    def render(self) -> str:
        """Render the complete result in the legacy textual format."""

    def to_budgeted(self, token_count: int, model_id: str) -> str:
        """Render at most ``token_count`` model tokens of the legacy view.

        Both the budget and tokenizer model are chosen explicitly by the caller.
        Filesystem inputs and execution remain oblivious to model/token concerns.
        """
        if not isinstance(token_count, int) or token_count < 0:
            raise ValueError("'token_count' must be a non-negative integer.")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("'model_id' must be a non-empty string.")
        if token_count == 0:
            return ""

        text = self.render()

        if tokenize(model_id, text) <= token_count:
            return text

        marker = "\n… [truncated]"
        if tokenize(model_id, marker) > token_count:
            marker = ""

        # Tokenizers are not assumed to expose encode/decode primitives. A
        # binary search over character prefixes keeps this helper dependent only
        # on Citra's stable tokenize(model_id, text) API.
        low = 0
        high = len(text)
        while low < high:
            mid = (low + high + 1) // 2
            candidate = text[:mid] + marker
            if tokenize(model_id, candidate) <= token_count:
                low = mid
            else:
                high = mid - 1

        return text[:low] + marker


class FilesystemInput(ABC, Generic[OutputT]):
    """Typed operation input sent through the sandbox worker protocol."""

    operation: ClassVar[str]
    output_type: ClassVar[type[FilesystemOutput]]

    @classmethod
    @abstractmethod
    def parse(cls, arguments: dict[str, Any]) -> "FilesystemInput[OutputT]":
        """Parse model/tool arguments into this input or raise ValueError."""

    @abstractmethod
    def to_arguments(self) -> dict[str, Any]:
        """Serialize this input for the fixed worker wire protocol."""

    def parse_output(self, payload: Any) -> OutputT:
        return cast(OutputT, self.output_type.from_payload(payload))


def require_string(arguments: dict[str, Any], key: str) -> str:
    if not isinstance(arguments, dict):
        raise ValueError("Filesystem arguments must be a JSON object.")
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string.")
    return value


def optional_string(
    arguments: dict[str, Any],
    key: str,
    default: str,
) -> str:
    if not isinstance(arguments, dict):
        raise ValueError("Filesystem arguments must be a JSON object.")
    value = arguments.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string.")
    return value


def require_payload_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Filesystem output payload must be a JSON object.")
    return payload
