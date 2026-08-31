from typing import Any
from enum import IntEnum

class SandboxMode(IntEnum):
    """
    Level of sandboxing required by a mode or workflow override.

    Each level includes the guarantees of the previous level.
    """

    FULL_ACCESS = 0
    ONLY_SOURCE = 1
    PARTIAL_SANDBOX = 2
    FULL_SANDBOX = 3
    ONLY_SANDBOX = 4
    ONLY_FULL_SANDBOX = 5

    @property
    def uses_direct_source(self) -> bool:
        """Whether the authoritative source is the active project root."""
        return self <= SandboxMode.ONLY_SOURCE

def _sandbox_mode(
    table: dict[str, Any],
    name: str,
    *,
    section: str,
    default: SandboxMode,
) -> SandboxMode:
    value = table.get(name, default.name)

    if isinstance(value, SandboxMode):
        return value

    if isinstance(value, int) and not isinstance(value, bool):
        try:
            return SandboxMode(value)
        except ValueError as error:
            raise ValueError(
                f"'{section}.{name}' is not a valid sandbox mode: {value}"
            ) from error

    if isinstance(value, str):
        normalized = value.strip().upper().replace("-", "_")

        try:
            return SandboxMode[normalized]
        except KeyError as error:
            valid = ", ".join(
                mode.name.lower()
                for mode in SandboxMode
            )

            raise ValueError(
                f"'{section}.{name}' must be one of: {valid}"
            ) from error

    raise ValueError(
        f"'{section}.{name}' must be a sandbox mode name or integer."
    )