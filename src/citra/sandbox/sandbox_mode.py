from enum import IntEnum
from typing import Any

class SandboxMode(IntEnum):
    """Host visibility granted to a mode or workflow.

    The project itself is writable in both modes. ``PARTIAL_SANDBOX`` is for
    explicitly bounded worker contexts; normal user-facing modes use
    ``FULL_SANDBOX``. Historical full-access and direct-source modes no longer
    exist.
    """

    PARTIAL_SANDBOX = 1
    FULL_SANDBOX = 2
    
def _sandbox_mode(
    table: dict[str, Any],
    name: str,
    *,
    section: str,
    default: SandboxMode,
) -> SandboxMode:
    """Handle sandbox mode."""
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
