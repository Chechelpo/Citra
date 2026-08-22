"""Language-server executable definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ServerDefinition:
    id: str
    executable: str
    arguments: tuple[str, ...]
    install_hint: str

