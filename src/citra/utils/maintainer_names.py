import secrets

_MODIFIERS = (
    "amber",
    "central",
    "clear",
    "eastern",
    "granite",
    "northern",
    "quiet",
    "silver",
    "steady",
    "verdant",
    "monarchical"
)

_NOUNS = (
    "archive",
    "atlas",
    "field",
    "harbor",
    "junction",
    "ledger",
    "meridian",
    "registry",
    "summit",
    "vector",
    "abbot"
)


def temporary_name() -> str:
    """Handle temporary name."""
    return (
        f"{secrets.choice(_MODIFIERS)}-"
        f"{secrets.choice(_NOUNS)}"
    )