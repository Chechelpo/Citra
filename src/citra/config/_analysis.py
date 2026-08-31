from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LspContextConfig:
    enabled: bool = True
    startup_timeout: float = 30.0
    request_timeout: float = 15.0
    diagnostics_timeout: float = 10.0
    cold_diagnostics_timeout: float = 45.0
    json_fallback: bool = True


@dataclass(frozen=True)
class LintRuleConfig:
    name: str
    command: tuple[str, ...]
    include: tuple[str, ...] = ("**/*",)
    exclude: tuple[str, ...] = ()
    cwd: str = "."


@dataclass(frozen=True)
class LintContextConfig:
    enabled: bool = True
    timeout: int = 30
    max_output_length: int = 20_000
    rules: tuple[LintRuleConfig, ...] = ()


def load_lsp_config(
    raw: dict[str, Any],
) -> LspContextConfig:
    value = raw.get("lsp", {})

    if not isinstance(value, dict):
        raise ValueError("'lsp' must be a TOML table.")

    config = LspContextConfig(
        enabled=_bool(
            value,
            "enabled",
            default=True,
            section="lsp",
        ),
        startup_timeout=_positive_float(
            value,
            "startup_timeout",
            default=30.0,
            section="lsp",
        ),
        request_timeout=_positive_float(
            value,
            "request_timeout",
            default=15.0,
            section="lsp",
        ),
        diagnostics_timeout=_positive_float(
            value,
            "diagnostics_timeout",
            default=10.0,
            section="lsp",
        ),
        cold_diagnostics_timeout=_positive_float(
            value,
            "cold_diagnostics_timeout",
            default=45.0,
            section="lsp",
        ),
        json_fallback=_bool(
            value,
            "json_fallback",
            default=True,
            section="lsp",
        ),
    )

    return config


def load_lint_config(
    raw: dict[str, Any],
) -> LintContextConfig:
    value = raw.get("lint", {})

    if not isinstance(value, dict):
        raise ValueError("'lint' must be a TOML table.")

    rules_raw = value.get(
        "rules",
        [],
    )

    if not isinstance(rules_raw, list):
        raise ValueError(
            "'lint.rules' must be an array of tables."
        )

    rules: list[LintRuleConfig] = []
    names: set[str] = set()

    for index, rule_raw in enumerate(rules_raw):
        section = f"lint.rules[{index}]"

        if not isinstance(rule_raw, dict):
            raise ValueError(
                f"'{section}' must be a TOML table."
            )

        name = rule_raw.get("name")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"'{section}.name' must be a non-empty string."
            )

        name = name.strip()

        if name in names:
            raise ValueError(
                f"Duplicate lint rule name: {name}"
            )

        names.add(name)

        command = rule_raw.get("command")

        if (
            not isinstance(command, list)
            or not command
            or not all(
                isinstance(item, str) and item
                for item in command
            )
        ):
            raise ValueError(
                f"'{section}.command' must be a non-empty array "
                "of non-empty strings."
            )

        include = _string_tuple(
            rule_raw,
            "include",
            section=section,
            default=("**/*",),
        )

        exclude = _string_tuple(
            rule_raw,
            "exclude",
            section=section,
            default=(),
        )

        cwd = rule_raw.get(
            "cwd",
            ".",
        )

        if not isinstance(cwd, str) or not cwd.strip():
            raise ValueError(
                f"'{section}.cwd' must be a non-empty string."
            )

        rules.append(
            LintRuleConfig(
                name=name,
                command=tuple(command),
                include=include,
                exclude=exclude,
                cwd=cwd,
            )
        )

    return LintContextConfig(
        enabled=_bool(
            value,
            "enabled",
            default=True,
            section="lint",
        ),
        timeout=_positive_int(
            value,
            "timeout",
            default=30,
            section="lint",
        ),
        max_output_length=_positive_int(
            value,
            "max_output_length",
            default=20_000,
            section="lint",
        ),
        rules=tuple(rules),
    )


def _bool(
    table: dict[str, Any],
    name: str,
    *,
    default: bool,
    section: str,
) -> bool:
    value = table.get(
        name,
        default,
    )

    if not isinstance(value, bool):
        raise ValueError(
            f"'{section}.{name}' must be a boolean."
        )

    return value


def _positive_int(
    table: dict[str, Any],
    name: str,
    *,
    default: int,
    section: str,
) -> int:
    value = table.get(
        name,
        default,
    )

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(
            f"'{section}.{name}' must be a positive integer."
        )

    return value


def _positive_float(
    table: dict[str, Any],
    name: str,
    *,
    default: float,
    section: str,
) -> float:
    value = table.get(
        name,
        default,
    )

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(
            f"'{section}.{name}' must be a positive number."
        )

    return float(value)


def _string_tuple(
    table: dict[str, Any],
    name: str,
    *,
    section: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = table.get(
        name,
        default,
    )

    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"'{section}.{name}' must be an array of strings."
        )

    if not all(
        isinstance(item, str)
        for item in value
    ):
        raise ValueError(
            f"'{section}.{name}' must contain only strings."
        )

    return tuple(value)