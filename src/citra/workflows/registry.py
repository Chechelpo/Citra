"""Workflow registry and pre-sandbox startup selection."""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from pathlib import Path

from citra.modes import Mode, ModeRegistry

from .builtins import architect_workflow, simple_workflow
from .serial_roles import SerialRolesWorkflow
from .workflow import Workflow


WORKFLOW_CONFIG_FILE = "workflow.toml"
BUILTIN_DEFAULT_WORKFLOW = "simple"


class WorkflowRegistry:
    """Registry for workflows selected before runtime and sandbox creation."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        mode_registry: ModeRegistry | None = None,
        workflows: Iterable[Workflow] | None = None,
        default_workflow: str | None = None,
    ) -> None:
        self.mode_registry = mode_registry or ModeRegistry(
            config_path=config_path
        )
        installed = tuple(workflows) if workflows is not None else (
            simple_workflow(self.mode_registry.default_mode),
            SerialRolesWorkflow(),
            architect_workflow(),
        )
        self._workflows: dict[str, Workflow] = {}
        for workflow in installed:
            workflow.validate()
            if workflow.name in self._workflows:
                raise ValueError(f"Duplicate workflow name: {workflow.name!r}")
            self._workflows[workflow.name] = workflow
        if not self._workflows:
            raise ValueError("At least one workflow must be registered")

        configured_default = (
            default_workflow
            if default_workflow is not None
            else self._load_default(config_path)
        ) or BUILTIN_DEFAULT_WORKFLOW
        if configured_default not in self._workflows:
            raise ValueError(
                f"Unknown default workflow {configured_default!r}. Available "
                f"workflows: {', '.join(self._workflows)}"
            )
        self._default = self._workflows[configured_default]
        self._active = self._default

    @staticmethod
    def _config_path(config_path: str | Path | None) -> Path | None:
        if config_path is None:
            return None
        path = Path(config_path).expanduser().resolve()
        return (
            path / WORKFLOW_CONFIG_FILE
            if path.is_dir()
            else path.parent / WORKFLOW_CONFIG_FILE
        )

    @classmethod
    def _load_default(cls, config_path: str | Path | None) -> str | None:
        path = cls._config_path(config_path)
        if path is None or not path.is_file():
            return None
        with path.open("rb") as file:
            raw = tomllib.load(file)
        unexpected = set(raw) - {"default"}
        if unexpected:
            raise ValueError(
                f"{WORKFLOW_CONFIG_FILE} contains unsupported key(s): "
                + ", ".join(sorted(unexpected))
            )
        default = raw.get("default")
        if not isinstance(default, str) or not default.strip():
            raise ValueError(
                f"'{WORKFLOW_CONFIG_FILE}.default' must be a non-empty string"
            )
        return default.strip()

    @property
    def workflows(self) -> tuple[Workflow, ...]:
        return tuple(self._workflows.values())

    @property
    def default_workflow(self) -> Workflow:
        return self._default

    @property
    def active_workflow(self) -> Workflow:
        return self._active

    def get(self, name: str) -> Workflow:
        try:
            return self._workflows[name]
        except KeyError as error:
            raise KeyError(
                f"Unknown workflow {name!r}. Available workflows: "
                + ", ".join(self._workflows)
            ) from error

    def select(self, selection: str | None = None) -> Workflow:
        value = (selection or "").strip()
        if not value:
            self._active = self._default
            return self._active
        if value.isdecimal():
            index = int(value)
            if 1 <= index <= len(self._workflows):
                self._active = self.workflows[index - 1]
                return self._active
            raise ValueError(
                f"Workflow number must be between 1 and {len(self._workflows)}"
            )
        self._active = self.get(value)
        return self._active

    def set_simple_mode(self, mode: Mode) -> Workflow:
        """Replace the simple workflow's selected mode before provisioning."""
        workflow = simple_workflow(mode)
        self._workflows[workflow.name] = workflow
        if self._default.name == workflow.name:
            self._default = workflow
        if self._active.name == workflow.name:
            self._active = workflow
        return workflow


__all__ = [
    "BUILTIN_DEFAULT_WORKFLOW",
    "WORKFLOW_CONFIG_FILE",
    "WorkflowRegistry",
]
