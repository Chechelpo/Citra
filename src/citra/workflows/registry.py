"""Unified workflow registry and pre-sandbox startup selection."""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from pathlib import Path

from .builtins import ArchitectWorkflow, ChatWorkflow, TaskWorkflow
from .serial_roles import SerialRolesWorkflow
from .workflow import Workflow


WORKFLOW_CONFIG_FILE = "workflow.toml"
BUILTIN_DEFAULT_WORKFLOW = "chat"


class WorkflowRegistry:
    """Registry for every selectable Citra execution workflow."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        workflows: Iterable[Workflow] | None = None,
        default_workflow: str | None = None,
    ) -> None:
        """Initialize the instance."""
        if config_path is not None and not isinstance(config_path, (str, Path)):
            raise TypeError("config_path must be a string or Path")
        if default_workflow is not None and (
            not isinstance(default_workflow, str)
            or not default_workflow.strip()
        ):
            raise ValueError("default_workflow must be a non-empty string")
        try:
            installed = tuple(workflows) if workflows is not None else (
                ChatWorkflow(),
                TaskWorkflow(),
                SerialRolesWorkflow(),
                ArchitectWorkflow(),
            )
        except TypeError as error:
            raise TypeError("workflows must be an iterable of Workflow objects") from error
        self._workflows: dict[str, Workflow] = {}
        for workflow in installed:
            if not isinstance(workflow, Workflow):
                raise TypeError("workflows must contain Workflow instances")
            workflow.validate()
            if workflow.name in self._workflows:
                raise ValueError(f"Duplicate workflow name: {workflow.name!r}")
            self._workflows[workflow.name] = workflow
        if not self._workflows:
            raise ValueError("At least one workflow must be registered")

        configured_default = (
            default_workflow.strip()
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
        """Handle config path."""
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
        """Handle load default."""
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
        """Handle workflows."""
        return tuple(self._workflows.values())

    @property
    def default_workflow(self) -> Workflow:
        """Handle default workflow."""
        return self._default

    @property
    def active_workflow(self) -> Workflow:
        """Handle active workflow."""
        return self._active

    def get(self, name: str) -> Workflow:
        """Handle get."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("workflow name must be a non-empty string")
        try:
            return self._workflows[name]
        except KeyError as error:
            raise KeyError(
                f"Unknown workflow {name!r}. Available workflows: "
                + ", ".join(self._workflows)
            ) from error

    def select(self, selection: str = "") -> Workflow:
        """Handle select."""
        if not isinstance(selection, str):
            raise TypeError("selection must be a string")
        value = selection.strip()
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


__all__ = [
    "BUILTIN_DEFAULT_WORKFLOW",
    "WORKFLOW_CONFIG_FILE",
    "WorkflowRegistry",
]
