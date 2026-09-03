"""Public controller source-baseline API."""

from .workspace_context.source_baseline import (
    MISSING_SOURCE_ENTRY,
    SourceEntry,
    capture_source_baseline,
    git_project_inventory,
    normalize_project_path,
    project_entry_path,
    snapshot_source_entry,
)

__all__ = [
    "MISSING_SOURCE_ENTRY",
    "SourceEntry",
    "capture_source_baseline",
    "git_project_inventory",
    "normalize_project_path",
    "project_entry_path",
    "snapshot_source_entry",
]
