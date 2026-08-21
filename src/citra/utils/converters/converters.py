from pathlib import Path

from citra.context import WorkspaceContext

from .readable_converter import ReadableConverter
from .jupyter_notebook import JupyterNotebookConverter
from .pdf import PdfConverter


_CONVERTERS: tuple[ReadableConverter, ...] = (
    PdfConverter(),
    JupyterNotebookConverter(),
)


def convert(
    path: Path,
    *,
    workspace: WorkspaceContext,
) -> Path:
    """
    Convert a file into an LLM-readable text representation when needed.

    Files without a registered converter are returned unchanged.
    """
    path = workspace.require_allowed_path(
        path
    )

    for converter in _CONVERTERS:
        if converter.supports(path):
            return converter.convert(
                path,
                workspace=workspace,
            )

    return path