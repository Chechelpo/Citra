from pathlib import Path

from .readable_converter import ReadableConverter
from .pdf import PdfConverter


_CONVERTERS: tuple[ReadableConverter, ...] = (
    PdfConverter(),
)


def convert(
    path: Path,
) -> Path:
    """
    Convert a file into an LLM-readable text representation when needed.

    Files without a registered converter are returned unchanged.
    """

    path = path.resolve()

    for converter in _CONVERTERS:
        if converter.supports(
            path
        ):
            return converter.convert(
                path
            )

    return path