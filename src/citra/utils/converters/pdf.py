from pypdf import PdfReader
from .readable_converter import ReadableConverter
from pathlib import Path

class PdfConverter(ReadableConverter):
    """
    Convert digitally-generated PDFs into layout-preserving UTF-8 text.

    Image-only pages are marked explicitly rather than silently omitted.
    """

    VERSION = 1

    @property
    def extensions(
        self,
    ) -> frozenset[str]:
        return frozenset(
            {
                ".pdf",
            }
        )

    def _convert(
        self,
        *,
        source: Path,
        destination: Path,
    ) -> None:
        reader = PdfReader(
            source
        )

        with destination.open(
            "w",
            encoding="utf-8",
        ) as output:
            output.write(
                f"PDF: {source.name}\n"
                f"Pages: {len(reader.pages)}\n"
            )

            metadata = reader.metadata

            if metadata:
                title = metadata.title

                if title:
                    output.write(
                        f"Title: {title}\n"
                    )

                author = metadata.author

                if author:
                    output.write(
                        f"Author: {author}\n"
                    )

            for index, page in enumerate(
                reader.pages,
                1,
            ):
                output.write(
                    "\n"
                    f"===== Page {index} =====\n\n"
                )

                try:
                    text = page.extract_text(
                        extraction_mode="layout",
                        layout_mode_space_vertically=False,
                    )

                except Exception as error:
                    output.write(
                        "[PDF text extraction failed for "
                        f"this page: {error}]\n"
                    )
                    continue

                if not text:
                    output.write(
                        "[No extractable text. "
                        "This page may be scanned or image-only.]\n"
                    )
                    continue

                text = self._normalize_text(
                    text
                )

                if not text:
                    output.write(
                        "[No extractable text. "
                        "This page may be scanned or image-only.]\n"
                    )
                    continue

                output.write(
                    text
                )

                if not text.endswith(
                    "\n"
                ):
                    output.write(
                        "\n"
                    )

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Keep layout extraction mostly intact while cleaning representation
        artifacts that are useless to an LLM.
        """

        text = text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        lines = [
            line.rstrip()
            for line in text.splitlines()
        ]

        while (
            lines
            and not lines[0]
        ):
            lines.pop(
                0
            )

        while (
            lines
            and not lines[-1]
        ):
            lines.pop()

        return "\n".join(
            lines
        )