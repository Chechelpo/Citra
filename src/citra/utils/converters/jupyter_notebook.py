import json
from typing import Any
from pathlib import Path
from .readable_converter import ReadableConverter
from .notebook_html_converter import NotebookHtmlTextExtractor

class JupyterNotebookConverter(ReadableConverter):
    """
    Convert Jupyter notebooks into a compact, LLM-readable text format.

    Cell order and source are preserved. Useful textual outputs are retained.
    Binary and image outputs are summarized instead of embedding their data.
    """

    VERSION = 1

    MAX_OUTPUT_CHARS = 20_000

    @property
    def extensions(
        self,
    ) -> frozenset[str]:
        return frozenset(
            {
                ".ipynb",
            }
        )

    def _convert(
        self,
        *,
        source: Path,
        destination: Path,
    ) -> None:
        with source.open(
            "r",
            encoding="utf-8",
        ) as file:
            notebook = json.load(
                file
            )

        if not isinstance(
            notebook,
            dict,
        ):
            raise ValueError(
                "Invalid Jupyter notebook: root must be an object."
            )

        cells = notebook.get(
            "cells"
        )

        if not isinstance(
            cells,
            list,
        ):
            raise ValueError(
                "Invalid Jupyter notebook: 'cells' must be an array."
            )

        metadata = notebook.get(
            "metadata",
            {},
        )

        with destination.open(
            "w",
            encoding="utf-8",
        ) as output:
            output.write(
                f"Jupyter Notebook: {source.name}\n"
                f"Cells: {len(cells)}\n"
            )

            self._write_metadata(
                output=output,
                metadata=metadata,
            )

            for index, cell in enumerate(
                cells,
                1,
            ):
                self._write_cell(
                    output=output,
                    index=index,
                    cell=cell,
                )

    def _write_metadata(
        self,
        *,
        output: Any,
        metadata: Any,
    ) -> None:
        if not isinstance(
            metadata,
            dict,
        ):
            return

        kernelspec = metadata.get(
            "kernelspec"
        )

        if isinstance(
            kernelspec,
            dict,
        ):
            kernel = (
                kernelspec.get("display_name")
                or kernelspec.get("name")
            )

            if kernel:
                output.write(
                    f"Kernel: {kernel}\n"
                )

        language_info = metadata.get(
            "language_info"
        )

        if isinstance(
            language_info,
            dict,
        ):
            language = language_info.get(
                "name"
            )

            version = language_info.get(
                "version"
            )

            if language:
                output.write(
                    f"Language: {language}"
                )

                if version:
                    output.write(
                        f" {version}"
                    )

                output.write(
                    "\n"
                )

    def _write_cell(
        self,
        *,
        output: Any,
        index: int,
        cell: Any,
    ) -> None:
        if not isinstance(
            cell,
            dict,
        ):
            output.write(
                "\n"
                f"===== Cell {index} [invalid] =====\n\n"
                "[Invalid notebook cell]\n"
            )
            return

        cell_type = cell.get(
            "cell_type",
            "unknown",
        )

        execution_count = cell.get(
            "execution_count"
        )

        heading = (
            f"===== Cell {index} "
            f"[{cell_type}"
        )

        if (
            cell_type == "code"
            and execution_count is not None
        ):
            heading += (
                f", execution={execution_count}"
            )

        heading += (
            "] ====="
        )

        output.write(
            f"\n{heading}\n\n"
        )

        source_text = self._join_text(
            cell.get(
                "source",
                "",
            )
        )

        if source_text:
            output.write(
                source_text
            )

            if not source_text.endswith(
                "\n"
            ):
                output.write(
                    "\n"
                )

        if cell_type != "code":
            return

        outputs = cell.get(
            "outputs",
            [],
        )

        if not isinstance(
            outputs,
            list,
        ):
            return

        for output_index, cell_output in enumerate(
            outputs,
            1,
        ):
            self._write_output(
                output=output,
                index=output_index,
                cell_output=cell_output,
            )

    def _write_output(
        self,
        *,
        output: Any,
        index: int,
        cell_output: Any,
    ) -> None:
        if not isinstance(
            cell_output,
            dict,
        ):
            return

        output_type = cell_output.get(
            "output_type",
            "unknown",
        )

        output.write(
            "\n"
            f"----- Output {index} [{output_type}] -----\n"
        )

        if output_type == "stream":
            text = self._join_text(
                cell_output.get(
                    "text",
                    "",
                )
            )

            self._write_bounded_text(
                output,
                text,
            )

            return

        if output_type == "error":
            error_name = cell_output.get(
                "ename"
            )

            error_value = cell_output.get(
                "evalue"
            )

            if error_name:
                output.write(
                    str(
                        error_name
                    )
                )

                if error_value:
                    output.write(
                        f": {error_value}"
                    )

                output.write(
                    "\n"
                )

            traceback = cell_output.get(
                "traceback"
            )

            if isinstance(
                traceback,
                list,
            ):
                text = "\n".join(
                    str(line)
                    for line in traceback
                )

                self._write_bounded_text(
                    output,
                    text,
                )

            return

        if output_type in (
            "execute_result",
            "display_data",
        ):
            data = cell_output.get(
                "data",
                {},
            )

            if isinstance(
                data,
                dict,
            ):
                self._write_display_data(
                    output=output,
                    data=data,
                )

            return

        output.write(
            "[Unsupported notebook output]\n"
        )

    def _write_display_data(
        self,
        *,
        output: Any,
        data: dict[str, Any],
    ) -> None:
        if "text/plain" in data:
            self._write_bounded_text(
                output,
                self._join_text(
                    data["text/plain"]
                ),
            )
            return

        if "text/markdown" in data:
            self._write_bounded_text(
                output,
                self._join_text(
                    data["text/markdown"]
                ),
            )
            return

        if "application/json" in data:
            value = data[
                "application/json"
            ]

            try:
                text = json.dumps(
                    value,
                    indent=2,
                    ensure_ascii=False,
                )
            except (
                TypeError,
                ValueError,
            ):
                text = str(
                    value
                )

            self._write_bounded_text(
                output,
                text,
            )
            return

        if "text/html" in data:
            html = self._join_text(
                data[
                    "text/html"
                ]
            )

            self._write_bounded_text(
                output,
                self._html_to_text(
                    html
                ),
            )
            return

        binary_types = sorted(
            mime
            for mime in data
            if self._is_binary_mime(
                mime
            )
        )

        if binary_types:
            output.write(
                "[Binary output omitted: "
                + ", ".join(
                    binary_types
                )
                + "]\n"
            )
            return

        if data:
            output.write(
                "[Unsupported output MIME types: "
                + ", ".join(
                    sorted(
                        data
                    )
                )
                + "]\n"
            )
            return

        output.write(
            "[Empty output]\n"
        )

    def _write_bounded_text(
        self,
        output: Any,
        text: str,
    ) -> None:
        text = self._normalize_text(
            text
        )

        if not text:
            output.write(
                "[Empty output]\n"
            )
            return

        if len(text) > self.MAX_OUTPUT_CHARS:
            omitted = (
                len(text)
                - self.MAX_OUTPUT_CHARS
            )

            text = (
                text[
                    :self.MAX_OUTPUT_CHARS
                ]
                + "\n"
                + (
                    f"[Output truncated: "
                    f"{omitted} characters omitted]"
                )
            )

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
    def _join_text(
        value: Any,
    ) -> str:
        if isinstance(
            value,
            str,
        ):
            return value

        if isinstance(
            value,
            list,
        ):
            return "".join(
                str(part)
                for part in value
            )

        if value is None:
            return ""

        return str(
            value
        )

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        return text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

    @staticmethod
    def _is_binary_mime(
        mime: str,
    ) -> bool:
        return (
            mime.startswith(
                "image/"
            )
            or mime.startswith(
                "audio/"
            )
            or mime.startswith(
                "video/"
            )
            or mime
            in {
                "application/pdf",
                "application/octet-stream",
            }
        )

    @staticmethod
    def _html_to_text(
        html: str,
    ) -> str:
        parser = NotebookHtmlTextExtractor()

        try:
            parser.feed(
                html
            )
            parser.close()
        except Exception:
            return html

        return parser.text()