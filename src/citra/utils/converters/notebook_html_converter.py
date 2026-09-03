from html.parser import HTMLParser

class NotebookHtmlTextExtractor(HTMLParser):
    """
    Minimal HTML-to-text conversion for notebook display outputs.
    """

    BLOCK_TAGS = frozenset(
        {
            "address",
            "article",
            "aside",
            "blockquote",
            "br",
            "div",
            "dl",
            "dt",
            "dd",
            "figcaption",
            "figure",
            "footer",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "hr",
            "li",
            "main",
            "nav",
            "ol",
            "p",
            "pre",
            "section",
            "table",
            "tr",
            "ul",
        }
    )

    def __init__(
        self,
    ) -> None:
        """Initialize the instance."""
        super().__init__(
            convert_charrefs=True
        )

        self.__parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[
            tuple[str, str | None]
        ],
    ) -> None:
        """Handle handle starttag."""
        if tag in self.BLOCK_TAGS:
            self.__newline()

        if tag in (
            "td",
            "th",
        ):
            if (
                self.__parts
                and not self.__parts[-1].endswith(
                    (
                        "\n",
                        "\t",
                    )
                )
            ):
                self.__parts.append(
                    "\t"
                )

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        """Handle handle endtag."""
        if tag in self.BLOCK_TAGS:
            self.__newline()

    def handle_data(
        self,
        data: str,
    ) -> None:
        """Handle handle data."""
        if data:
            self.__parts.append(
                data
            )

    def text(
        self,
    ) -> str:
        """Handle text."""
        text = "".join(
            self.__parts
        )

        lines = [
            line.rstrip()
            for line in text.splitlines()
        ]

        compact: list[str] = []
        previous_blank = False

        for line in lines:
            blank = not line.strip()

            if (
                blank
                and previous_blank
            ):
                continue

            compact.append(
                line
            )

            previous_blank = blank

        return "\n".join(
            compact
        ).strip()

    def __newline(
        self,
    ) -> None:
        """Handle newline."""
        if (
            self.__parts
            and not self.__parts[-1].endswith(
                "\n"
            )
        ):
            self.__parts.append(
                "\n"
            )