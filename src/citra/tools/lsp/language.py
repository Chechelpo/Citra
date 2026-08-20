"""Language identification for the LSP subsystem.

Only Python is wired up with a server adapter in this phase.  The
:class:`Language` enum keeps additional language entries so future
adapters can be added without changing call sites.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class Language(str, Enum):
    """A source language supported by the LSP subsystem."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    HTML = "html"
    CSS = "css"
    VUE = "vue"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return _EXTENSIONS[self]

    @property
    def language_id(self) -> str:
        """The LSP ``LanguageId`` string used in ``textDocument`` notifications."""
        return _LANGUAGE_IDS[self]


_EXTENSIONS: dict[Language, tuple[str, ...]] = {
    Language.PYTHON: (".py", ".pyi"),
    Language.TYPESCRIPT: (".ts", ".tsx"),
    Language.JAVASCRIPT: (".js", ".jsx", ".mjs", ".cjs"),
    Language.JAVA: (".java",),
    Language.HTML: (".html", ".htm"),
    Language.CSS: (".css",),
    Language.VUE: (".vue",),
}

_LANGUAGE_IDS: dict[Language, str] = {
    Language.PYTHON: "python",
    Language.TYPESCRIPT: "typescript",
    Language.JAVASCRIPT: "javascript",
    Language.JAVA: "java",
    Language.HTML: "html",
    Language.CSS: "css",
    Language.VUE: "vue",
}


def _extension_to_language() -> dict[str, Language]:
    mapping: dict[str, Language] = {}
    for language, exts in _EXTENSIONS.items():
        for ext in exts:
            mapping[ext] = language
    return mapping


_EXT_TO_LANGUAGE = _extension_to_language()


# Languages for which a concrete server adapter is implemented.
_IMPLEMENTED_LANGUAGES: frozenset[Language] = frozenset({Language.PYTHON})


def detect_language(path: str | Path) -> Language | None:
    """Return the :class:`Language` for *path* based on its extension.

    Returns ``None`` when the extension is not recognised.
    """
    suffix = Path(path).suffix.lower()
    return _EXT_TO_LANGUAGE.get(suffix)


def language_for_extension(extension: str) -> Language | None:
    """Return the language registered for *extension*.

    *extension* may or may not include the leading dot.
    """
    ext = extension.lower()
    if not ext.startswith("."):
        ext = "." + ext
    return _EXT_TO_LANGUAGE.get(ext)


def language_for_path(path: str | Path) -> Language | None:
    """Return the :class:`Language` for *path* based on its extension.

    Alias of :func:`detect_language` with a name that reads better at call
    sites operating on file paths.
    """
    return detect_language(path)


def is_supported_source_file(path: str | Path) -> bool:
    """Return ``True`` when *path* maps to a language with a server adapter."""
    language = detect_language(path)
    if language is None:
        return False
    return language in _IMPLEMENTED_LANGUAGES


def supports_language(language: Language) -> bool:
    """Return ``True`` when a server adapter is available for *language*."""
    return language in _IMPLEMENTED_LANGUAGES


def server_for_language(language: Language) -> str:
    """Return the server identifier (adapter key) for *language*.

    Raises:
        ValueError: when no server adapter is implemented for *language*.
    """
    if language not in _IMPLEMENTED_LANGUAGES:
        raise ValueError(
            f"No server adapter implemented for language {language!r}"
        )
    return _LANGUAGE_SERVERS[language]


_LANGUAGE_SERVERS: dict[Language, str] = {
    Language.PYTHON: "pyright",
}
