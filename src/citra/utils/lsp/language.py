"""Language identification and server routing for Citra's LSP subsystem."""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class Language(str, Enum):
    """Represent Language."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    VUE = "vue"
    JAVA = "java"
    RUBY = "ruby"
    JSON = "json"
    JSONC = "jsonc"
    CSS = "css"
    SCSS = "scss"
    LESS = "less"
    SQL = "sql"
    HTML = "html"
    YAML = "yaml"
    BASH = "bash"
    C = "c"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    LUA = "lua"
    TOML = "toml"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        """Handle file extensions."""
        return _EXTENSIONS[self]

    @property
    def language_id(self) -> str:
        """Handle language id."""
        return _LANGUAGE_IDS[self]


_EXTENSIONS: dict[Language, tuple[str, ...]] = {
    Language.PYTHON: (".py", ".pyi"),
    Language.JAVASCRIPT: (".js", ".jsx", ".mjs", ".cjs"),
    Language.TYPESCRIPT: (".ts", ".tsx", ".mts", ".cts"),
    Language.VUE: (".vue",),
    Language.JAVA: (".java",),
    Language.RUBY: (".rb", ".rake"),
    Language.JSON: (".json",),
    Language.JSONC: (".jsonc",),
    Language.CSS: (".css",),
    Language.SCSS: (".scss",),
    Language.LESS: (".less",),
    Language.SQL: (".sql",),
    Language.HTML: (".html", ".htm"),
    Language.YAML: (".yaml", ".yml"),
    Language.BASH: (".sh", ".bash"),
    Language.C: (".c", ".h"),
    Language.CPP: (".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"),
    Language.GO: (".go",),
    Language.RUST: (".rs",),
    Language.LUA: (".lua",),
    Language.TOML: (".toml",),
}

_LANGUAGE_IDS: dict[Language, str] = {
    Language.PYTHON: "python",
    Language.JAVASCRIPT: "javascript",
    Language.TYPESCRIPT: "typescript",
    Language.VUE: "vue",
    Language.JAVA: "java",
    Language.RUBY: "ruby",
    Language.JSON: "json",
    Language.JSONC: "jsonc",
    Language.CSS: "css",
    Language.SCSS: "scss",
    Language.LESS: "less",
    Language.SQL: "sql",
    Language.HTML: "html",
    Language.YAML: "yaml",
    Language.BASH: "shellscript",
    Language.C: "c",
    Language.CPP: "cpp",
    Language.GO: "go",
    Language.RUST: "rust",
    Language.LUA: "lua",
    Language.TOML: "toml",
}

_LANGUAGE_SERVERS: dict[Language, str] = {
    Language.PYTHON: "pyright",
    # Preserve the historical public routing value for JS/TS.  The manager
    # normalizes this executable-name alias back to the declarative server id.
    Language.JAVASCRIPT: "typescript-language-server",
    Language.TYPESCRIPT: "typescript-language-server",
    Language.VUE: "vue",
    Language.JAVA: "jdtls",
    Language.RUBY: "ruby",
    Language.JSON: "json",
    Language.JSONC: "json",
    Language.CSS: "css",
    Language.SCSS: "css",
    Language.LESS: "css",
    Language.SQL: "sql",
    Language.HTML: "html",
    Language.YAML: "yaml",
    Language.BASH: "bash",
    Language.C: "clangd",
    Language.CPP: "clangd",
    Language.GO: "gopls",
    Language.RUST: "rust-analyzer",
    Language.LUA: "lua",
    Language.TOML: "taplo",
}

IMPLEMENTED_LANGUAGES: frozenset[Language] = frozenset(_LANGUAGE_SERVERS)


def _extension_to_language() -> dict[str, Language]:
    """Handle extension to language."""
    mapping: dict[str, Language] = {}
    for language, exts in _EXTENSIONS.items():
        for ext in exts:
            mapping[ext] = language
    return mapping


_EXT_TO_LANGUAGE = _extension_to_language()


def detect_language(path: str | Path) -> Language | None:
    """Return the language inferred from a path's extension."""
    suffix = Path(path).suffix.casefold()
    return _EXT_TO_LANGUAGE.get(suffix)


def language_for_extension(extension: str) -> Language | None:
    """Handle language for extension."""
    ext = extension.casefold()
    if not ext.startswith("."):
        ext = "." + ext
    return _EXT_TO_LANGUAGE.get(ext)


def language_for_path(path: str | Path) -> Language | None:
    """Handle language for path."""
    return detect_language(path)


def language_id_for_path(path: str | Path, language: Language) -> str:
    """Handle language id for path."""
    suffix = Path(path).suffix.casefold()
    if suffix == ".tsx":
        return "typescriptreact"
    if suffix == ".jsx":
        return "javascriptreact"
    return language.language_id


def is_supported_source_file(path: str | Path) -> bool:
    """Return whether is supported source file."""
    return detect_language(path) in IMPLEMENTED_LANGUAGES


def supports_language(language: Language) -> bool:
    """Handle supports language."""
    return language in IMPLEMENTED_LANGUAGES


def server_for_language(language: Language) -> str:
    """Handle server for language."""
    try:
        return _LANGUAGE_SERVERS[language]
    except KeyError as error:
        raise ValueError(f"No server adapter implemented for language {language!r}") from error
