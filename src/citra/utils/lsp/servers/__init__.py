"""Built-in language-server registry."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..language import Language
from .base import InstallCandidate, ServerDefinition
from .pyright import PYRIGHT
from .typescript import TYPESCRIPT_LANGUAGE_SERVER


def _jdtls_command(executable: str, root: Path, cache: Path) -> tuple[str, ...]:
    digest = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:20]
    data_dir = cache / "lsp" / "jdtls" / digest
    data_dir.mkdir(parents=True, exist_ok=True)
    return (executable, "-data", str(data_dir))


VUE = ServerDefinition(
    id="vue",
    executable="vue-language-server",
    languages=(Language.VUE,),
    arguments=("--stdio",),
    requires=("node", "typescript-language-server"),
    install_candidates=(
        InstallCandidate(
            "pacman",
            ("vue-language-server", "typescript-language-server", "typescript"),
            ("sudo", "pacman", "-S", "--needed", "vue-language-server", "typescript-language-server", "typescript"),
        ),
        InstallCandidate(
            "npm",
            ("@vue/language-server", "@vue/typescript-plugin", "typescript-language-server", "typescript@^6"),
            ("npm", "install", "-g", "@vue/language-server", "@vue/typescript-plugin", "typescript-language-server", "typescript@^6"),
        ),
    ),
    install_hint="Install Vue Language Server plus TypeScript LS and the Vue TypeScript plugin.",
)

JDTLS = ServerDefinition(
    id="jdtls",
    executable="jdtls",
    languages=(Language.JAVA,),
    requires=("java",),
    cold_diagnostics_timeout=90.0,
    command_factory=_jdtls_command,
    install_hint="Install Eclipse JDT Language Server and a compatible Java runtime.",
)

RUBY = ServerDefinition(
    id="ruby",
    executable="ruby-lsp",
    languages=(Language.RUBY,),
    requires=("ruby",),
    install_candidates=(
        InstallCandidate("pacman", ("ruby-lsp",), ("sudo", "pacman", "-S", "--needed", "ruby-lsp")),
        InstallCandidate("gem", ("ruby-lsp",), ("gem", "install", "ruby-lsp")),
    ),
    install_hint="Install ruby-lsp in the project's Ruby environment.",
)

JSON = ServerDefinition(
    id="json",
    executable="vscode-json-language-server",
    languages=(Language.JSON, Language.JSONC),
    arguments=("--stdio",),
    requires=("node",),
    install_candidates=(
        InstallCandidate(
            "pacman",
            ("vscode-json-languageserver",),
            ("sudo", "pacman", "-S", "--needed", "vscode-json-languageserver"),
        ),
        InstallCandidate(
            "npm",
            ("vscode-langservers-extracted",),
            ("npm", "install", "-g", "vscode-langservers-extracted"),
        ),
    ),
    install_hint="Install vscode-json-languageserver (the executable is vscode-json-language-server).",
)

CSS = ServerDefinition(
    id="css",
    executable="vscode-css-language-server",
    languages=(Language.CSS, Language.SCSS, Language.LESS),
    arguments=("--stdio",),
    requires=("node",),
    install_candidates=(
        InstallCandidate(
            "pacman",
            ("vscode-css-languageserver",),
            ("sudo", "pacman", "-S", "--needed", "vscode-css-languageserver"),
        ),
        InstallCandidate(
            "npm",
            ("vscode-langservers-extracted",),
            ("npm", "install", "-g", "vscode-langservers-extracted"),
        ),
    ),
    install_hint="Install vscode-css-languageserver.",
)

HTML = ServerDefinition(
    id="html",
    executable="vscode-html-language-server",
    languages=(Language.HTML,),
    arguments=("--stdio",),
    requires=("node",),
    install_candidates=(
        InstallCandidate(
            "pacman",
            ("vscode-html-languageserver",),
            ("sudo", "pacman", "-S", "--needed", "vscode-html-languageserver"),
        ),
        InstallCandidate(
            "npm",
            ("vscode-langservers-extracted",),
            ("npm", "install", "-g", "vscode-langservers-extracted"),
        ),
    ),
    install_hint="Install vscode-html-languageserver.",
)

YAML = ServerDefinition(
    id="yaml",
    executable="yaml-language-server",
    languages=(Language.YAML,),
    arguments=("--stdio",),
    requires=("node",),
    settings={"yaml": {"validate": True, "schemaStore": {"enable": False}}},
    install_candidates=(
        InstallCandidate("pacman", ("yaml-language-server",), ("sudo", "pacman", "-S", "--needed", "yaml-language-server")),
        InstallCandidate("npm", ("yaml-language-server",), ("npm", "install", "-g", "yaml-language-server")),
    ),
    install_hint="Install yaml-language-server.",
)

SQL = ServerDefinition(
    id="sql",
    executable="sqls",
    languages=(Language.SQL,),
    install_candidates=(
        InstallCandidate(
            "go",
            ("github.com/sqls-server/sqls@latest",),
            ("go", "install", "github.com/sqls-server/sqls@latest"),
        ),
    ),
    install_hint="Install sqls; database credentials are optional.",
)

BASH = ServerDefinition(
    id="bash",
    executable="bash-language-server",
    languages=(Language.BASH,),
    arguments=("start",),
    requires=("node",),
    install_candidates=(
        InstallCandidate("npm", ("bash-language-server",), ("npm", "install", "-g", "bash-language-server")),
    ),
    install_hint="Install bash-language-server.",
)

CLANGD = ServerDefinition(
    id="clangd",
    executable="clangd",
    languages=(Language.C, Language.CPP),
    install_hint="Install clangd with your compiler toolchain.",
)

GOPLS = ServerDefinition(
    id="gopls",
    executable="gopls",
    languages=(Language.GO,),
    install_candidates=(
        InstallCandidate("go", ("golang.org/x/tools/gopls@latest",), ("go", "install", "golang.org/x/tools/gopls@latest")),
    ),
    install_hint="Install gopls.",
)

RUST_ANALYZER = ServerDefinition(
    id="rust-analyzer",
    executable="rust-analyzer",
    languages=(Language.RUST,),
    install_hint="Install rust-analyzer with your Rust toolchain or system package manager.",
)

LUA = ServerDefinition(
    id="lua",
    executable="lua-language-server",
    languages=(Language.LUA,),
    install_hint="Install lua-language-server.",
)

TAPLO = ServerDefinition(
    id="taplo",
    executable="taplo",
    languages=(Language.TOML,),
    arguments=("lsp", "stdio"),
    install_candidates=(
        InstallCandidate(
            "cargo",
            ("taplo-cli",),
            ("cargo", "install", "taplo-cli", "--locked"),
        ),
    ),
    install_hint="Install Taplo.",
)

SERVERS: dict[str, ServerDefinition] = {
    definition.id: definition
    for definition in (
        PYRIGHT,
        TYPESCRIPT_LANGUAGE_SERVER,
        VUE,
        JDTLS,
        RUBY,
        JSON,
        CSS,
        HTML,
        YAML,
        SQL,
        BASH,
        CLANGD,
        GOPLS,
        RUST_ANALYZER,
        LUA,
        TAPLO,
    )
}

SERVER_ALIASES: dict[str, str] = {
    "typescript-language-server": "typescript",
    "vue-language-server": "vue",
    "ruby-lsp": "ruby",
    "vscode-json-language-server": "json",
    "vscode-css-language-server": "css",
    "vscode-html-language-server": "html",
    "yaml-language-server": "yaml",
    "bash-language-server": "bash",
    "lua-language-server": "lua",
}

__all__ = [
    "InstallCandidate",
    "PYRIGHT",
    "SERVERS",
    "SERVER_ALIASES",
    "ServerDefinition",
    "TYPESCRIPT_LANGUAGE_SERVER",
]
