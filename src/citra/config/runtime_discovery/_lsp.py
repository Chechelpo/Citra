"""Discover installed language servers and their runtime resource roots.

Language servers are controller-discovered before the sandbox is created.  A
server is useful only when its executable and any package resources it loads
after startup are provisioned into the isolated runtime.  This module keeps
that policy in one registry object so adding another server requires only an
entry in :data:`LANGUAGE_SERVER_COMMANDS`.
"""

from __future__ import annotations

from pathlib import Path

from citra.logging import Logger

from ._base import RuntimeDiscoveryResult, StandardDiscovery
from ._roots import is_broad_install_prefix, is_runtime_prefix


_logger = Logger(__name__)


LANGUAGE_SERVER_COMMANDS: tuple[str, ...] = (
    "pyright-langserver",
    "typescript-language-server",
    "vue-language-server",
    "jdtls",
    "ruby-lsp",
    "vscode-json-language-server",
    "vscode-css-language-server",
    "vscode-html-language-server",
    "yaml-language-server",
    "sqls",
    "bash-language-server",
    "clangd",
    "gopls",
    "rust-analyzer",
    "lua-language-server",
    "taplo",
)

_NODE_LANGUAGE_SERVERS = frozenset(
    {
        "pyright-langserver",
        "typescript-language-server",
        "vue-language-server",
        "vscode-json-language-server",
        "vscode-css-language-server",
        "vscode-html-language-server",
        "yaml-language-server",
        "bash-language-server",
    }
)

_SERVER_RESOURCE_ROOTS: tuple[tuple[str, tuple[Path, ...]], ...] = (
    (
        "jdtls",
        (
            Path("/usr/share/java/jdtls"),
            Path("/usr/local/share/java/jdtls"),
            Path("/opt/jdtls"),
        ),
    ),
    (
        "lua-language-server",
        (
            Path("/usr/share/lua-language-server"),
            Path("/usr/lib/lua-language-server"),
            Path("/usr/local/share/lua-language-server"),
            Path("/usr/local/lib/lua-language-server"),
        ),
    ),
)


class LanguageServerRuntimeDiscovery(StandardDiscovery):
    """Discover every built-in LSP executable and its bounded package data."""

    commands = LANGUAGE_SERVER_COMMANDS

    @classmethod
    def discover(cls) -> RuntimeDiscoveryResult:
        """Return sandbox assets for installed language-server commands."""
        _logger.debug(
            "Starting language-server runtime discovery",
            registered=len(cls.commands),
        )
        result = super().discover()
        roots = set(result.readonly_binds)
        for command, executable in result.command_paths:
            command_roots = cls._resource_roots(command, executable)
            roots.update(command_roots)
            _logger.trace(
                "Discovered language-server command",
                command=command,
                executable=str(executable),
                resource_roots=tuple(str(path) for path in command_roots),
            )

        completed = RuntimeDiscoveryResult(
            readonly_binds=tuple(sorted(roots, key=str)),
            available_commands=result.available_commands,
            command_paths=result.command_paths,
        )
        _logger.info(
            "Language-server runtime discovery completed",
            installed=len(completed.command_paths),
            runtime_paths=len(completed.readonly_binds),
        )
        return completed

    @classmethod
    def _resource_roots(cls, command: str, executable: Path) -> tuple[Path, ...]:
        """Return bounded resource directories loaded by one server."""
        roots: list[Path] = []
        candidates = tuple(dict.fromkeys((executable.absolute(), executable.resolve())))
        for candidate in candidates:
            if command in _NODE_LANGUAGE_SERVERS:
                module_store = cls._node_module_store(candidate)
                if module_store is not None:
                    roots.append(module_store)

            if candidate.parent.name != "bin":
                continue
            prefix = candidate.parent.parent
            if is_runtime_prefix(prefix):
                roots.append(prefix)
                continue
            if command in _NODE_LANGUAGE_SERVERS and is_broad_install_prefix(prefix):
                module_store = prefix / "lib" / "node_modules"
                if module_store.is_dir():
                    roots.append(module_store)

        for resource_command, candidates in _SERVER_RESOURCE_ROOTS:
            if resource_command != command:
                continue
            roots.extend(path for path in candidates if path.exists())

        result = tuple(dict.fromkeys(roots))
        _logger.trace(
            "Resolved language-server resource roots",
            command=command,
            roots=tuple(str(path) for path in result),
        )
        return result

    @staticmethod
    def _node_module_store(path: Path) -> Path | None:
        """Return the package-store root containing a Node server script."""
        absolute = path.expanduser().absolute()
        parts = absolute.parts
        if "node_modules" not in parts:
            _logger.trace(
                "Language-server path is outside a Node package store",
                path=str(absolute),
            )
            return None
        index = parts.index("node_modules")
        store = Path(absolute.anchor, *parts[1 : index + 1])
        if not store.is_dir():
            _logger.warning(
                "Language-server Node package store is unavailable",
                path=str(absolute),
                store=str(store),
            )
            return None
        _logger.debug(
            "Resolved language-server Node package store",
            path=str(absolute),
            store=str(store),
        )
        return store


__all__ = [
    "LANGUAGE_SERVER_COMMANDS",
    "LanguageServerRuntimeDiscovery",
]
