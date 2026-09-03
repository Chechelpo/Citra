"""Language-specific runtime discovery extensions.

Each object extends the standard executable closure with installation roots
that a compiler, interpreter, or package manager loads after process startup.
Adding a language runtime requires only a new :class:`RuntimeDiscovery`
instance in ``DISCOVERIES``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from ._base import RuntimeDiscoveryResult, StandardDiscovery
from ._roots import is_broad_install_prefix

logger = logging.getLogger(__name__)


class _ExtendedCommandDiscovery(StandardDiscovery):
    """Discover a command set plus language-specific filesystem roots."""

    commands: tuple[str, ...] = ()

    @classmethod
    def extra_roots(cls) -> tuple[Path, ...]:
        """Return existing language roots required after process startup."""
        return ()

    @classmethod
    def discover(cls) -> RuntimeDiscoveryResult:
        """Merge executable closures with the language's installation roots."""
        result = super().discover()
        roots = {
            path.expanduser().absolute()
            for path in (*result.readonly_binds, *cls.extra_roots())
            if path.expanduser().exists()
        }
        logger.debug(
            "Language runtime discovery completed",
            extra={
                "origin": __name__,
                "discovery": cls.__name__,
                "commands": len(result.command_paths),
                "roots": len(roots),
            },
        )
        return RuntimeDiscoveryResult(
            readonly_binds=tuple(sorted(roots, key=str)),
            available_commands=result.available_commands,
            command_paths=result.command_paths,
        )


class PythonRuntimeDiscovery(_ExtendedCommandDiscovery):
    """Discover Python interpreters, package tools, linters, and type checkers."""

    commands = (
        "python",
        "python3",
        "uv",
        "pip",
        "pip3",
        "pytest",
        "ruff",
        "pyrefly",
        "pyright",
        "mypy",
        "black",
        "isort",
        "tox",
        "nox",
        "poetry",
        "pdm",
    )

    @classmethod
    def extra_roots(cls) -> tuple[Path, ...]:
        """Return bounded interpreter prefixes and Python library roots."""
        roots: list[Path] = []
        for prefix in (Path(sys.prefix), Path(sys.base_prefix)):
            if is_broad_install_prefix(prefix):
                logger.debug(
                    "Expanded broad Python prefix into library roots",
                    extra={"origin": __name__, "prefix": str(prefix)},
                )
                continue
            roots.append(prefix)

        paths = sysconfig.get_paths()
        for name in (
            "stdlib",
            "platstdlib",
            "purelib",
            "platlib",
            "include",
            "platinclude",
        ):
            value = paths.get(name)
            if value:
                roots.append(Path(value))
        return tuple(dict.fromkeys(roots))


class NodeRuntimeDiscovery(_ExtendedCommandDiscovery):
    """Discover JavaScript runtimes and package-installed CLI tools."""

    commands = (
        "node",
        "npm",
        "npx",
        "pnpm",
        "yarn",
        "bun",
        "deno",
        "tsc",
        "eslint",
        "prettier",
        "vitest",
        "jest",
        "mmdc",
    )

    @classmethod
    def extra_roots(cls) -> tuple[Path, ...]:
        """Return installation prefixes owning discovered Node entry points."""
        roots: list[Path] = []
        for command in cls.commands:
            raw = shutil.which(command)
            if raw is None:
                continue
            executable = Path(raw).absolute()
            for candidate in (executable, executable.resolve()):
                module_store = _node_module_store(candidate)
                if module_store is not None:
                    roots.append(module_store)
                if candidate.parent.name == "bin":
                    prefix = candidate.parent.parent
                    if (prefix / "bin" / "node").exists() or (
                        prefix / "lib" / "node_modules"
                    ).is_dir():
                        if is_broad_install_prefix(prefix):
                            global_modules = prefix / "lib" / "node_modules"
                            if global_modules.is_dir():
                                roots.append(global_modules)
                            logger.debug(
                                "Expanded broad Node prefix into module roots",
                                extra={
                                    "origin": __name__,
                                    "prefix": str(prefix),
                                },
                            )
                        else:
                            roots.append(prefix)
        return tuple(dict.fromkeys(roots))


def _node_module_store(path: Path) -> Path | None:
    """Return the bounded package store containing a Node command script."""
    parts = path.expanduser().absolute().parts
    for marker in ("node_modules", "nodejs"):
        if marker not in parts:
            continue
        index = parts.index(marker)
        store = Path(path.anchor, *parts[1 : index + 1])
        if store.is_dir():
            logger.debug(
                "Discovered Node module store",
                extra={"origin": __name__, "store": str(store)},
            )
            return store
    return None


class RustRuntimeDiscovery(_ExtendedCommandDiscovery):
    """Discover Cargo, rustup-managed toolchains, and Rust developer tools."""

    commands = ("cargo", "rustc", "rustfmt", "clippy-driver", "rustup")

    @classmethod
    def extra_roots(cls) -> tuple[Path, ...]:
        """Return Cargo and rustup homes when present."""
        home = Path.home()
        cargo_home = Path(os.environ.get("CARGO_HOME", home / ".cargo"))
        rustup_home = Path(os.environ.get("RUSTUP_HOME", home / ".rustup"))
        return cargo_home, rustup_home


class CRuntimeDiscovery(_ExtendedCommandDiscovery):
    """Discover C/C++ compilers, headers, linkers, and resource directories."""

    commands = (
        "clang",
        "clang++",
        "clangd",
        "gcc",
        "g++",
        "cc",
        "c++",
        "ld",
        "ar",
        "as",
        "make",
        "cmake",
        "ninja",
        "gdb",
        "lldb",
    )

    @classmethod
    def extra_roots(cls) -> tuple[Path, ...]:
        """Return conventional headers and compiler-reported resource roots."""
        roots: list[Path] = [
            Path("/usr/include"),
            Path("/usr/local/include"),
            Path("/usr/lib/gcc"),
            Path("/usr/lib/clang"),
            Path("/usr/local/lib/clang"),
        ]
        clang = shutil.which("clang")
        if clang is not None:
            try:
                completed = subprocess.run(
                    (clang, "-print-resource-dir"),
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                if completed.returncode == 0 and completed.stdout.strip():
                    roots.append(Path(completed.stdout.strip()))
            except (OSError, subprocess.TimeoutExpired):
                logger.warning(
                    "Could not inspect the Clang resource directory",
                    extra={"origin": __name__, "executable": clang},
                )
        gcc = shutil.which("gcc")
        if gcc is not None:
            try:
                completed = subprocess.run(
                    (gcc, "-print-multiarch"),
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                multiarch = completed.stdout.strip()
                if completed.returncode == 0 and multiarch:
                    roots.extend(
                        (Path("/usr/lib") / multiarch, Path("/lib") / multiarch)
                    )
            except (OSError, subprocess.TimeoutExpired):
                logger.warning(
                    "Could not inspect the GCC multiarch runtime",
                    extra={"origin": __name__, "executable": gcc},
                )
        return tuple(dict.fromkeys(roots))


class GoRuntimeDiscovery(_ExtendedCommandDiscovery):
    """Discover the Go SDK, compiler helpers, formatters, and linters."""

    commands = ("go", "gofmt", "golangci-lint", "staticcheck")

    @classmethod
    def extra_roots(cls) -> tuple[Path, ...]:
        """Return the Go SDK root reported by the active toolchain."""
        go = shutil.which("go")
        if go is None:
            return ()
        try:
            completed = subprocess.run(
                (go, "env", "GOROOT"),
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            logger.warning(
                "Could not inspect the Go SDK root",
                extra={"origin": __name__, "executable": go},
            )
            return ()
        value = completed.stdout.strip()
        return (Path(value),) if completed.returncode == 0 and value else ()


class JvmRuntimeDiscovery(_ExtendedCommandDiscovery):
    """Discover JVM commands and their complete JDK installation roots."""

    commands = (
        "java",
        "javac",
        "jar",
        "jshell",
        "mvn",
        "mvnw",
        "gradle",
        "gradlew",
        "kotlin",
        "kotlinc",
    )

    @classmethod
    def extra_roots(cls) -> tuple[Path, ...]:
        """Return JDK prefixes inferred from resolved executable paths."""
        roots: list[Path] = []
        for command in cls.commands:
            raw = shutil.which(command)
            if raw is None:
                continue
            resolved = Path(raw).resolve()
            if resolved.parent.name == "bin":
                roots.append(resolved.parent.parent)
        roots.extend((Path("/etc/java"), Path("/usr/share/java")))
        return tuple(dict.fromkeys(roots))


class DotNetRuntimeDiscovery(_ExtendedCommandDiscovery):
    """Discover .NET SDK commands and the shared framework installation."""

    commands = ("dotnet", "msbuild", "csc")

    @classmethod
    def extra_roots(cls) -> tuple[Path, ...]:
        """Return the prefix containing the resolved dotnet host."""
        dotnet = shutil.which("dotnet")
        if dotnet is None:
            return ()
        executable = Path(dotnet).resolve()
        return (executable.parent,)


class RubyRuntimeDiscovery(_ExtendedCommandDiscovery):
    """Discover Ruby commands together with interpreter libraries."""

    commands = ("ruby", "bundle", "bundler", "gem", "rake", "rspec")

    @classmethod
    def extra_roots(cls) -> tuple[Path, ...]:
        """Return conventional Ruby standard-library and gem roots."""
        gem_home = os.environ.get("GEM_HOME")
        roots = [Path("/usr/lib/ruby"), Path("/usr/local/lib/ruby")]
        if gem_home:
            roots.append(Path(gem_home))
        return tuple(dict.fromkeys(roots))


class GitRuntimeDiscovery(_ExtendedCommandDiscovery):
    """Discover Git plus its out-of-process transport helpers and templates."""

    commands = ("git", "ssh", "scp")

    @classmethod
    def extra_roots(cls) -> tuple[Path, ...]:
        """Return Git's executable and shared-data directories."""
        roots: list[Path] = [Path("/usr/share/git-core"), Path("/etc/gitconfig")]
        git = shutil.which("git")
        if git is not None:
            try:
                completed = subprocess.run(
                    (git, "--exec-path"),
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                value = completed.stdout.strip()
                if completed.returncode == 0 and value:
                    roots.append(Path(value))
            except (OSError, subprocess.TimeoutExpired):
                logger.warning(
                    "Could not inspect Git's helper directory",
                    extra={"origin": __name__, "executable": git},
                )
        return tuple(dict.fromkeys(roots))
