from __future__ import annotations

from dataclasses import dataclass

from .execution_context import ExecutionContext


@dataclass(frozen=True)
class CommandGroup:
    name: str
    commands: tuple[str, ...]


_COMMAND_GROUPS: tuple[CommandGroup, ...] = (
    CommandGroup(
        name="Shell / filesystem",
        commands=(
            "bash",
            "sh",
            "zsh",
            "fish",
            "find",
            "fd",
            "xargs",
            "tree",
            "stat",
            "file",
            "realpath",
            "readlink",
            "dirname",
            "basename",
            "ls",
            "cp",
            "mv",
            "mkdir",
            "touch",
        ),
    ),
    CommandGroup(
        name="Text / structured data",
        commands=(
            "tree",
            "rg",
            "grep",
            "sed",
            "awk",
            "cut",
            "sort",
            "uniq",
            "tr",
            "head",
            "tail",
            "wc",
            "jq",
            "yq",
            "xmllint",
        ),
    ),
    CommandGroup(
        name="Python",
        commands=(
            "python",
            "python3",
            "uv",
            "pip",
            "pip3",
            "poetry",
            "pdm",
            "pytest",
            "ruff",
            "black",
            "isort",
            "mypy",
            "pyright",
            "tox",
            "nox",
        ),
    ),
    CommandGroup(
        name="JavaScript / TypeScript",
        commands=(
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
        ),
    ),
    CommandGroup(
        name="Java / JVM",
        commands=(
            "java",
            "javac",
            "jar",
            "jshell",
            "mvn",
            "mvnw",
            "gradle",
            "gradlew",
            "kotlinc",
            "kotlin",
        ),
    ),
    CommandGroup(
        name="Rust",
        commands=(
            "rustc",
            "cargo",
            "rustfmt",
            "clippy-driver",
        ),
    ),
    CommandGroup(
        name="Go",
        commands=(
            "go",
            "gofmt",
            "golangci-lint",
            "staticcheck",
        ),
    ),
    CommandGroup(
        name="C / C++",
        commands=(
            "gcc",
            "g++",
            "clang",
            "clang++",
            "cc",
            "c++",
            "gdb",
            "lldb",
            "valgrind",
        ),
    ),
    CommandGroup(
        name="Build systems",
        commands=(
            "make",
            "cmake",
            "ninja",
            "meson",
            "bazel",
            "buck",
        ),
    ),
    CommandGroup(
        name="Local databases",
        commands=(
            "sqlite3",
        ),
    ),
    CommandGroup(
        name=".NET",
        commands=(
            "dotnet",
            "msbuild",
            "csc",
        ),
    ),
    CommandGroup(
        name="Ruby",
        commands=(
            "ruby",
            "gem",
            "bundle",
            "rake",
            "rspec",
        ),
    ),
    CommandGroup(
        name="PHP",
        commands=(
            "php",
            "composer",
            "phpunit",
        ),
    ),
    CommandGroup(
        name="Swift / Apple",
        commands=(
            "swift",
            "swiftc",
            "swiftformat",
            "swiftlint",
            "xcodebuild",
        ),
    ),
    CommandGroup(
        name="Compression / archives",
        commands=(
            "tar",
            "zip",
            "unzip",
            "gzip",
            "gunzip",
            "bzip2",
            "xz",
            "7z",
        ),
    ),
    CommandGroup(
        name="System inspection",
        commands=(
            "ps",
            "top",
            "htop",
            "free",
            "df",
            "du",
            "uname",
            "env",
            "printenv",
            "which",
            "whereis",
            "lsof",
            "strace",
        ),
    ),
)


def get_available_tools(
    context: ExecutionContext,
) -> str:
    """
    Return useful commands available inside the sandboxed Bash
    execution environment.

    Network-oriented, source-control, and host-control commands are
    intentionally omitted even if installed on the host.
    """
    sections: list[str] = [
        (
            "Bash runs without network access. Filesystem writes are "
            "restricted to the active workspace and temporary agent "
            "filesystem. Use the dedicated git tool for Git operations."
        )
    ]

    for group in _COMMAND_GROUPS:
        available = tuple(
            command
            for command in group.commands
            if context.has_command(command)
        )

        if not available:
            continue

        sections.append(
            f"- **{group.name}:** "
            + ", ".join(available)
        )

    if len(sections) == 1:
        sections.append(
            "No commonly useful sandboxed CLI utilities were detected."
        )

    return "\n".join(sections)