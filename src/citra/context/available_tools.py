from __future__ import annotations

from dataclasses import dataclass

from .execution_context import ExecutionContext


@dataclass(frozen=True)
class CommandCapability:
    """One useful CLI capability with commands ordered by preference."""

    name: str
    commands: tuple[str, ...]


@dataclass(frozen=True)
class CommandGroup:
    name: str
    capabilities: tuple[CommandCapability, ...]


_COMMAND_GROUPS: tuple[CommandGroup, ...] = (
    CommandGroup(
        name="Filesystem",
        capabilities=(
            CommandCapability("shell", ("bash",)),
            CommandCapability("list", ("ls",)),
            CommandCapability("find files", ("fd", "find")),
            CommandCapability("inspect file", ("file",)),
            CommandCapability("metadata", ("stat",)),
            CommandCapability("resolve path", ("realpath", "readlink")),
            CommandCapability("copy", ("cp",)),
            CommandCapability("move", ("mv",)),
            CommandCapability("create directory", ("mkdir",)),
            CommandCapability("create/update file", ("touch",)),
            CommandCapability("batch arguments", ("xargs",)),
        ),
    ),
    CommandGroup(
        name="Text / data",
        capabilities=(
            CommandCapability("search text", ("rg", "grep")),
            CommandCapability("stream editing", ("sed",)),
            CommandCapability("text processing", ("awk",)),
            CommandCapability("columns", ("cut",)),
            CommandCapability("sort", ("sort",)),
            CommandCapability("deduplicate", ("uniq",)),
            CommandCapability("translate characters", ("tr",)),
            CommandCapability("head", ("head",)),
            CommandCapability("tail", ("tail",)),
            CommandCapability("count", ("wc",)),
            CommandCapability("JSON", ("jq",)),
            CommandCapability("YAML", ("yq",)),
            CommandCapability("XML", ("xmllint",)),
        ),
    ),
    CommandGroup(
        name="Python",
        capabilities=(
            CommandCapability("runtime", ("python", "python3")),
            CommandCapability("packages", ("uv", "pip", "pip3")),
            CommandCapability("project manager", ("poetry", "pdm")),
            CommandCapability("tests", ("pytest",)),
            CommandCapability("lint", ("ruff",)),
            CommandCapability("format", ("ruff", "black")),
            CommandCapability("imports", ("ruff", "isort")),
            CommandCapability("types", ("pyright", "mypy")),
            CommandCapability("test environments", ("tox", "nox")),
        ),
    ),
    CommandGroup(
        name="JavaScript / TypeScript",
        capabilities=(
            CommandCapability("runtime", ("node", "bun", "deno")),
            CommandCapability("packages", ("pnpm", "npm", "yarn", "bun")),
            CommandCapability("package execution", ("npx", "pnpm")),
            CommandCapability("TypeScript", ("tsc",)),
            CommandCapability("lint", ("eslint",)),
            CommandCapability("format", ("prettier",)),
            CommandCapability("tests", ("vitest", "jest")),
        ),
    ),
    CommandGroup(
        name="Rust",
        capabilities=(
            CommandCapability("toolchain", ("cargo",)),
            CommandCapability("compiler", ("rustc",)),
            CommandCapability("format", ("rustfmt",)),
            CommandCapability("lint", ("clippy-driver",)),
        ),
    ),
    CommandGroup(
        name="Go",
        capabilities=(
            CommandCapability("toolchain", ("go",)),
            CommandCapability("format", ("gofmt",)),
            CommandCapability("lint", ("golangci-lint", "staticcheck")),
        ),
    ),
    CommandGroup(
        name="C / C++",
        capabilities=(
            CommandCapability("C compiler", ("clang", "gcc", "cc")),
            CommandCapability("C++ compiler", ("clang++", "g++", "c++")),
            CommandCapability("debugger", ("gdb", "lldb")),
            CommandCapability("memory analysis", ("valgrind",)),
        ),
    ),
    CommandGroup(
        name="JVM",
        capabilities=(
            CommandCapability("Java runtime", ("java",)),
            CommandCapability("Java compiler", ("javac",)),
            CommandCapability("Java archive", ("jar",)),
            CommandCapability("Java REPL", ("jshell",)),
            CommandCapability("build", ("mvnw", "mvn", "gradlew", "gradle")),
            CommandCapability("Kotlin compiler", ("kotlinc",)),
            CommandCapability("Kotlin runtime", ("kotlin",)),
        ),
    ),
    CommandGroup(
        name=".NET",
        capabilities=(
            CommandCapability("toolchain", ("dotnet",)),
            CommandCapability("build", ("msbuild",)),
            CommandCapability("C# compiler", ("csc",)),
        ),
    ),
    CommandGroup(
        name="Ruby",
        capabilities=(
            CommandCapability("runtime", ("ruby",)),
            CommandCapability("packages", ("bundle", "gem")),
            CommandCapability("tasks", ("rake",)),
            CommandCapability("tests", ("rspec",)),
        ),
    ),
    CommandGroup(
        name="PHP",
        capabilities=(
            CommandCapability("runtime", ("php",)),
            CommandCapability("packages", ("composer",)),
            CommandCapability("tests", ("phpunit",)),
        ),
    ),
    CommandGroup(
        name="Swift",
        capabilities=(
            CommandCapability("toolchain", ("swift",)),
            CommandCapability("compiler", ("swiftc",)),
            CommandCapability("format", ("swiftformat",)),
            CommandCapability("lint", ("swiftlint",)),
            CommandCapability("Xcode build", ("xcodebuild",)),
        ),
    ),
    CommandGroup(
        name="Build",
        capabilities=(
            CommandCapability("Make", ("make",)),
            CommandCapability("CMake", ("cmake",)),
            CommandCapability("Ninja", ("ninja",)),
            CommandCapability("Meson", ("meson",)),
            CommandCapability("large-project build", ("bazel", "buck")),
        ),
    ),
    CommandGroup(
        name="Archives",
        capabilities=(
            CommandCapability("tar archives", ("tar",)),
            CommandCapability("zip archives", ("zip",)),
            CommandCapability("unzip", ("unzip",)),
            CommandCapability("compression", ("gzip", "xz", "bzip2", "7z")),
        ),
    ),
    CommandGroup(
        name="Database",
        capabilities=(
            CommandCapability("SQLite", ("sqlite3",)),
        ),
    ),
    CommandGroup(
        name="System inspection",
        capabilities=(
            CommandCapability("processes", ("ps",)),
            CommandCapability("process monitor", ("htop", "top")),
            CommandCapability("memory", ("free",)),
            CommandCapability("filesystem space", ("df",)),
            CommandCapability("directory size", ("du",)),
            CommandCapability("system", ("uname",)),
            CommandCapability("environment", ("printenv", "env")),
            CommandCapability("locate executable", ("which", "whereis")),
            CommandCapability("open files", ("lsof",)),
            CommandCapability("syscall tracing", ("strace",)),
        ),
    ),
)


def _available_command(
    context: ExecutionContext,
    capability: CommandCapability,
) -> str | None:
    for command in capability.commands:
        if context.has_command(command):
            return command
    return None


def get_available_tools(
    context: ExecutionContext,
) -> str:
    """
    Return useful command capabilities available inside the sandbox.

    Equivalent utilities are collapsed to the preferred available command so
    the model is not distracted by redundant choices. Network-oriented,
    source-control, language-server, and host-control commands are omitted.
    """
    sections: list[str] = [
        (
            "Bash runs without network access. Filesystem writes are "
            "restricted to the active workspace and lifecycle agent "
            "filesystem. Use dedicated Citra tools for Git and LSP operations."
        )
    ]

    for group in _COMMAND_GROUPS:
        available: list[str] = []
        seen_commands: set[str] = set()

        for capability in group.capabilities:
            command = _available_command(context, capability)
            if command is None or command in seen_commands:
                continue
            seen_commands.add(command)
            available.append(f"{capability.name}: `{command}`")

        if available:
            sections.append(
                f"- **{group.name}:** " + "; ".join(available)
            )

    if len(sections) == 1:
        sections.append(
            "No commonly useful sandboxed CLI utilities were detected."
        )

    return "\n".join(sections)
