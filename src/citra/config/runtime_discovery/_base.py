from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import os
import re
import shutil
import subprocess


@dataclass(frozen=True)
class CommandCapability:
    """One useful CLI capability with commands ordered by preference."""

    name: str
    commands: tuple[str, ...]


@dataclass(frozen=True)
class CommandGroup:
    """Related model-facing command capabilities."""

    name: str
    capabilities: tuple[CommandCapability, ...]


COMMAND_GROUPS: tuple[CommandGroup, ...] = (
    CommandGroup(
        name="Filesystem",
        capabilities=(
            CommandCapability("shell", ("bash",)),
            CommandCapability("list", ("ls",)),
            CommandCapability("find files", ("fd", "find")),
            CommandCapability("inspect file", ("file",)),
            CommandCapability("metadata", ("stat",)),
            CommandCapability(
                "resolve path",
                ("realpath", "readlink"),
            ),
            CommandCapability("copy", ("cp",)),
            CommandCapability("move", ("mv",)),
            CommandCapability(
                "create directory",
                ("mkdir",),
            ),
            CommandCapability(
                "create/update file",
                ("touch",),
            ),
            CommandCapability(
                "batch arguments",
                ("xargs",),
            ),
        ),
    ),
    CommandGroup(
        name="Text / data",
        capabilities=(
            CommandCapability(
                "search text",
                ("rg", "grep"),
            ),
            CommandCapability(
                "stream editing",
                ("sed",),
            ),
            CommandCapability(
                "text processing",
                ("awk",),
            ),
            CommandCapability("columns", ("cut",)),
            CommandCapability("sort", ("sort",)),
            CommandCapability(
                "deduplicate",
                ("uniq",),
            ),
            CommandCapability(
                "translate characters",
                ("tr",),
            ),
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
            CommandCapability(
                "runtime",
                ("python", "python3"),
            ),
            CommandCapability(
                "packages",
                ("uv", "pip", "pip3"),
            ),
            CommandCapability(
                "project manager",
                ("poetry", "pdm"),
            ),
            CommandCapability(
                "tests",
                ("pytest",),
            ),
            CommandCapability(
                "lint",
                ("ruff",),
            ),
            CommandCapability(
                "format",
                ("ruff", "black"),
            ),
            CommandCapability(
                "imports",
                ("ruff", "isort"),
            ),
            CommandCapability(
                "types",
                ("pyright", "mypy"),
            ),
            CommandCapability(
                "test environments",
                ("tox", "nox"),
            ),
        ),
    ),
    CommandGroup(
        name="JavaScript / TypeScript",
        capabilities=(
            CommandCapability(
                "runtime",
                ("node", "bun", "deno"),
            ),
            CommandCapability(
                "packages",
                ("pnpm", "npm", "yarn", "bun"),
            ),
            CommandCapability(
                "package execution",
                ("npx", "pnpm"),
            ),
            CommandCapability(
                "TypeScript",
                ("tsc",),
            ),
            CommandCapability(
                "lint",
                ("eslint",),
            ),
            CommandCapability(
                "format",
                ("prettier",),
            ),
            CommandCapability(
                "tests",
                ("vitest", "jest"),
            ),
        ),
    ),
    CommandGroup(
        name="Rust",
        capabilities=(
            CommandCapability(
                "toolchain",
                ("cargo",),
            ),
            CommandCapability(
                "compiler",
                ("rustc",),
            ),
            CommandCapability(
                "format",
                ("rustfmt",),
            ),
            CommandCapability(
                "lint",
                ("clippy-driver",),
            ),
        ),
    ),
    CommandGroup(
        name="Go",
        capabilities=(
            CommandCapability(
                "toolchain",
                ("go",),
            ),
            CommandCapability(
                "format",
                ("gofmt",),
            ),
            CommandCapability(
                "lint",
                (
                    "golangci-lint",
                    "staticcheck",
                ),
            ),
        ),
    ),
    CommandGroup(
        name="C / C++",
        capabilities=(
            CommandCapability(
                "C compiler",
                ("clang", "gcc", "cc"),
            ),
            CommandCapability(
                "C++ compiler",
                ("clang++", "g++", "c++"),
            ),
            CommandCapability(
                "debugger",
                ("gdb", "lldb"),
            ),
            CommandCapability(
                "memory analysis",
                ("valgrind",),
            ),
        ),
    ),
    CommandGroup(
        name="JVM",
        capabilities=(
            CommandCapability(
                "Java runtime",
                ("java",),
            ),
            CommandCapability(
                "Java compiler",
                ("javac",),
            ),
            CommandCapability(
                "Java archive",
                ("jar",),
            ),
            CommandCapability(
                "Java REPL",
                ("jshell",),
            ),
            CommandCapability(
                "build",
                (
                    "mvnw",
                    "mvn",
                    "gradlew",
                    "gradle",
                ),
            ),
            CommandCapability(
                "Kotlin compiler",
                ("kotlinc",),
            ),
            CommandCapability(
                "Kotlin runtime",
                ("kotlin",),
            ),
        ),
    ),
    CommandGroup(
        name=".NET",
        capabilities=(
            CommandCapability(
                "toolchain",
                ("dotnet",),
            ),
            CommandCapability(
                "build",
                ("msbuild",),
            ),
            CommandCapability(
                "C# compiler",
                ("csc",),
            ),
        ),
    ),
    CommandGroup(
        name="Ruby",
        capabilities=(
            CommandCapability(
                "runtime",
                ("ruby",),
            ),
            CommandCapability(
                "packages",
                ("bundle", "gem"),
            ),
            CommandCapability(
                "tasks",
                ("rake",),
            ),
            CommandCapability(
                "tests",
                ("rspec",),
            ),
        ),
    ),
    CommandGroup(
        name="PHP",
        capabilities=(
            CommandCapability(
                "runtime",
                ("php",),
            ),
            CommandCapability(
                "packages",
                ("composer",),
            ),
            CommandCapability(
                "tests",
                ("phpunit",),
            ),
        ),
    ),
    CommandGroup(
        name="Swift",
        capabilities=(
            CommandCapability(
                "toolchain",
                ("swift",),
            ),
            CommandCapability(
                "compiler",
                ("swiftc",),
            ),
            CommandCapability(
                "format",
                ("swiftformat",),
            ),
            CommandCapability(
                "lint",
                ("swiftlint",),
            ),
            CommandCapability(
                "Xcode build",
                ("xcodebuild",),
            ),
        ),
    ),
    CommandGroup(
        name="Build",
        capabilities=(
            CommandCapability("Make", ("make",)),
            CommandCapability("CMake", ("cmake",)),
            CommandCapability("Ninja", ("ninja",)),
            CommandCapability("Meson", ("meson",)),
            CommandCapability(
                "large-project build",
                ("bazel", "buck"),
            ),
        ),
    ),
    CommandGroup(
        name="Archives",
        capabilities=(
            CommandCapability(
                "tar archives",
                ("tar",),
            ),
            CommandCapability(
                "zip archives",
                ("zip",),
            ),
            CommandCapability(
                "unzip",
                ("unzip",),
            ),
            CommandCapability(
                "compression",
                (
                    "gzip",
                    "xz",
                    "bzip2",
                    "7z",
                ),
            ),
        ),
    ),
    CommandGroup(
        name="Database",
        capabilities=(
            CommandCapability(
                "SQLite",
                ("sqlite3",),
            ),
        ),
    ),
    CommandGroup(
        name="System inspection",
        capabilities=(
            CommandCapability(
                "processes",
                ("ps",),
            ),
            CommandCapability(
                "process monitor",
                ("htop", "top"),
            ),
            CommandCapability(
                "memory",
                ("free",),
            ),
            CommandCapability(
                "filesystem space",
                ("df",),
            ),
            CommandCapability(
                "directory size",
                ("du",),
            ),
            CommandCapability(
                "system",
                ("uname",),
            ),
            CommandCapability(
                "environment",
                ("printenv", "env"),
            ),
            CommandCapability(
                "locate executable",
                ("which", "whereis"),
            ),
            CommandCapability(
                "open files",
                ("lsof",),
            ),
            CommandCapability(
                "syscall tracing",
                ("strace",),
            ),
        ),
    ),
)


def declared_commands() -> tuple[str, ...]:
    """Return all commands considered for sandbox exposure."""

    result: dict[str, None] = {}

    for group in COMMAND_GROUPS:
        for capability in group.capabilities:
            for command in capability.commands:
                result.setdefault(
                    command,
                    None,
                )

    return tuple(result)


@dataclass(frozen=True)
class RuntimeDiscoveryResult:
    """
    Host runtime requirements discovered before sandbox creation.

    ``readonly_binds`` is the filesystem closure required by the discovered
    commands.

    ``available_commands`` contains only commands whose entry points were
    found and included in that closure.
    """

    readonly_binds: tuple[Path, ...] = ()
    available_commands: tuple[str, ...] = ()

    def has_command(
        self,
        command: str,
    ) -> bool:
        return command in self.available_commands

    def preferred_command(
        self,
        capability: CommandCapability,
    ) -> str | None:
        for command in capability.commands:
            if command in self.available_commands:
                return command

        return None

    def available_tools_section(self) -> str:
        """Render model-facing command capabilities."""

        sections: list[str] = []

        for group in COMMAND_GROUPS:
            available: list[str] = []
            seen: set[str] = set()

            for capability in group.capabilities:
                command = self.preferred_command(
                    capability
                )

                if (
                    command is None
                    or command in seen
                ):
                    continue

                seen.add(command)

                available.append(
                    f"{capability.name}: `{command}`"
                )

            if available:
                sections.append(
                    f"- **{group.name}:** "
                    + "; ".join(available)
                )

        if not sections:
            return (
                "No commonly useful sandboxed CLI utilities "
                "were detected."
            )

        return "\n".join(sections)


class RuntimeDiscovery(ABC):
    """Discover one portion of the sandbox's host runtime closure."""

    @classmethod
    @abstractmethod
    def discover(
        cls,
    ) -> RuntimeDiscoveryResult:
        """Return mounts and commands supplied by this discovery."""


class StandardDiscovery(RuntimeDiscovery):
    """
    Discover standard CLI commands and their runtime requirements.

    Each successfully discovered command contributes:
    - its visible executable path;
    - its symlink chain;
    - the resolved executable;
    - ELF shared-library dependencies;
    - a launcher/runtime prefix for non-native script commands where needed.
    """

    commands: tuple[str, ...] = declared_commands()

    @classmethod
    def discover(
        cls,
    ) -> RuntimeDiscoveryResult:
        readonly_binds: set[Path] = set()
        available_commands: list[str] = []

        for command in cls.commands:
            executable = cls._resolve_command(
                command
            )

            if executable is None:
                continue

            command_binds = cls._discover_command_binds(
                executable
            )

            if not command_binds:
                continue

            readonly_binds.update(
                command_binds
            )

            available_commands.append(
                command
            )

        return RuntimeDiscoveryResult(
            readonly_binds=tuple(
                sorted(
                    readonly_binds,
                    key=str,
                )
            ),
            available_commands=tuple(
                available_commands
            ),
        )

    @classmethod
    def _discover_command_binds(
        cls,
        executable: Path,
    ) -> set[Path]:
        result: set[Path] = set()

        symlink_chain = cls._resolve_symlink_chain(
            executable
        )

        result.update(
            symlink_chain
        )

        resolved = executable.resolve()

        result.add(
            resolved
        )

        if cls._is_native_executable(
            resolved
        ):
            result.update(
                cls._discover_shared_dependencies(
                    resolved
                )
            )
        else:
            result.update(
                cls._discover_script_runtime(
                    resolved
                )
            )

        return result

    @staticmethod
    def _resolve_command(
        command: str,
    ) -> Path | None:
        """
        Resolve a command through the controller PATH.

        Do not resolve symlinks here: the visible path itself must remain part
        of the mount closure.
        """

        location = shutil.which(
            command
        )

        if location is None:
            return None

        return Path(
            location
        ).absolute()

    @staticmethod
    def _resolve_symlink_chain(
        path: Path,
    ) -> set[Path]:
        """Return every path involved in the executable's symlink chain."""

        result: set[Path] = set()

        current = path.absolute()
        seen: set[Path] = set()

        while current not in seen:
            seen.add(
                current
            )

            result.add(
                current
            )

            if not current.is_symlink():
                break

            target = current.readlink()

            if target.is_absolute():
                current = target
            else:
                current = (
                    current.parent / target
                ).absolute()

        return result

    @classmethod
    def _discover_script_runtime(
        cls,
        executable: Path,
    ) -> set[Path]:
        """
        Discover the interpreter and conventional package closure for a script.

        A copied/bound script launcher on its own is usually insufficient:
        Python, Node, Ruby, etc. launchers commonly depend on siblings beneath
        the same installation prefix.
        """

        result: set[Path] = set()

        interpreter = cls._read_shebang_interpreter(
            executable
        )

        if interpreter is not None:
            result.update(
                cls._resolve_symlink_chain(
                    interpreter
                )
            )

            resolved = interpreter.resolve()

            result.add(
                resolved
            )

            result.update(
                cls._discover_shared_dependencies(
                    resolved
                )
            )

        if executable.parent.name == "bin":
            prefix = executable.parent.parent

            if not cls._is_system_prefix(
                prefix
            ):
                result.add(
                    prefix
                )

        return result

    @staticmethod
    def _read_shebang_interpreter(
        executable: Path,
    ) -> Path | None:
        try:
            with executable.open(
                "rb"
            ) as file:
                first_line = file.readline(
                    4096
                )
        except OSError:
            return None

        if not first_line.startswith(
            b"#!"
        ):
            return None

        try:
            shebang = first_line[
                2:
            ].decode(
                "utf-8"
            ).strip()
        except UnicodeDecodeError:
            return None

        if not shebang:
            return None

        parts = shebang.split()

        if not parts:
            return None

        interpreter = parts[0]

        # Handle "#!/usr/bin/env python".
        if (
            Path(interpreter).name == "env"
            and len(parts) >= 2
        ):
            resolved = shutil.which(
                parts[1]
            )

            if resolved is None:
                return None

            return Path(
                resolved
            ).absolute()

        path = Path(
            interpreter
        )

        if not path.is_absolute():
            return None

        if not path.exists():
            return None

        return path

    @staticmethod
    def _is_native_executable(
        path: Path,
    ) -> bool:
        try:
            with path.open(
                "rb"
            ) as file:
                header = file.read(
                    4
                )
        except OSError:
            return False

        return header == b"\x7fELF" or header in {
            b"\xcf\xfa\xed\xfe",
            b"\xfe\xed\xfa\xcf",
            b"\xca\xfe\xba\xbe",
        }

    @staticmethod
    def _is_system_prefix(
        path: Path,
    ) -> bool:
        for root in (
            Path("/usr"),
            Path("/bin"),
            Path("/lib"),
            Path("/lib64"),
        ):
            try:
                path.resolve().relative_to(
                    root
                )
                return True
            except ValueError:
                continue

        return False

    @staticmethod
    def _discover_shared_dependencies(
        executable: Path,
    ) -> set[Path]:
        """
        Discover ELF dependencies using ldd.

        Both ``foo => /path/foo`` and direct loader lines such as
        ``/lib64/ld-linux-x86-64.so.2 (...)`` are handled.
        """

        result: set[Path] = set()

        try:
            process = subprocess.run(
                (
                    "ldd",
                    str(executable),
                ),
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (
            FileNotFoundError,
            OSError,
            subprocess.TimeoutExpired,
        ):
            return result

        if process.returncode != 0:
            return result

        for line in process.stdout.splitlines():
            path = _ldd_path(
                line
            )

            if (
                path is not None
                and path.exists()
            ):
                result.add(
                    path.resolve()
                )

        return result


_LDD_DIRECT_PATH = re.compile(
    r"^\s*(/[^\s]+)"
)


def _ldd_path(
    line: str,
) -> Path | None:
    if "=>" in line:
        _, value = line.split(
            "=>",
            1,
        )

        candidate = value.strip().split(
            maxsplit=1
        )[0]

        if candidate == "not":
            return None

        if candidate.startswith("/"):
            return Path(candidate)

        return None

    match = _LDD_DIRECT_PATH.match(
        line
    )

    if match is None:
        return None

    return Path(
        match.group(1)
    )