from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from typing import TYPE_CHECKING

from .runtime import CopyPolicy, RuntimeAsset, ToolDefinition

if TYPE_CHECKING:
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


def declared_commands() -> tuple[str, ...]:
    """Return the unique command catalog used for runtime provisioning."""
    commands: dict[str, None] = {"git": None}
    for group in _COMMAND_GROUPS:
        for capability in group.capabilities:
            for command in capability.commands:
                commands.setdefault(command, None)
    return tuple(commands)


def default_tool_definitions() -> tuple[ToolDefinition, ...]:
    """Discover command entry points and express them as runtime assets.

    Native binaries are safe to copy as individual entry points because the
    declared OS compatibility layer supplies their dynamic loader/libraries.
    Script launchers retain an explicit read-only bind: copying a launcher
    without its package/module closure would advertise a broken tool.
    """
    definitions: list[ToolDefinition] = []
    for command in declared_commands():
        discovered = shutil.which(command)
        if discovered is None:
            continue
        visible_path = Path(discovered).absolute()
        source = visible_path.resolve()
        policy = (
            CopyPolicy.COPY_OR_BIND
            if _is_native_executable(source)
            and source.stat().st_size <= 2 * 1024 * 1024
            else CopyPolicy.BIND_ONLY
        )
        asset_id = f"command-{command}"
        assets = [
            RuntimeAsset(
                id=asset_id,
                source=source,
                destination=PurePosixPath("bin") / command,
                policy=policy,
                bind_target=visible_path,
            )
        ]
        # A non-system executable in a conventional bin/ directory commonly
        # loads modules, standard libraries, or sibling tools from its prefix.
        # Declare that closure explicitly instead of rediscovering it inside
        # the sandbox for each command.
        if source.parent.name == "bin" and not _is_under_system_root(source):
            prefix = source.parent.parent
            assets.append(
                RuntimeAsset(
                    id=f"command-closure-{command}",
                    source=prefix,
                    destination=PurePosixPath("compat") / f"tool-{command}",
                    policy=CopyPolicy.BIND_ONLY,
                    bind_target=prefix,
                    priority=150,
                )
            )
        health_check = None
        if command in {"bash", "git", "python", "python3"}:
            health_check = ("{executable}", "--version")
        definitions.append(
            ToolDefinition(
                id=f"command:{command}",
                commands=(command,),
                assets=tuple(assets),
                command_assets={command: asset_id},
                health_check=health_check,
            )
        )
    return tuple(definitions)


def default_runtime_assets(
    *,
    browser_path: str | Path | None = None,
) -> tuple[RuntimeAsset, ...]:
    """Declare narrow immutable host compatibility assets.

    These replace the historical blanket read-only bind of ``/``.  They are
    deliberately recorded in the runtime manifest even when their policy is
    bind-only, so the effective filesystem can be diagnosed precisely.
    """
    assets: list[RuntimeAsset] = []
    candidates: list[tuple[str, Path, bool]] = [
        ("os-usr", Path("/usr"), True),
        ("os-bin", Path("/bin"), True),
        ("os-lib", Path("/lib"), False),
        ("os-lib64", Path("/lib64"), False),
        ("etc-ssl", Path("/etc/ssl"), False),
        ("etc-ca-certificates", Path("/etc/ca-certificates"), False),
        ("etc-alternatives", Path("/etc/alternatives"), False),
        ("etc-passwd", Path("/etc/passwd"), False),
        ("etc-group", Path("/etc/group"), False),
        ("etc-nsswitch", Path("/etc/nsswitch.conf"), False),
        ("etc-ld-cache", Path("/etc/ld.so.cache"), False),
        ("etc-ld-conf", Path("/etc/ld.so.conf"), False),
        ("etc-ld-conf-d", Path("/etc/ld.so.conf.d"), False),
        ("etc-hosts", Path("/etc/hosts"), False),
        ("etc-localtime", Path("/etc/localtime"), False),
    ]

    install_root = os.environ.get("CITRA_INSTALL_ROOT")
    inferred_install = Path(__file__).resolve().parents[3]
    candidates.append(
        (
            "citra-install",
            Path(install_root).expanduser() if install_root else inferred_install,
            True,
        )
    )
    candidates.append(("citra-python-prefix", Path(sys.prefix), True))
    if Path(sys.base_prefix) != Path(sys.prefix):
        candidates.append(
            ("citra-python-base-prefix", Path(sys.base_prefix), True)
        )

    for asset_id, source, required in candidates:
        absolute = source.absolute()
        assets.append(
            RuntimeAsset(
                id=asset_id,
                source=absolute,
                destination=PurePosixPath("compat") / asset_id,
                policy=CopyPolicy.BIND_ONLY,
                required=required,
                bind_target=absolute,
                priority=200,
            )
        )

    if browser_path is not None:
        expanded = Path(browser_path).expanduser().absolute()
        assets.append(
            RuntimeAsset(
                id="playwright-browsers",
                source=expanded,
                destination=PurePosixPath("browsers/playwright"),
                policy=CopyPolicy.COPY_OR_BIND,
                required=False,
                bind_target=expanded,
                priority=10,
            )
        )
    return tuple(assets)


def _is_native_executable(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(4)
    except OSError:
        return False
    return header == b"\x7fELF" or header in {
        b"\xcf\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xca\xfe\xba\xbe",
    }


def _is_under_system_root(path: Path) -> bool:
    for root in (Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64")):
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


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
