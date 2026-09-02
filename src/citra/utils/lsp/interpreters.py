"""Project-local interpreter resolvers for language servers.

Language servers such as Pyright, gopls, ruby-lsp, or the TypeScript language
server need to know which interpreter / toolchain to use to resolve installed
third-party dependencies. The Citra runtime already provisions a lifecycle
``env/`` virtual environment, but many projects ship their own per-project
interpreter (``.venv``/``venv`` for Python, ``vendor/bundle`` for Ruby,
``node_modules/.bin`` for Node, ``$GOPATH/bin`` for Go, ...). Surfacing those
to the language server keeps the type checker honest without forcing every
project to relink its dependencies into the runtime venv.

The public surface of this module is intentionally narrow and language
agnostic:

* :class:`ResolvedInterpreter` is a value object describing what was found.
* :class:`InterpreterResolver` is the :class:`typing.Protocol` used by
  :class:`~citra.utils.lsp.servers.base.ServerDefinition` to plug a resolver
  in. The protocol only needs a project root and the active
  :class:`~citra.context.session_context.WorkspaceContext`; it does not depend
  on any LSP internals.
* :func:`find_python_venv` and :func:`resolve_python` implement the Python
  case in full.
* :func:`resolve_ruby`, :func:`resolve_node` and :func:`resolve_go` are
  documented stubs that return ``interpreter=None`` today. They exist so
  that adding a new language later is a one-line ``interpreter_resolver=``
  assignment, not a change to the manager.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from citra.context.session_context import WorkspaceContext


logger = logging.getLogger(__name__)


# Order in which project-local venvs are looked up. The convention matches
# `.vscode/settings.json` which advertises `.venv` as the project interpreter.
_PYTHON_VENV_CANDIDATES: tuple[str, ...] = (".venv", "venv", ".env", "env")
# Last-resort fallback ``sysconfig.get_python_version()`` returns (e.g.) "3.12".
# We only use the prefix ``python3`` when probing; the interpreter's actual
# version is resolved at runtime from sysconfig.
_PYTHON_VERSION_FALLBACK = "3.12"


@dataclass(frozen=True)
class ResolvedInterpreter:
    """The output of an :class:`InterpreterResolver`.

    All fields default to a "nothing found" value so that a partial match is
    still representable; the manager merges every populated field into the
    :class:`~citra.utils.lsp.config.ServerConfig` and the spawned process
    environment.
    """

    language: str
    interpreter: str | None = None
    #: Directories to prepend to ``PATH`` for the spawned language server,
    #: in the order they should appear (leftmost wins).
    path_prepend: tuple[str, ...] = ()
    #: Extra environment variables (e.g. ``VIRTUAL_ENV``) to set for the
    #: spawned process, merged on top of the workspace environment.
    environment: Mapping[str, str] = field(default_factory=dict)
    #: LSP ``workspace/didChangeConfiguration`` payload. Nested mappings are
    #: merged section by section so unrelated settings are preserved.
    settings: Mapping[str, Any] = field(default_factory=dict)
    #: LSP ``initialize`` ``initializationOptions`` payload. Shallow-merged
    #: on top of the server's defaults.
    initialization_options: Mapping[str, Any] = field(default_factory=dict)


class InterpreterResolver(Protocol):
    """Callable contract for project-interpreter resolution.

    A resolver is invoked once per LSP server start with the active project
    root and the surrounding :class:`WorkspaceContext`. The workspace is
    passed in (rather than read from a module global) so the resolver can be
    exercised in unit tests with a stub and so future resolvers can take
    runtime decisions based on the workspace layout.
    """

    def __call__(  # noqa: D102 - documented on the enclosing Protocol.
        self,
        project_root: Path,
        *,
        workspace: WorkspaceContext,
    ) -> ResolvedInterpreter: ...


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _python_executable_path(venv: Path) -> Path | None:
    """Return the interpreter path inside ``venv`` if it is usable.

    A venv is considered usable if it has ``pyvenv.cfg`` and either
    ``bin/python3`` (POSIX) or ``Scripts/python.exe`` (Windows). On POSIX we
    additionally accept ``bin/python`` as a fallback for handcrafted venvs
    that only ship the unversioned name.
    """
    try:
        if not venv.is_dir():
            return None
    except OSError:
        return None
    try:
        if not (venv / "pyvenv.cfg").is_file():
            return None
    except OSError:
        return None
    if _is_windows():
        candidate = venv / "Scripts" / "python.exe"
        return candidate if candidate.is_file() else None
    for name in ("python3", "python"):
        candidate = venv / "bin" / name
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def find_python_venv(project_root: Path) -> Path | None:
    """Return the first usable project-local venv, or ``None``.

    Candidates are checked in the order ``.venv``, ``venv``, ``.env``,
    ``env`` at ``project_root``. A candidate is accepted iff
    :func:`_python_executable_path` considers it usable.
    """
    try:
        if not project_root.is_dir():
            return None
    except OSError:
        return None
    for candidate_name in _PYTHON_VENV_CANDIDATES:
        candidate = project_root / candidate_name
        resolved = _python_executable_path(candidate)
        if resolved is not None:
            return candidate
    return None


# Matches the first dotted version of the form ``X.Y`` in the string. The
# interpreter's ``sysconfig.get_paths()`` always returns a ``stdlib`` path
# containing the actual ``pythonX.Y`` directory.
_PYTHON_VERSION_RE = re.compile(r"python(\d+\.\d+)")


def _python_site_packages(venv: Path, interpreter: Path) -> Path:
    """Return the venv's ``site-packages`` directory.

    Tries (in order):

    1. Invoking the interpreter briefly to read ``sysconfig.get_paths()``.
    2. Globbing ``<venv>/lib/pythonX.Y/site-packages`` for any ``X.Y``
       directory (handles CPython, PyPy, and stackless).
    3. Falling back to ``<venv>/lib/python3.12/site-packages`` to match the
       ``.vscode/settings.json`` ``python.analysis.extraPaths`` convention.
    """
    if not _is_windows():
        try:
            completed = subprocess.run(
                [
                    str(interpreter),
                    "-c",
                    "import sysconfig; print(sysconfig.get_paths().get('purelib', ''))",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            completed = None
        if completed is not None and completed.returncode == 0:
            purelib = completed.stdout.strip()
            if purelib and os.path.isabs(purelib):
                return Path(purelib)
    lib = venv / "lib"
    try:
        if lib.is_dir():
            for child in lib.iterdir():
                name = child.name
                if (
                    _PYTHON_VERSION_RE.fullmatch(name)
                    and (child / "site-packages").is_dir()
                ):
                    return child / "site-packages"
    except OSError:
        pass
    return lib / f"python{_PYTHON_VERSION_FALLBACK}" / "site-packages"


def resolve_python(
    project_root: Path,
    *,
    workspace: WorkspaceContext,
) -> ResolvedInterpreter:
    """Resolve the project-local Python interpreter, if any.

    The resolver is intentionally defensive: a missing or broken venv is
    reported as ``interpreter=None`` rather than as an exception, so the
    language server can fall back to the runtime's auto-provisioned
    interpreter (or the system Python). The ``workspace`` parameter is part
    of the language-agnostic protocol contract and is not consulted for the
    Python case; it is reserved for future resolvers that need it.
    """
    del workspace  # The Python resolver is fully driven by ``project_root``.

    venv = find_python_venv(project_root)
    if venv is None:
        return ResolvedInterpreter(language="python")

    interpreter_path = _python_executable_path(venv)
    if interpreter_path is None:
        return ResolvedInterpreter(language="python")

    interpreter_str = str(interpreter_path)
    if _is_windows():
        bin_dir = venv / "Scripts"
    else:
        bin_dir = venv / "bin"
    site_packages = _python_site_packages(venv, interpreter_path)

    return ResolvedInterpreter(
        language="python",
        interpreter=interpreter_str,
        path_prepend=(str(bin_dir),),
        environment={"VIRTUAL_ENV": str(venv)},
        settings={
            "python": {
                "pythonPath": interpreter_str,
                "analysis": {"extraPaths": [str(site_packages)]},
            },
        },
        initialization_options={"pythonPath": interpreter_str},
    )


# ---------------------------------------------------------------------------
# Stubs for other languages
# ---------------------------------------------------------------------------


def resolve_ruby(
    project_root: Path,
    *,
    workspace: WorkspaceContext,
) -> ResolvedInterpreter:
    """Stub Ruby resolver.

    Planned behaviour: locate ``vendor/bundle/ruby/*/bin`` (Bundler's project
    install path) or fall back to ``$(ruby -e 'print Gem.bindir')``. Today
    this returns ``interpreter=None``; the language-agnostic contract keeps
    the wiring intact so a future implementation is a one-line change.
    """
    del project_root, workspace
    return ResolvedInterpreter(language="ruby")


def resolve_node(
    project_root: Path,
    *,
    workspace: WorkspaceContext,
) -> ResolvedInterpreter:
    """Stub Node / TypeScript resolver.

    Planned behaviour: return ``<project_root>/node_modules/.bin`` so the
    language server picks up the project's locally installed binaries and
    libraries. Today this returns ``interpreter=None``; the language
    agnostic contract keeps the wiring intact so a future implementation is
    a one-line change.
    """
    del project_root, workspace
    return ResolvedInterpreter(language="node")


def resolve_go(
    project_root: Path,
    *,
    workspace: WorkspaceContext,
) -> ResolvedInterpreter:
    """Stub Go resolver.

    Planned behaviour: surface ``$GOPATH/bin`` (or ``$GOBIN``) and
    ``GOROOT/bin`` so ``gopls`` can locate installed Go tools. Today this
    returns ``interpreter=None``; the language-agnostic contract keeps the
    wiring intact so a future implementation is a one-line change.
    """
    del project_root, workspace
    return ResolvedInterpreter(language="go")


__all__ = [
    "InterpreterResolver",
    "ResolvedInterpreter",
    "find_python_venv",
    "resolve_go",
    "resolve_node",
    "resolve_python",
    "resolve_ruby",
]
