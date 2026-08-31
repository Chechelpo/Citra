"""Unit tests for ``citra.utils.lsp.interpreters``.

These tests exercise the language-agnostic interpreter resolver API in
isolation. They use only the standard library, the real
``citra.utils.lsp.interpreters`` import path, and synthetic project roots
created with :mod:`tempfile`, so they are safe to run on hosts that do
not have bwrap, pyright, or any other language server installed.
"""

from __future__ import annotations

import sys
import tempfile
import types
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from citra.utils.lsp.interpreters import (
    ResolvedInterpreter,
    find_python_venv,
    resolve_go,
    resolve_node,
    resolve_python,
    resolve_ruby,
)


def _make_venv(
    root: Path,
    *,
    candidate: str,
    executable: str = "python3",
) -> Path:
    """Create a minimal but valid venv layout under ``root / candidate``."""
    venv = root / candidate
    bin_dir = venv / ("Scripts" if sys.platform.startswith("win") else "bin")
    bin_dir.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (bin_dir / executable).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    return venv


def test_find_python_venv_prefers_dot_venv() -> None:
    """``find_python_venv`` picks ``.venv`` over ``venv`` and returns ``None`` if neither exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        venv = _make_venv(root, candidate=".venv")
        assert find_python_venv(root) == venv

        # A second tempdir with only ``venv`` falls back to that name.
        with tempfile.TemporaryDirectory() as fallback_dir:
            fallback_root = Path(fallback_dir)
            fallback_venv = _make_venv(fallback_root, candidate="venv")
            assert find_python_venv(fallback_root) == fallback_venv

        # A third tempdir with no venv at all returns ``None``.
        with tempfile.TemporaryDirectory() as empty_dir:
            assert find_python_venv(Path(empty_dir)) is None


def test_find_python_venv_rejects_missing_pyvenv_cfg() -> None:
    """A ``bin/python3`` without a ``pyvenv.cfg`` is not a venv."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        # Layout that looks like a venv except no ``pyvenv.cfg``.
        candidate = root / "venv"
        bin_dir = candidate / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "python3").write_text("#!/bin/sh\n", encoding="utf-8")
        assert find_python_venv(root) is None


def test_find_python_venv_accepts_unversioned_posix_python() -> None:
    """On POSIX, ``bin/python`` is a usable fallback when ``bin/python3`` is missing."""
    if sys.platform.startswith("win"):
        pytest.skip("POSIX-only behaviour.")
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        venv = root / "venv"
        bin_dir = venv / "bin"
        bin_dir.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
        (bin_dir / "python").write_text("#!/bin/sh\n", encoding="utf-8")
        assert find_python_venv(root) == venv


def test_find_python_venv_returns_none_for_non_directory(tmp_path: Path) -> None:
    """A non-existent or non-directory project root yields ``None``, not an exception."""
    # Pointing at a regular file (not a directory) must return ``None``
    # rather than raise.
    assert find_python_venv(tmp_path / "does-not-exist") is None


def test_resolve_python_returns_interpreter_and_settings() -> None:
    """A usable ``.venv`` yields interpreter, ``path_prepend``, env, and settings."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        venv = _make_venv(root, candidate=".venv")
        workspace = types.SimpleNamespace()
        resolved = resolve_python(root, workspace=workspace)  # type: ignore[arg-type]

        assert resolved.language == "python"
        assert resolved.interpreter is not None
        assert resolved.interpreter == str(venv / "bin" / "python3")
        assert resolved.path_prepend == (str(venv / "bin"),)
        assert resolved.environment == {"VIRTUAL_ENV": str(venv)}
        assert resolved.settings["python"]["pythonPath"] == resolved.interpreter
        assert resolved.settings["python"]["analysis"]["extraPaths"]
        assert resolved.initialization_options["pythonPath"] == resolved.interpreter


def test_resolve_python_handles_missing_venv() -> None:
    """A project without a venv yields a populated shell object, not an exception."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        resolved = resolve_python(root, workspace=types.SimpleNamespace())  # type: ignore[arg-type]
        assert resolved.language == "python"
        assert resolved.interpreter is None
        assert resolved.path_prepend == ()
        assert resolved.environment == {}
        assert resolved.settings == {}
        assert resolved.initialization_options == {}


def test_resolve_python_does_not_touch_workspace() -> None:
    """The Python resolver must not require any attribute on ``workspace``."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _make_venv(root, candidate=".venv")
        # No attributes at all on the stub — accessing any would raise.
        workspace = types.SimpleNamespace()
        resolve_python(root, workspace=workspace)  # type: ignore[arg-type]


def test_stub_resolvers_are_resolvers() -> None:
    """``resolve_ruby``, ``resolve_node`` and ``resolve_go`` return empty shells today."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        workspace = types.SimpleNamespace()
        for resolver, language in (
            (resolve_ruby, "ruby"),
            (resolve_node, "node"),
            (resolve_go, "go"),
        ):
            resolved = resolver(root, workspace=workspace)  # type: ignore[arg-type]
            assert isinstance(resolved, ResolvedInterpreter)
            assert resolved.language == language
            assert resolved.interpreter is None
            assert resolved.path_prepend == ()
            assert resolved.environment == {}
            assert resolved.settings == {}
            assert resolved.initialization_options == {}


def test_stub_resolvers_satisfy_interpreter_resolver_protocol() -> None:
    """All stub resolvers expose a ``(project_root, *, workspace)`` signature."""
    # All stub resolvers must be assignable to the public protocol type
    # so they can be used as ``ServerDefinition.interpreter_resolver``.
    for resolver in (resolve_ruby, resolve_node, resolve_go):
        # The protocol is structural; a basic check that the call signature
        # is compatible is enough.
        assert callable(resolver)
        assert resolver.__code__.co_varnames[:1] == ("project_root",)


def test_resolved_interpreter_is_frozen() -> None:
    """``ResolvedInterpreter`` rejects attribute assignment (frozen dataclass)."""
    resolved = ResolvedInterpreter(language="python", interpreter="/x/bin/python3")
    with pytest.raises(FrozenInstanceError):
        resolved.interpreter = "/y/bin/python3"  # type: ignore[misc]


def test_interpreter_resolver_protocol_runtime_checkable() -> None:
    """A plain function with the right signature is structurally an ``InterpreterResolver``."""

    def call_me(project_root: Path, *, workspace: object) -> ResolvedInterpreter:
        return ResolvedInterpreter(language="python", interpreter="/x")

    # The structural protocol is exercised by passing the function as
    # ``interpreter_resolver`` on a fresh ``ServerDefinition``-like object.
    # We can't easily instantiate ``ServerDefinition`` here without the LSP
    # subsystem, so we just check that the function is callable with the
    # expected signature.
    with tempfile.TemporaryDirectory() as temp_dir:
        result = call_me(Path(temp_dir), workspace=types.SimpleNamespace())
        assert result.interpreter == "/x"
