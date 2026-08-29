from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any

import pytest

from citra.sandbox.filesystem_ops import ReadBinaryInput, ReadBinaryOutput
from citra.sandbox.filesystem_ops.read_binary import DEFAULT_MAX_BYTES
from citra.sandbox.filesystem_ops.read_binary import execute as execute_read_binary
from citra.sandbox.filesystem_ops.scope import ScopedFilesystem
from citra.tools.default_registry import (
    _CORE_TOOL_TYPES,
    _DEFERRED_TOOL_TYPES,
    all_tools,
)
from citra.tools.transient import ReadImage


# A 1x1 transparent PNG (smallest valid PNG).
_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63000100000005000166aeb47a0000000049454e44ae426082"
)
# A minimal valid JPEG: SOI + APP0 + EOI markers. Not a real image but
# enough to exercise the JPEG signature path. The magic-byte detector
# only inspects the leading bytes; the read_image tool does the same.
_JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00\x00" + b"\xff\xd9"
# A minimal GIF89a header (no actual frame data, but the signature is what
# the detector inspects).
_GIF_BYTES = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00,\x00"
# Minimal RIFF/WEBP container.
_WEBP_BYTES = b"RIFF\x00\x00\x00\x00WEBPVP8 "


def _make_output(
    content_b64: str,
    size: int | None = None,
    mime_type: str | None = None,
) -> ReadBinaryOutput:
    if size is None:
        size = len(base64.b64decode(content_b64))
    return ReadBinaryOutput(
        content_b64=content_b64,
        size=size,
        mime_type=mime_type,
    )


class SpyFilesystem:
    """Captures the FilesystemInput sent by the tool layer.

    Matches the ``SandboxedFilesystem.execute(operation)`` signature: the
    operation carries its own arguments via ``to_arguments()``.
    """

    def __init__(
        self,
        *,
        output: ReadBinaryOutput | None = None,
        raise_value: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._output = output
        self._raise = raise_value

    def execute(self, operation):
        self.calls.append(
            (operation.operation, dict(operation.to_arguments()))
        )
        if self._raise is not None:
            raise self._raise
        assert self._output is not None
        return self._output


def _tool(filesystem: SpyFilesystem) -> ReadImage:
    """Return a ReadImage with a stub context, bypassing Tool.__init__.

    ``Tool.__init__`` resolves a model-specific definition, which requires a
    real ``ExecutionContext``. The unit tests in this module exercise the
    pure read-and-format logic and do not need a full execution context.
    """
    instance = ReadImage.__new__(ReadImage)
    instance._Tool__context = SimpleNamespace(filesystem=filesystem)  # type: ignore[attr-defined]
    return instance


# ---------------------------------------------------------------------------
# Magic-byte detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_bytes", "expected_mime"),
    [
        (_PNG_BYTES, "image/png"),
        (_JPEG_BYTES, "image/jpeg"),
        (_GIF_BYTES, "image/gif"),
        (_WEBP_BYTES, "image/webp"),
        (b"", None),
        (b"plain text content", None),
        # RIFF but not WEBP: not a supported image.
        (b"RIFF\x00\x00\x00\x00WAVE", None),
    ],
)
def test_detect_image_format(
    raw_bytes: bytes,
    expected_mime: str | None,
) -> None:
    from citra.tools.transient.read_image import _detect_image_format

    assert _detect_image_format(raw_bytes) == expected_mime


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def test_read_image_returns_openai_image_url_payload() -> None:
    filesystem = SpyFilesystem(
        output=_make_output(base64.b64encode(_PNG_BYTES).decode("ascii"))
    )
    result = _tool(filesystem)._execute({"path": "logo.png"})

    import json as _json

    payload = _json.loads(result)
    assert payload["type"] == "image_url"
    assert payload["image_url"]["url"].startswith("data:image/png;base64,")
    assert (
        payload["image_url"]["url"]
        == f"data:image/png;base64,{base64.b64encode(_PNG_BYTES).decode('ascii')}"
    )
    assert filesystem.calls == [
        (
            "read_binary",
            {"path": "logo.png", "max_bytes": DEFAULT_MAX_BYTES},
        )
    ]


def test_read_image_rejects_non_image_payload() -> None:
    plaintext = b"hello world\n"
    filesystem = SpyFilesystem(
        output=_make_output(base64.b64encode(plaintext).decode("ascii"))
    )

    with pytest.raises(ValueError, match="Not a supported image"):
        _tool(filesystem)._execute({"path": "notes.txt"})


def test_read_image_rejects_missing_path() -> None:
    filesystem = SpyFilesystem()
    with pytest.raises(ValueError, match="'path' must be a non-empty string"):
        _tool(filesystem)._execute({})


def test_read_image_rejects_non_positive_max_bytes() -> None:
    filesystem = SpyFilesystem()
    with pytest.raises(ValueError, match="'max_bytes' must be a positive integer"):
        _tool(filesystem)._execute({"path": "x.png", "max_bytes": 0})

    with pytest.raises(ValueError, match="'max_bytes' must be a positive integer"):
        _tool(filesystem)._execute({"path": "x.png", "max_bytes": -1})


def test_read_image_propagates_filesystem_errors() -> None:
    filesystem = SpyFilesystem(raise_value=FileNotFoundError("missing.png"))
    with pytest.raises(FileNotFoundError):
        _tool(filesystem)._execute({"path": "missing.png"})


def test_read_image_format_result_log_reports_data_url_size() -> None:
    filesystem = SpyFilesystem(
        output=_make_output(base64.b64encode(_PNG_BYTES).decode("ascii"))
    )
    tool = _tool(filesystem)
    result = tool._execute({"path": "logo.png"})

    log = tool.format_result_log(result)
    assert "image_url data URL" in log
    assert "chars" in log


def test_read_image_definition_exposes_image_url_tool() -> None:
    assert ReadImage.TOOL_ID == "read_image"
    definition = ReadImage.DEFINITION
    assert definition.function.name == "read_image"
    params = definition.function.parameters.to_dict()
    assert set(params["properties"].keys()) == {"path", "max_bytes"}
    assert params["properties"]["path"]["type"] == "string"


# ---------------------------------------------------------------------------
# read_binary op
# ---------------------------------------------------------------------------


def _scope(monkeypatch, tmp_path) -> ScopedFilesystem:
    roots = {
        "CITRA_WORKSPACE": tmp_path / "workspace",
        "CITRA_SOURCE": tmp_path / "source",
        "HOME": tmp_path / "home",
        "CITRA_TMP": tmp_path / "tmp",
        "CITRA_CACHE": tmp_path / "cache",
        "XDG_CONFIG_HOME": tmp_path / "config",
        "XDG_DATA_HOME": tmp_path / "data",
        "XDG_RUNTIME_DIR": tmp_path / "runtime",
    }
    for name, path in roots.items():
        path.mkdir()
        monkeypatch.setenv(name, str(path))
    return ScopedFilesystem()


def test_read_binary_input_round_trip_arguments() -> None:
    parsed = ReadBinaryInput.parse({"path": "logo.png", "max_bytes": 1024})
    assert parsed.path == "logo.png"
    assert parsed.max_bytes == 1024
    assert parsed.to_arguments() == {
        "path": "logo.png",
        "max_bytes": 1024,
    }


def test_read_binary_input_rejects_non_positive_max_bytes() -> None:
    with pytest.raises(ValueError, match="'max_bytes' must be a positive integer"):
        ReadBinaryInput.parse({"path": "x", "max_bytes": 0})


def test_read_binary_input_uses_default_max_bytes() -> None:
    parsed = ReadBinaryInput.parse({"path": "x"})
    assert parsed.max_bytes == DEFAULT_MAX_BYTES


def test_read_binary_execute_reads_and_detects_mime(
    monkeypatch,
    tmp_path,
) -> None:
    fs = _scope(monkeypatch, tmp_path)
    target = tmp_path / "workspace" / "diagram.png"
    target.write_bytes(_PNG_BYTES)

    output = execute_read_binary(
        ReadBinaryInput(path="diagram.png"),
        fs,
    )

    assert output.size == len(_PNG_BYTES)
    assert output.content_b64 == base64.b64encode(_PNG_BYTES).decode("ascii")
    assert output.mime_type == "image/png"


def test_read_binary_execute_rejects_oversize_files(
    monkeypatch,
    tmp_path,
) -> None:
    fs = _scope(monkeypatch, tmp_path)
    target = tmp_path / "workspace" / "big.png"
    # 1024 bytes of zero, with a tiny cap of 100 bytes.
    target.write_bytes(b"\x00" * 1024)

    with pytest.raises(ValueError, match="File too large to read as binary"):
        execute_read_binary(
            ReadBinaryInput(path="big.png", max_bytes=100),
            fs,
        )


def test_read_binary_execute_rejects_missing_file(
    monkeypatch,
    tmp_path,
) -> None:
    fs = _scope(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError):
        execute_read_binary(
            ReadBinaryInput(path="does-not-exist.png"),
            fs,
        )


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_read_image_is_in_default_deferred_registry() -> None:
    assert ReadImage in _DEFERRED_TOOL_TYPES
    assert ReadImage not in _CORE_TOOL_TYPES
    assert ReadImage in all_tools(are_deferred=True)


def test_read_image_appears_in_greenfield_mode() -> None:
    from citra.modes.greenfield import LongTaskHorizon

    assert ReadImage in LongTaskHorizon._TOOLS.deferred_tools
