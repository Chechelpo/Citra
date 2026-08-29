"""Model-facing ``read_image`` tool.

Adapts a vision-friendly OpenAI ``image_url`` payload to Citra's text-based
tool-result protocol.

The tool reads a binary file from the workspace through the sandbox filesystem
worker (the new ``read_binary`` op), validates the file is a real image by
checking the leading magic bytes for the formats vision-capable OpenAI models
support, and returns the OpenAI ``image_url`` content-part payload as a JSON
string. The agent runner then forwards that string as a ``role="tool"``
message; the model client of the calling project is expected to surface the
structured payload to its vision input (for example by extracting the data URL
and feeding it through a multimodal user message).
"""

from __future__ import annotations

import base64
import json
import mimetypes
from typing import Any, override

from ...context import ExecutionContext
from ...sandbox.filesystem_ops import ReadBinaryInput, ReadBinaryOutput
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)
from ..tool import Tool, ToolDefinition


# Vision-capable OpenAI models accept these image formats. Keep the set
# aligned with the OpenAI Vision API reference. The MIME type is the
# authoritative label; the magic-byte checks below are the authoritative
# content check.
_SUPPORTED_IMAGE_FORMATS: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
)


# Image-format signatures: (leading bytes, mime type). The leading-byte check
# is the only reliable way to confirm a file is a real image regardless of the
# extension. ``mimetypes`` only inspects extensions and is therefore advisory
# rather than authoritative.
_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # full WEBP check happens after the RIFF/WEBP block
)


# Generous hard cap on raw bytes. The result is base64 encoded so the JSON
# payload is ~33% larger than the raw file. Capping at 20 MB keeps the JSON
# over-stdio worker responsive and stays well within the per-tool result
# token budget used by other image-aware Citra tools.
DEFAULT_MAX_BYTES: int = 20 * 1024 * 1024


def _detect_image_format(data: bytes) -> str | None:
    """Return the canonical image MIME type for the given bytes, or None.

    Performs a content check rather than relying on file extensions.
    """
    if not data:
        return None

    for signature, mime_type in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            if mime_type == "image/webp":
                # WEBP is a RIFF container. The format string 'WEBP' must
                # appear at bytes 8..12.
                if len(data) >= 12 and data[8:12] == b"WEBP":
                    return "image/webp"
                continue
            return mime_type

    return None


def _data_url(mime_type: str, raw_bytes: bytes) -> str:
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class ReadImage(Tool):
    """Read a local image and return an OpenAI ``image_url`` payload."""

    TOOL_ID = "read_image"

    # Vision payloads are large. Disable the result cache: a re-read with
    # identical arguments should still hand the model a fresh, up-to-date
    # image rather than a possibly-stale cached payload.
    CACHEABLE = False
    INVALIDATES_TOOL_CACHE = False

    # Allow plenty of headroom for base64 expansion. The tool's own
    # ``max_bytes`` argument is the more meaningful ceiling for callers.
    MAX_OUTPUT_TOKENS = 32_000

    _DESCRIPTION = (
        "Read a local image file and return it in a format suitable for "
        "sending to a vision-capable model. Supports PNG, JPEG, GIF, and "
        "WebP. The result is a JSON object with the OpenAI image_url "
        "shape, including a base64 data URL."
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="read_image",
            description=(
                "Read a local image and return it in a format suitable "
                "for sending to a vision-capable model. Supports PNG, "
                "JPEG, GIF, and WebP. The result is a JSON object that "
                "matches the OpenAI image_url content-part shape, with a "
                "base64 data URL. The path must point to a real image; "
                "the file's leading magic bytes are validated."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="path",
                        schema=JsonSchema.string(
                            description=(
                                "Image file path. Examples: "
                                "'diagram.png', '@source/assets/logo.webp'."
                            )
                        ),
                    ),
                    JsonProperty(
                        name="max_bytes",
                        schema=JsonSchema.integer(
                            description=(
                                "Optional upper bound on the raw file size "
                                "in bytes. Defaults to 20 MiB. Larger files "
                                "are rejected with an error."
                            )
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        ),
    )

    @classmethod
    @override
    def definitions_for_context(
        cls,
        context: ExecutionContext,
    ) -> tuple[ToolDefinition, ...]:
        del context
        return (ToolDefinition(definition=cls.DEFINITION),)

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        path_value = arguments.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("'path' must be a non-empty string.")

        max_bytes = arguments.get("max_bytes", DEFAULT_MAX_BYTES)
        if (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or max_bytes <= 0
        ):
            raise ValueError("'max_bytes' must be a positive integer.")

        order = ReadBinaryInput(
            path=path_value,
            max_bytes=max_bytes,
        )

        output: ReadBinaryOutput = self.context.filesystem.execute(order)

        try:
            raw_bytes = base64.b64decode(
                output.content_b64,
                validate=True,
            )
        except (ValueError, TypeError) as error:
            # The worker is expected to base64-encode; a decode failure is
            # an internal-protocol problem rather than a user error.
            raise RuntimeError(
                "Sandbox worker returned an invalid base64 payload for "
                f"'{path_value}': {error}"
            ) from error

        mime_type = _detect_image_format(raw_bytes)
        if mime_type is None:
            # Fall back to the worker's extension-based guess. This keeps
            # the tool useful for valid image files whose signatures we
            # have not whitelisted, and produces a clear error if the file
            # is genuinely not an image.
            mime_type = output.mime_type or mimetypes.guess_type(path_value)[0]
            if mime_type not in _SUPPORTED_IMAGE_FORMATS:
                raise ValueError(
                    f"Not a supported image: '{path_value}'. "
                    f"Detected mime_type: {mime_type!r}."
                )

        payload = {
            "type": "image_url",
            "image_url": {
                "url": _data_url(mime_type, raw_bytes),
            },
        }

        # The runner serializes this dict via json.dumps in
        # ``serialize_tool_result``; emitting the JSON here keeps the
        # tool self-contained and lets callers introspect the result.
        return json.dumps(
            payload,
            ensure_ascii=False,
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        path = arguments.get("path", "")
        max_bytes = arguments.get("max_bytes")
        if max_bytes is None:
            return f"path={path}"
        return f"path={path} | max_bytes={max_bytes}"

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return "invalid JSON result"
        if not isinstance(payload, dict):
            return f"{type(payload).__name__} result"
        url = payload.get("image_url", {}).get("url", "")
        if not isinstance(url, str):
            return "image_url result without string url"
        size = len(url)
        return f"image_url data URL, {size} chars"
