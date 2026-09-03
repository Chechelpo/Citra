"""Read the runtime interpreter declared by an ELF executable."""

from __future__ import annotations

import logging
import os
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

_ELF_MAGIC = b"\x7fELF"
_ELF_CLASS_32 = 1
_ELF_CLASS_64 = 2
_ELF_DATA_LITTLE_ENDIAN = 1
_ELF_DATA_BIG_ENDIAN = 2
_PT_INTERP = 3
_MAX_PROGRAM_HEADERS = 4096
_MAX_INTERPRETER_BYTES = 4096


def elf_interpreter(executable: Path) -> Path | None:
    """Return the absolute ``PT_INTERP`` path declared by an ELF executable."""
    try:
        with executable.open("rb") as stream:
            header = stream.read(64)
            layout = _program_header_layout(header)
            if layout is None:
                return None
            offset, entry_size, entry_count, byte_order, elf_class = layout
            if entry_count > _MAX_PROGRAM_HEADERS:
                logger.warning(
                    "Rejected ELF executable with excessive program headers",
                    extra={
                        "origin": __name__,
                        "executable": str(executable),
                        "program_headers": entry_count,
                    },
                )
                return None
            for index in range(entry_count):
                stream.seek(offset + index * entry_size)
                entry = stream.read(entry_size)
                interpreter_offset = _interpreter_offset(
                    entry,
                    byte_order=byte_order,
                    elf_class=elf_class,
                )
                if interpreter_offset is None:
                    continue
                stream.seek(interpreter_offset)
                raw = stream.read(_MAX_INTERPRETER_BYTES).split(b"\0", 1)[0]
                interpreter = Path(os.fsdecode(raw))
                if not interpreter.is_absolute():
                    logger.warning(
                        "Rejected non-absolute ELF interpreter",
                        extra={
                            "origin": __name__,
                            "executable": str(executable),
                            "interpreter": str(interpreter),
                        },
                    )
                    return None
                logger.debug(
                    "Discovered ELF interpreter",
                    extra={
                        "origin": __name__,
                        "executable": str(executable),
                        "interpreter": str(interpreter),
                    },
                )
                return interpreter
    except (OSError, OverflowError, struct.error) as error:
        logger.debug(
            "Could not inspect ELF interpreter",
            extra={
                "origin": __name__,
                "executable": str(executable),
                "error": str(error),
            },
        )
    return None


def _program_header_layout(
    header: bytes,
) -> tuple[int, int, int, str, int] | None:
    """Decode the ELF program-header table location and representation."""
    if len(header) < 52 or header[:4] != _ELF_MAGIC:
        return None
    elf_class = header[4]
    encoding = header[5]
    if encoding == _ELF_DATA_LITTLE_ENDIAN:
        byte_order = "<"
    elif encoding == _ELF_DATA_BIG_ENDIAN:
        byte_order = ">"
    else:
        return None
    if elf_class == _ELF_CLASS_64:
        if len(header) < 58:
            return None
        offset = struct.unpack_from(f"{byte_order}Q", header, 32)[0]
        entry_size = struct.unpack_from(f"{byte_order}H", header, 54)[0]
        entry_count = struct.unpack_from(f"{byte_order}H", header, 56)[0]
    elif elf_class == _ELF_CLASS_32:
        offset = struct.unpack_from(f"{byte_order}I", header, 28)[0]
        entry_size = struct.unpack_from(f"{byte_order}H", header, 42)[0]
        entry_count = struct.unpack_from(f"{byte_order}H", header, 44)[0]
    else:
        return None
    if offset <= 0 or entry_size <= 0:
        return None
    return offset, entry_size, entry_count, byte_order, elf_class


def _interpreter_offset(
    entry: bytes,
    *,
    byte_order: str,
    elf_class: int,
) -> int | None:
    """Return the file offset from a ``PT_INTERP`` program-header entry."""
    if len(entry) < 8:
        return None
    segment_type = struct.unpack_from(f"{byte_order}I", entry, 0)[0]
    if segment_type != _PT_INTERP:
        return None
    if elf_class == _ELF_CLASS_64:
        if len(entry) < 16:
            return None
        return struct.unpack_from(f"{byte_order}Q", entry, 8)[0]
    return struct.unpack_from(f"{byte_order}I", entry, 4)[0]


__all__ = ["elf_interpreter"]
