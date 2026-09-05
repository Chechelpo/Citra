from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .base import FilesystemInput, FilesystemOutput
from .edit import EditInput, execute as execute_edit
from .find import FindInput, execute as execute_find
from .glob import GlobInput, execute as execute_glob
from .grep import GrepInput, execute as execute_grep
from .read import ReadInput, execute as execute_read
from .read_binary import ReadBinaryInput, execute as execute_read_binary
from .read_raw import ReadRawInput, execute as execute_read_raw
from .scope import ScopedFilesystem
from .tree import TreeInput, execute as execute_tree
from .write import WriteInput, execute as execute_write


Executor = Callable[[Any, ScopedFilesystem], FilesystemOutput]


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Represent OperationSpec."""
    input_type: type[FilesystemInput[Any]]
    execute: Executor


OPERATIONS: dict[str, OperationSpec] = {
    ReadInput.operation: OperationSpec(ReadInput, execute_read),
    ReadRawInput.operation: OperationSpec(ReadRawInput, execute_read_raw),
    ReadBinaryInput.operation: OperationSpec(ReadBinaryInput, execute_read_binary),
    WriteInput.operation: OperationSpec(WriteInput, execute_write),
    EditInput.operation: OperationSpec(EditInput, execute_edit),
    GlobInput.operation: OperationSpec(GlobInput, execute_glob),
    GrepInput.operation: OperationSpec(GrepInput, execute_grep),
    TreeInput.operation: OperationSpec(TreeInput, execute_tree),
    FindInput.operation: OperationSpec(FindInput, execute_find),
}
