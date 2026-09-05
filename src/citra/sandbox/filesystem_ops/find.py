"""Combined filesystem search (filename + extension + content) primitive.

``Find`` merges the responsibilities of Unix ``find``, glob, and grep into one
sandbox-scoped operation. The agent can:

* recursively walk one or more root paths
* filter by filename glob and/or file extension
* optionally search file contents (literal or regex, with case sensitivity)
* prune excluded directories before descending; a built-in junk set
  (``DEFAULT_TREE_SKIPS`` — ``__pycache__``, ``.venv``, ``.git``,
  ``node_modules``, etc.) is skipped by default unless the caller opts out
  with ``use_default_skips=False``
* cap traversal depth
* ask for ``mode="files"`` (path list, ``GlobOutput``-shaped),
  ``mode="matches"`` (per-file structured hits with context windows), or
  ``mode="count"`` (one ``(path, count)`` row per file with at least one
  match, mirroring :mod:`grep`'s ``count`` mode)

The operation reuses ``_walk_files``, ``_glob_matches``, ``MAX_FILE_BYTES``,
``MAX_LINE_CHARS``, and the binary-skip heuristic from :mod:`grep` to keep a
single, well-tested traversal core across filesystem ops.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import (
    FilesystemInput,
    FilesystemOutput,
    require_payload_dict,
)
from .grep import (
    MAX_FILE_BYTES,
    MAX_LINE_CHARS,
    _glob_matches,
)
from .scope import ScopedFilesystem
from .tree import DEFAULT_TREE_SKIPS

OUTPUT_MODES = ("files", "matches", "count")
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000


def _require_paths(arguments: dict[str, Any]) -> tuple[str, ...]:
    """Validate and return the ``paths`` array from model arguments."""
    raw = arguments.get("paths")
    if raw is None:
        raise ValueError("'paths' is required.")
    if not isinstance(raw, list) or not raw:
        raise ValueError("'paths' must be a non-empty array of strings.")
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value:
            raise ValueError(f"'paths[{index}]' must be a non-empty string.")
    return tuple(raw)


def _normalize_name(
    arguments: dict[str, Any],
) -> tuple[str, ...] | None:
    """Validate the optional ``name`` field and normalize it to a tuple."""
    if "name" not in arguments:
        return None
    raw = arguments["name"]
    if isinstance(raw, str):
        if not raw:
            raise ValueError("'name' must be a non-empty string or array.")
        return (raw,)
    if isinstance(raw, list):
        if not raw:
            raise ValueError("'name' must be a non-empty string or array.")
        for index, value in enumerate(raw):
            if not isinstance(value, str) or not value:
                raise ValueError(f"'name[{index}]' must be a non-empty string.")
        return tuple(raw)
    raise ValueError("'name' must be a string or array of strings.")


def _normalize_extensions(
    arguments: dict[str, Any],
) -> tuple[str, ...] | None:
    """Validate the optional ``extensions`` field and normalize leading dots."""
    if "extensions" not in arguments:
        return None
    raw = arguments["extensions"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("'extensions' must be a non-empty array of strings.")
    normalized: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value:
            raise ValueError(f"'extensions[{index}]' must be a non-empty string.")
        stripped = value.lstrip(".")
        if not stripped:
            raise ValueError(
                f"'extensions[{index}]' must contain at least one character."
            )
        if stripped not in normalized:
            normalized.append(stripped)
    return tuple(normalized)


def _normalize_exclude(
    arguments: dict[str, Any],
) -> tuple[str, ...] | None:
    """Validate the optional ``exclude`` field."""
    if "exclude" not in arguments:
        return None
    raw = arguments["exclude"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("'exclude' must be a non-empty array of strings.")
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value:
            raise ValueError(f"'exclude[{index}]' must be a non-empty string.")
    return tuple(raw)


def _require_optional_non_negative_int(
    arguments: dict[str, Any],
    name: str,
) -> int | None:
    """Validate an optional non-negative integer argument."""
    if name not in arguments:
        return None
    value = arguments[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{name}' must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"'{name}' must be non-negative.")
    return value


def _require_positive_int(
    arguments: dict[str, Any],
    name: str,
) -> int | None:
    """Validate an optional strictly positive integer argument."""
    if name not in arguments:
        return None
    value = arguments[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{name}' must be an integer.")
    if value < 1:
        raise ValueError(f"'{name}' must be at least 1.")
    return value


def _require_optional_bool(
    arguments: dict[str, Any],
    name: str,
) -> bool | None:
    """Validate an optional boolean argument."""
    if name not in arguments:
        return None
    value = arguments[name]
    if not isinstance(value, bool):
        raise ValueError(f"'{name}' must be a boolean.")
    return value


def _require_optional_string(
    arguments: dict[str, Any],
    name: str,
) -> str | None:
    """Validate an optional string argument."""
    if name not in arguments:
        return None
    value = arguments[name]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{name}' must be a string.")
    return value


def _path_matches_any_name(
    name_patterns: tuple[str, ...],
    path: Path,
    base: Path,
) -> bool:
    """Return True if any of the supplied glob patterns match ``path``."""
    for pattern in name_patterns:
        if _glob_matches(pattern, path, base):
            return True
    return False


def _path_matches_any_extension(
    extensions: tuple[str, ...],
    path: Path,
) -> bool:
    """Return True if the file extension matches any of the supplied values."""
    suffix = path.suffix.lstrip(".").casefold()
    if not suffix:
        return False
    return any(suffix == candidate.casefold() for candidate in extensions)


def _is_excluded(
    directory: Path,
    base: Path,
    patterns: tuple[str, ...],
) -> bool:
    """Decide whether a directory should be pruned from the walk.

    The match is performed against both the directory's ``name`` and its
    relative POSIX path against ``base``, which covers both simple basename
    excludes (``"node_modules"``) and shallow-directory glob excludes
    (``"node_modules/**"``, ``"src/vendor/**"``).
    """
    if not patterns:
        return False
    try:
        relative = directory.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        relative = directory.name
    candidate_names = (directory.name, relative)
    for pattern in patterns:
        for candidate in candidate_names:
            if candidate == pattern or fnmatch.fnmatchcase(candidate, pattern):
                return True
    return False


def _compile_content_expression(
    content: str,
    *,
    regex: bool,
    case_sensitive: bool,
) -> re.Pattern[str]:
    """Compile the model-supplied content expression or raise ``ValueError``."""
    flags = 0 if case_sensitive else re.IGNORECASE
    source = content if regex else re.escape(content)
    try:
        return re.compile(source, flags)
    except re.error as error:
        raise ValueError(f"Invalid search pattern: {error}") from error


@dataclass(frozen=True, slots=True)
class FindMatch:
    """One content hit within a single file."""

    line: int
    text: str
    context: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Any) -> FindMatch:
        """Create a match from a worker payload."""
        if not isinstance(payload, dict):
            raise ValueError("Find match must be an object.")
        line = payload.get("line")
        text = payload.get("text")
        context = payload.get("context")
        if not isinstance(line, int) or line < 1:
            raise ValueError("Find match 'line' must be a positive integer.")
        if not isinstance(text, str):
            raise ValueError("Find match 'text' must be a string.")
        if not isinstance(context, list) or not all(
            isinstance(value, str) for value in context
        ):
            raise ValueError("Find match 'context' must be an array of strings.")
        return cls(line=line, text=text, context=tuple(context))

    def to_payload(self) -> dict[str, Any]:
        """Convert the value to payload."""
        return {
            "line": self.line,
            "text": self.text,
            "context": list(self.context),
        }


@dataclass(frozen=True, slots=True)
class FindFileMatches:
    """A per-file group of :class:`FindMatch` instances."""

    path: str
    matches: tuple[FindMatch, ...]

    @classmethod
    def from_payload(cls, payload: Any) -> FindFileMatches:
        """Create an instance from payload."""
        if not isinstance(payload, dict):
            raise ValueError("Find file-matches entry must be an object.")
        path = payload.get("path")
        matches = payload.get("matches")
        if not isinstance(path, str):
            raise ValueError("Find file-matches 'path' must be a string.")
        if not isinstance(matches, list):
            raise ValueError("Find file-matches 'matches' must be an array.")
        return cls(
            path=path,
            matches=tuple(FindMatch.from_payload(item) for item in matches),
        )

    def to_payload(self) -> dict[str, Any]:
        """Convert the value to payload."""
        return {
            "path": self.path,
            "matches": [match.to_payload() for match in self.matches],
        }


@dataclass(frozen=True, slots=True)
class FindOutput(FilesystemOutput):
    """Structured result of one ``find`` operation.

    The shape depends on ``mode``:

    * ``mode="files"`` carries ``paths`` (matching ``GlobOutput``).
    * ``mode="matches"`` carries ``results`` of per-file matches.
    * ``mode="count"`` carries ``counts``: one ``(path, count)`` entry per
      file with at least one content match, where ``count`` is the per-file
      match total.
    """

    mode: str = "files"
    paths: tuple[str, ...] = ()
    results: tuple[FindFileMatches, ...] = ()
    counts: tuple[tuple[str, int], ...] = ()
    truncated: bool = False

    @classmethod
    def from_payload(cls, payload: Any) -> FindOutput:
        """Deserialize the worker payload."""
        raw = require_payload_dict(payload)
        mode = raw.get("mode", "files")
        if not isinstance(mode, str) or mode not in OUTPUT_MODES:
            raise ValueError("Find output 'mode' is invalid.")

        truncated = raw.get("truncated", False)
        if not isinstance(truncated, bool):
            raise ValueError("Find output 'truncated' must be boolean.")

        if mode == "files":
            raw_paths = raw.get("paths", [])
            if not isinstance(raw_paths, list) or not all(
                isinstance(path, str) for path in raw_paths
            ):
                raise ValueError("Find output 'paths' must be an array of strings.")
            return cls(
                mode=mode,
                paths=tuple(raw_paths),
                results=(),
                counts=(),
                truncated=truncated,
            )

        if mode == "count":
            raw_counts = raw.get("counts", [])
            if not isinstance(raw_counts, list) or not all(
                isinstance(item, dict)
                and isinstance(item.get("path"), str)
                and isinstance(item.get("count"), int)
                and not isinstance(item.get("count"), bool)
                and item.get("count") >= 0
                for item in raw_counts
            ):
                raise ValueError(
                    "Find output 'counts' must be an array of "
                    "{'path': str, 'count': non-negative int} objects."
                )
            return cls(
                mode=mode,
                paths=(),
                results=(),
                counts=tuple(
                    (item["path"], item["count"]) for item in raw_counts
                ),
                truncated=truncated,
            )

        raw_results = raw.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError("Find output 'results' must be an array.")
        return cls(
            mode=mode,
            paths=(),
            results=tuple(FindFileMatches.from_payload(item) for item in raw_results),
            counts=(),
            truncated=truncated,
        )

    def to_payload(self) -> dict[str, Any]:
        """Convert the value to payload."""
        if self.mode == "files":
            return {
                "mode": self.mode,
                "paths": list(self.paths),
                "truncated": self.truncated,
            }
        if self.mode == "count":
            return {
                "mode": self.mode,
                "counts": [
                    {"path": path, "count": count} for path, count in self.counts
                ],
                "truncated": self.truncated,
            }
        return {
            "mode": self.mode,
            "results": [item.to_payload() for item in self.results],
            "truncated": self.truncated,
        }

    def render(self) -> str:
        """Render the full result in the legacy textual format."""
        if self.mode == "files":
            if not self.paths:
                text = "no matches"
            else:
                text = "\n".join(self.paths)
        elif self.mode == "count":
            if not self.counts:
                text = "no matches"
            else:
                text = "\n".join(f"{path}: {count}" for path, count in self.counts)
        else:
            if not self.results:
                text = "no matches"
            else:
                rendered: list[str] = []
                for entry in self.results:
                    rendered.append(f"===== {entry.path} =====")
                    for match in entry.matches:
                        rendered.append(f"{entry.path}:{match.line}: {match.text}")
                        if match.context:
                            for index, context_line in enumerate(match.context):
                                marker = (
                                    "  ›"
                                    if index != len(match.context) - 1
                                    or context_line != match.text
                                    else "  ›"
                                )
                                rendered.append(f"{marker} {context_line}")
                text = "\n".join(rendered)
        if self.truncated:
            text += "\n... <truncated: showing first results>"
        return text


@dataclass(frozen=True, slots=True)
class FindInput(FilesystemInput[FindOutput]):
    """Combined filesystem search request."""

    operation = "find"
    output_type = FindOutput

    paths: tuple[str, ...]
    name: tuple[str, ...] | None = None
    extensions: tuple[str, ...] | None = None
    content: str | None = None
    regex: bool = False
    case_sensitive: bool = True
    exclude: tuple[str, ...] | None = None
    max_depth: int | None = None
    context: int = 0
    limit: int = DEFAULT_LIMIT
    mode: str = "files"
    use_default_skips: bool = True

    def __post_init__(self) -> None:
        """Validate the search request."""
        if not isinstance(self.paths, tuple) or not self.paths:
            raise ValueError("'paths' must be a non-empty tuple of strings.")
        for index, value in enumerate(self.paths):
            if not isinstance(value, str) or not value:
                raise ValueError(f"'paths[{index}]' must be a non-empty string.")
        if self.name is not None:
            if not isinstance(self.name, tuple) or not self.name:
                raise ValueError("'name' must be a non-empty tuple of strings.")
            for index, value in enumerate(self.name):
                if not isinstance(value, str) or not value:
                    raise ValueError(f"'name[{index}]' must be a non-empty string.")
        if self.extensions is not None:
            if not isinstance(self.extensions, tuple) or not self.extensions:
                raise ValueError("'extensions' must be a non-empty tuple of strings.")
            for index, value in enumerate(self.extensions):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"'extensions[{index}]' must be a non-empty string."
                    )
        if self.content is not None and not isinstance(self.content, str):
            raise ValueError("'content' must be a string when provided.")
        if not isinstance(self.regex, bool):
            raise ValueError("'regex' must be a boolean.")
        if not isinstance(self.case_sensitive, bool):
            raise ValueError("'case_sensitive' must be a boolean.")
        if self.exclude is not None:
            if not isinstance(self.exclude, tuple) or not self.exclude:
                raise ValueError("'exclude' must be a non-empty tuple of strings.")
            for index, value in enumerate(self.exclude):
                if not isinstance(value, str) or not value:
                    raise ValueError(f"'exclude[{index}]' must be a non-empty string.")
        if self.max_depth is not None:
            if (
                isinstance(self.max_depth, bool)
                or not isinstance(self.max_depth, int)
                or self.max_depth < 0
            ):
                raise ValueError("'max_depth' must be a non-negative integer.")
        if (
            isinstance(self.context, bool)
            or not isinstance(self.context, int)
            or self.context < 0
        ):
            raise ValueError("'context' must be a non-negative integer.")
        if (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or not 1 <= self.limit <= MAX_LIMIT
        ):
            raise ValueError(f"'limit' must be between 1 and {MAX_LIMIT}.")
        if self.mode not in OUTPUT_MODES:
            raise ValueError("'mode' must be one of: " + ", ".join(OUTPUT_MODES) + ".")
        if not isinstance(self.use_default_skips, bool):
            raise ValueError("'use_default_skips' must be a boolean.")
        if self.mode == "count" and not self.content:
            # Per-file counts have nothing to count without a content filter;
            # fail fast so the model sees a clear error before the worker
            # dispatches a filesystem walk.
            raise ValueError("'content' is required when 'mode' is 'count'.")
        # Pre-compile regex up-front so invalid patterns surface before we
        # touch the filesystem. ``compile_content_expression`` raises
        # ``ValueError`` for bad patterns; ``content`` may also be None.
        if self.content is not None and self.regex:
            _compile_content_expression(
                self.content,
                regex=self.regex,
                case_sensitive=self.case_sensitive,
            )

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> FindInput:
        """Parse canonical and harness-alias arguments.

        The model-facing schema advertises camelCase names (``maxDepth``,
        ``caseSensitive``, ``limit``); the worker wire protocol and internal
        representation use snake_case. Both spellings are accepted here so
        transient tools can pass arguments through verbatim.
        """
        paths = _require_paths(arguments)
        name = _normalize_name(arguments)
        extensions = _normalize_extensions(arguments)
        exclude = _normalize_exclude(arguments)

        content = _require_optional_string(arguments, "content")

        regex_value = arguments.get("regex")
        if regex_value is None:
            regex_value = arguments.get("is_regex", False)
        if not isinstance(regex_value, bool):
            raise ValueError("'regex' must be a boolean.")
        regex = regex_value

        case_sensitive_value = arguments.get(
            "caseSensitive",
            arguments.get(
                "case_sensitive",
                arguments.get("caseInsensitive", True),
            ),
        )
        # ``caseInsensitive`` is a Gemini-style alias; the inverse of
        # ``case_insensitive`` defaults to True when neither flag is set.
        if not isinstance(case_sensitive_value, bool):
            raise ValueError("'caseSensitive' must be a boolean.")
        case_sensitive = case_sensitive_value

        max_depth = _require_optional_non_negative_int(arguments, "maxDepth")
        if max_depth is None:
            max_depth = _require_optional_non_negative_int(arguments, "max_depth")

        context_value = arguments.get("context")
        if context_value is not None and (
            isinstance(context_value, bool)
            or not isinstance(context_value, int)
            or context_value < 0
        ):
            raise ValueError("'context' must be a non-negative integer.")
        context = 0 if context_value is None else context_value

        limit_value = arguments.get(
            "limit", arguments.get("max_results", DEFAULT_LIMIT)
        )
        if (
            isinstance(limit_value, bool)
            or not isinstance(limit_value, int)
            or not 1 <= limit_value <= MAX_LIMIT
        ):
            raise ValueError(f"'limit' must be between 1 and {MAX_LIMIT}.")
        limit = limit_value

        mode = arguments.get("mode", arguments.get("output_mode", "files"))
        if not isinstance(mode, str):
            raise ValueError("'mode' must be a string.")
        if mode not in OUTPUT_MODES:
            raise ValueError("'mode' must be one of: " + ", ".join(OUTPUT_MODES) + ".")

        if "use_default_skips" in arguments:
            use_default_skips = arguments["use_default_skips"]
        elif "useDefaultSkips" in arguments:
            use_default_skips = arguments["useDefaultSkips"]
        else:
            use_default_skips = True
        if not isinstance(use_default_skips, bool):
            raise ValueError("'useDefaultSkips' must be a boolean.")

        return cls(
            paths=paths,
            name=name,
            extensions=extensions,
            content=content,
            regex=regex,
            case_sensitive=case_sensitive,
            exclude=exclude,
            max_depth=max_depth,
            context=context,
            limit=limit,
            mode=mode,
            use_default_skips=use_default_skips,
        )

    def to_arguments(self) -> dict[str, Any]:
        """Serialize this input for the fixed worker wire protocol."""
        result: dict[str, Any] = {"paths": list(self.paths)}
        if self.name is not None:
            result["name"] = list(self.name)
        if self.extensions is not None:
            result["extensions"] = list(self.extensions)
        if self.content is not None:
            result["content"] = self.content
        if self.regex:
            result["regex"] = True
        if not self.case_sensitive:
            result["caseSensitive"] = False
        if self.exclude is not None:
            result["exclude"] = list(self.exclude)
        if self.max_depth is not None:
            result["maxDepth"] = self.max_depth
        if self.context:
            result["context"] = self.context
        if self.limit != DEFAULT_LIMIT:
            result["limit"] = self.limit
        if self.mode != "files":
            result["mode"] = self.mode
        # Only surface the opt-out so the wire-protocol default stays "on"
        # for existing callers that omit ``useDefaultSkips``.
        if not self.use_default_skips:
            result["useDefaultSkips"] = False
        return result


def _effective_exclude_patterns(input: FindInput) -> tuple[str, ...]:
    """Return the exclude patterns actually applied during the walk.

    When ``use_default_skips`` is true (the default), the canonical
    repository-junk set from :data:`DEFAULT_TREE_SKIPS` is unioned with any
    model-supplied ``exclude`` entries so:

    * callers that omit ``exclude`` still get the default junk prune
      (mirroring :mod:`tree` and :mod:`glob`);
    * callers that pass an explicit ``exclude`` get a union of their entries
      with the defaults — their patterns stack on top, never replace them;
    * passing ``exclude=[]`` does **not** silently disable the defaults,
      because the union is computed from the defaults plus the explicit
      empty tuple.
    """
    patterns: set[str] = set()
    if input.use_default_skips:
        patterns.update(DEFAULT_TREE_SKIPS)
    if input.exclude:
        patterns.update(input.exclude)
    return tuple(patterns)


def _scan_paths_for_files(
    input: FindInput,
    fs: ScopedFilesystem,
) -> list[Path]:
    """Resolve each root path and walk it with depth/exclude filtering."""
    seen: set[str] = set()
    resolved: list[Path] = []
    effective_exclude = _effective_exclude_patterns(input)
    for raw in input.paths:
        try:
            base = fs.require_allowed_path(fs.resolve_path(raw))
        except ValueError:
            continue
        try:
            base = base.resolve()
        except OSError:
            continue
        base_key = str(base)
        if base_key in seen:
            continue
        seen.add(base_key)

        if not base.exists():
            continue

        if base.is_file():
            resolved.append(base)
            continue

        if not base.is_dir():
            continue

        for root_str, directories, files in os.walk(str(base), followlinks=False):
            root_path = Path(root_str)
            relative = (
                root_path.resolve().relative_to(base).as_posix()
                if (root_path.resolve() != base)
                else ""
            )
            depth = 0 if not relative else relative.count(os.sep) + 1
            if input.max_depth is not None and depth >= input.max_depth:
                # Stop descending once the cap is reached: prune all child
                # directories and skip the current directory's files.
                directories[:] = []
                files = []
                continue

            pruned: list[str] = []
            for directory in directories:
                directory_path = root_path / directory
                if input.max_depth is not None and (
                    depth + 1
                ) >= input.max_depth:
                    pruned.append(directory)
                    continue
                if _is_excluded(directory_path, base, effective_exclude):
                    pruned.append(directory)
                    continue
            directories[:] = [d for d in directories if d not in pruned]

            for filename in files:
                candidate = root_path / filename
                try:
                    candidate = candidate.resolve()
                except OSError:
                    continue
                if not candidate.is_file():
                    continue
                if input.name and not _path_matches_any_name(
                    input.name, candidate, base
                ):
                    continue
                if input.extensions and not _path_matches_any_extension(
                    input.extensions, candidate
                ):
                    continue
                resolved.append(candidate)
                if len(resolved) >= input.limit * 4 + 50:
                    # Early-exit: collect a small overshoot so truncation
                    # detection works without scanning everything.
                    return resolved
    return resolved


def _filter_paths_for_match_mode(
    paths: list[Path],
    fs: ScopedFilesystem,
) -> list[Path]:
    """Re-filter an already-walked list into one entry per file.

    Caller-side filters (``name``, ``extensions``, ``exclude``, ``maxDepth``)
    have already been applied during :func:`_scan_paths_for_files`. This pass
    only enforces sandbox visibility, regular-file status, and the binary-skip
    size cap so we do not open ``find`` content matches against oversized or
    non-text payloads.
    """
    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in paths:
        try:
            allowed = fs.require_allowed_path(candidate)
        except ValueError:
            continue
        if not allowed.is_file():
            continue
        try:
            stat_result = allowed.stat()
        except OSError:
            continue
        if stat_result.st_size > MAX_FILE_BYTES:
            continue
        key = str(allowed.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(allowed)
    return unique


def _content_matches_for_file(
    allowed: Path,
    input: FindInput,
) -> tuple[FindMatch, ...]:
    """Return the ``FindMatch`` instances for one candidate file.

    The caller has already resolved ``display`` via ``fs.display_path``;
    this helper only needs the file path and the input parameters.
    """
    try:
        with allowed.open("rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
    except OSError:
        return ()
    if len(raw) > MAX_FILE_BYTES:
        return ()
    if b"\x00" in raw[:8192]:
        return ()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ()

    lines = text.splitlines()
    if not lines:
        return ()

    if input.regex:
        expression = _compile_content_expression(
            input.content or "",
            regex=True,
            case_sensitive=input.case_sensitive,
        )
    else:
        # Literal mode escapes ``content`` so we can rely on the same regex
        # code path with consistent invalid-pattern reporting.
        expression = _compile_content_expression(
            input.content or "",
            regex=False,
            case_sensitive=input.case_sensitive,
        )

    matches: list[FindMatch] = []
    context = input.context
    line_total = len(lines)
    for index, line in enumerate(lines):
        if not expression.search(line):
            continue
        line_number = index + 1
        if context:
            start = max(0, index - context)
            end = min(line_total, index + context + 1)
            window = lines[start:end]
        else:
            window = [line]
        truncated_text = line
        if len(truncated_text) > MAX_LINE_CHARS:
            truncated_text = truncated_text[:MAX_LINE_CHARS] + "..."
        matches.append(
            FindMatch(
                line=line_number,
                text=truncated_text,
                context=tuple(window),
            )
        )
    return tuple(matches)


def execute(input: FindInput, fs: ScopedFilesystem) -> FindOutput:
    """Execute the combined search operation.

    The implementation follows the staged pipeline described in the spec:

    1. resolve and dedupe ``paths``;
    2. walk each root while honouring ``exclude`` / ``useDefaultSkips`` /
       ``maxDepth``;
    3. apply ``name`` and ``extensions`` filters during the walk;
    4. when ``content`` is provided, run the per-file regex search;
    5. stop early once ``limit`` is reached and set ``truncated``;
    6. render to either ``paths`` (``mode="files"``), ``results``
       (``mode="matches"``), or ``counts`` (``mode="count"``).
    """
    candidates = _scan_paths_for_files(input, fs)

    if input.content is None:
        seen: set[str] = set()
        unique_paths: list[str] = []
        truncated = False
        for candidate in candidates:
            try:
                allowed = fs.require_allowed_path(candidate)
            except ValueError:
                continue
            display = fs.display_path(allowed)
            if display in seen:
                continue
            seen.add(display)
            unique_paths.append(display)
            if len(unique_paths) > input.limit:
                truncated = True
                unique_paths = unique_paths[: input.limit]
                break

        if input.mode == "matches":
            empty_results: list[FindFileMatches] = [
                FindFileMatches(path=display, matches=()) for display in unique_paths
            ]
            return FindOutput(
                mode="matches",
                results=tuple(empty_results),
                truncated=truncated,
            )

        # ``mode='count'`` with no ``content`` is rejected at parse time,
        # but defensive guard keeps the type-narrowing honest.
        if input.mode == "count":
            return FindOutput(
                mode="count",
                counts=(),
                truncated=truncated,
            )

        return FindOutput(
            mode="files",
            paths=tuple(unique_paths),
            truncated=truncated,
        )

    seen_paths: set[str] = set()
    unique_files = _filter_paths_for_match_mode(candidates, fs)

    file_results: list[FindFileMatches] = []
    truncated = False
    emitted = 0
    for allowed in unique_files:
        display = fs.display_path(allowed)
        if display in seen_paths:
            continue
        seen_paths.add(display)
        matches = _content_matches_for_file(allowed, input)
        if not matches:
            continue
        file_results.append(FindFileMatches(path=display, matches=matches))
        emitted += len(matches)
        if emitted >= input.limit:
            truncated = True
            break

    # In ``matches`` / ``count`` modes the ``limit`` counts total matches;
    # once we have reached it we may still have an overshoot in the current
    # file. Trim the last file's matches if necessary so the count is at
    # most ``limit``.
    if truncated and file_results:
        kept: list[FindFileMatches] = []
        running = 0
        for entry in file_results:
            available = input.limit - running
            if available <= 0:
                break
            if len(entry.matches) <= available:
                kept.append(entry)
                running += len(entry.matches)
            else:
                kept.append(
                    FindFileMatches(
                        path=entry.path,
                        matches=entry.matches[:available],
                    )
                )
                running += available
        file_results = kept

    if input.mode == "files":
        unique_paths = [
            entry.path for entry in file_results if entry.matches
        ]
        truncated = truncated or len(unique_paths) > input.limit
        unique_paths = unique_paths[: input.limit]
        return FindOutput(
            mode="files",
            paths=tuple(unique_paths),
            truncated=truncated,
        )

    if input.mode == "count":
        # Per-file totals; emit one ``(path, count)`` entry per file with
        # at least one content match. ``limit`` bounds the *number of
        # files* we surface (not the raw match totals), mirroring the
        # grep ``count`` mode semantics where ``max_results`` caps the
        # number of file rows.
        count_rows: list[tuple[str, int]] = [
            (entry.path, len(entry.matches)) for entry in file_results
        ]
        count_truncated = truncated or len(count_rows) > input.limit
        count_rows = count_rows[: input.limit]
        return FindOutput(
            mode="count",
            counts=tuple(count_rows),
            truncated=count_truncated,
        )

    return FindOutput(
        mode="matches",
        results=tuple(file_results),
        truncated=truncated,
    )


# Silence "imported but unused" linters for module-private types re-exported
# below; they are part of the public payload surface.
_ = field
