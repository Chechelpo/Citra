"""Aider-style structural repository maps backed by grep-ast/tree-sitter."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import TYPE_CHECKING, Any, Iterable

from citra.utils.model_tokenizer import tokenize

if TYPE_CHECKING:
    from citra.context.session_context import WorkspaceContext


DEFAULT_MAP_TOKENS = 3_500
MAX_MAP_TOKENS = 6_000
MAX_SOURCE_FILE_BYTES = 1_000_000
_MAX_RENDERED_LINE_LENGTH = 160

_SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
        "venv",
    }
)

# Tree-sitter grammars use different node names, but the structural concepts
# are remarkably consistent. grep-ast supplies the parsers; these names only
# identify which AST nodes are useful repository-map definitions.
_DEFINITION_NODE_TYPES = frozenset(
    {
        "class_definition",
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "enum_item",
        "struct_item",
        "struct_specifier",
        "trait_item",
        "impl_item",
        "function_definition",
        "function_declaration",
        "function_item",
        "method_definition",
        "method_declaration",
        "constructor_declaration",
        "generator_function_declaration",
        "type_alias_declaration",
        "type_definition",
        "type_spec",
        "type_item",
        "module_declaration",
        "namespace_definition",
        "namespace_declaration",
        "mod_item",
        "macro_definition",
        "variable_declarator",
        "var_spec",
        "const_spec",
        "const_item",
        "static_item",
    }
)

_REFERENCE_NODE_TYPES = frozenset(
    {
        "identifier",
        "type_identifier",
        "field_identifier",
        "property_identifier",
        "namespace_identifier",
        "module_name",
    }
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$-]*$")


@dataclass(frozen=True)
class Definition:
    path: str
    line: int
    name: str


@dataclass(frozen=True)
class FileIndex:
    mtime_ns: int
    size: int
    definitions: tuple[Definition, ...]
    references: tuple[tuple[str, int], ...]

    def reference_counter(self) -> Counter[str]:
        return Counter(dict(self.references))


@dataclass(frozen=True)
class _Edge:
    source: str
    destination: str
    identifier: str
    weight: float


class RepoMap:
    """
    Build a compact structural map of the effective Citra project.

    Files are indexed from the current project. Parsed tags are cached by
    path/mtime/size for the ExecutionContext lifetime.
    """

    def __init__(self, workspace: WorkspaceContext) -> None:
        self.workspace = workspace
        self._index_cache: dict[Path, FileIndex] = {}

    def render(
        self,
        *,
        model_id: str,
        path: str = ".",
        focus: Iterable[str] = (),
        max_tokens: int = DEFAULT_MAP_TOKENS,
    ) -> str:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if max_tokens > MAX_MAP_TOKENS:
            raise ValueError(
                f"max_tokens cannot exceed {MAX_MAP_TOKENS}"
            )

        tmp_subtree = self._normalize_tmp_subtree(path)
        if tmp_subtree is not None:
            effective_files = self._tmp_files(tmp_subtree)
        else:
            subtree = self._normalize_subtree(path)
            effective_files = self._effective_files(subtree)

        if not effective_files:
            return "No source files found. Use glob for raw path discovery."

        indexes: dict[str, FileIndex] = {}
        physical_paths: dict[str, Path] = {}

        for relative, physical in effective_files.items():
            index = self._index_file(relative, physical)
            if index is None:
                continue
            indexes[relative] = index
            physical_paths[relative] = physical

        if not indexes:
            return (
                "No tree-sitter-supported source files found. "
                "Use glob for raw path discovery."
            )

        focus_terms = {
            term.strip()
            for term in focus
            if isinstance(term, str) and term.strip()
        }
        ranked = self._rank_definitions(indexes, focus_terms)

        if not ranked:
            return self._fallback_file_map(
                indexes,
                model_id=model_id,
                max_tokens=max_tokens,
            )

        return self._fit_ranked_map(
            ranked,
            physical_paths,
            model_id=model_id,
            max_tokens=max_tokens,
        )

    def _normalize_tmp_subtree(
        self,
        raw: str,
    ) -> PurePosixPath | None:
        value = raw.strip() or "."
        if value == "@tmp":
            return PurePosixPath(".")
        if not value.startswith("@tmp/"):
            return None

        candidate = PurePosixPath(
            value[len("@tmp/"):] or "."
        )
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(
                "tree @tmp path must stay within @tmp"
            )
        return candidate

    def _normalize_subtree(self, raw: str) -> PurePosixPath:
        value = raw.strip() or "."

        candidate = PurePosixPath(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(
                "tree path must be a project-relative subtree"
            )
        return candidate

    def _tmp_files(
        self,
        subtree: PurePosixPath,
    ) -> dict[str, Path]:
        root = self.workspace.tmp
        target = (root / subtree.as_posix()).resolve()

        try:
            target.relative_to(root.resolve())
        except (OSError, ValueError):
            raise ValueError(
                "tree @tmp path must stay within @tmp"
            ) from None

        if target.is_file():
            if not self._safe_regular_file(target, root):
                return {}
            relative = target.relative_to(root.resolve()).as_posix()
            if self._skip_relative(relative):
                return {}
            return {f"@tmp/{relative}": target}

        if not target.is_dir():
            return {}

        files: dict[str, Path] = {}
        for directory, dirnames, filenames in os.walk(target):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in _SKIP_DIRECTORIES
            ]
            base = Path(directory)
            for filename in filenames:
                physical = base / filename
                if not self._safe_regular_file(physical, root):
                    continue
                try:
                    relative = physical.relative_to(root).as_posix()
                except ValueError:
                    continue
                if self._skip_relative(relative):
                    continue
                files[f"@tmp/{relative}"] = physical

        return dict(sorted(files.items()))

    def _effective_files(
        self,
        subtree: PurePosixPath,
    ) -> dict[str, Path]:
        files: dict[str, Path] = {}
        subtree_text = subtree.as_posix()
        if subtree_text == ".":
            subtree_text = ""

        for relative, physical in self._walk_workspace_files():
            if not self._within_subtree(relative, subtree_text):
                continue
            files[relative] = physical

        return dict(sorted(files.items()))

    @staticmethod
    def _within_subtree(relative: str, subtree: str) -> bool:
        if not subtree:
            return True
        return relative == subtree or relative.startswith(subtree + "/")

    def _walk_workspace_files(self) -> Iterable[tuple[str, Path]]:
        root = self.workspace.workspace
        for directory, dirnames, filenames in os.walk(root):
            base = Path(directory)
            dirnames[:] = [
                name
                for name in dirnames
                if name not in _SKIP_DIRECTORIES
                and not self.workspace.is_controller_private_source_path(
                    base / name
                )
            ]
            for filename in filenames:
                path = base / filename
                if self.workspace.is_controller_private_source_path(path):
                    continue
                try:
                    relative = path.relative_to(root).as_posix()
                except ValueError:
                    continue
                if self._skip_relative(relative):
                    continue
                if self._safe_regular_file(path, root):
                    yield relative, path

    @staticmethod
    def _skip_relative(relative: str) -> bool:
        parts = PurePosixPath(relative).parts
        return any(part in _SKIP_DIRECTORIES for part in parts)

    @staticmethod
    def _safe_regular_file(path: Path, root: Path) -> bool:
        try:
            resolved = path.resolve()
            resolved.relative_to(root.resolve())
            return resolved.is_file()
        except (OSError, ValueError):
            return False

    def _index_file(
        self,
        relative: str,
        physical: Path,
    ) -> FileIndex | None:
        filename_to_lang, _tree_context, get_parser = self._grep_ast()
        lang = filename_to_lang(relative)

        # grep-ast 0.8.1 currently maps TSX to the TypeScript grammar even
        # though tree-sitter-language-pack exposes a dedicated TSX grammar.
        # Prefer the dedicated grammar when available.
        if physical.suffix.lower() == ".tsx":
            lang = "tsx"

        if not lang:
            return None

        try:
            metadata = physical.stat()
        except OSError:
            return None

        if metadata.st_size > MAX_SOURCE_FILE_BYTES:
            return None

        cached = self._index_cache.get(physical)
        if (
            cached is not None
            and cached.mtime_ns == metadata.st_mtime_ns
            and cached.size == metadata.st_size
        ):
            return cached

        try:
            raw = physical.read_bytes()
            parser = get_parser(lang)
            tree = parser.parse(raw)
        except (OSError, UnicodeError, LookupError, RuntimeError):
            return None
        except Exception:
            # Individual third-party grammars can fail independently. A repo
            # map should degrade by skipping that file rather than failing the
            # entire model tool call.
            return None

        definitions, references = self._extract_tags(
            tree.root_node,
            raw,
            relative,
        )

        index = FileIndex(
            mtime_ns=metadata.st_mtime_ns,
            size=metadata.st_size,
            definitions=tuple(definitions),
            references=tuple(sorted(references.items())),
        )
        self._index_cache[physical] = index
        return index

    def _extract_tags(
        self,
        root: Any,
        source: bytes,
        relative: str,
    ) -> tuple[list[Definition], Counter[str]]:
        definitions: list[Definition] = []
        references: Counter[str] = Counter()
        definition_ranges: set[tuple[int, int]] = set()

        stack = [root]
        while stack:
            node = stack.pop()
            children = list(getattr(node, "named_children", ()) or ())
            stack.extend(reversed(children))

            if self._is_definition_node(getattr(node, "type", "")):
                name_node = self._definition_name_node(node)
                if name_node is not None:
                    name = self._node_identifier(name_node, source)
                    if name is not None:
                        definitions.append(
                            Definition(
                                path=relative,
                                line=int(name_node.start_point[0]),
                                name=name,
                            )
                        )
                        definition_ranges.add(
                            (name_node.start_byte, name_node.end_byte)
                        )

            if (
                getattr(node, "type", "") in _REFERENCE_NODE_TYPES
                and not children
            ):
                node_range = (node.start_byte, node.end_byte)
                if node_range in definition_ranges:
                    continue
                name = self._node_identifier(node, source)
                if name is not None:
                    references[name] += 1

        return definitions, references

    @staticmethod
    def _is_definition_node(node_type: str) -> bool:
        if node_type in _DEFINITION_NODE_TYPES:
            return True
        if not (
            node_type.endswith("_definition")
            or node_type.endswith("_declaration")
        ):
            return False
        return any(
            marker in node_type
            for marker in (
                "class",
                "enum",
                "function",
                "interface",
                "method",
                "module",
                "namespace",
                "struct",
                "trait",
                "type",
            )
        )

    def _definition_name_node(self, node: Any) -> Any | None:
        for field in ("name", "declarator", "pattern", "left"):
            try:
                candidate = node.child_by_field_name(field)
            except Exception:
                candidate = None
            if candidate is None:
                continue
            identifier = self._first_identifier_node(candidate)
            if identifier is not None:
                return identifier
        return self._first_identifier_node(node)

    def _first_identifier_node(self, node: Any) -> Any | None:
        if getattr(node, "type", "") in _REFERENCE_NODE_TYPES:
            return node
        for child in getattr(node, "named_children", ()) or ():
            result = self._first_identifier_node(child)
            if result is not None:
                return result
        return None

    @staticmethod
    def _node_identifier(node: Any, source: bytes) -> str | None:
        try:
            value = source[node.start_byte:node.end_byte].decode("utf-8")
        except (AttributeError, UnicodeDecodeError):
            return None
        value = value.strip()
        if len(value) < 2 or len(value) > 128:
            return None
        if not _IDENTIFIER_RE.fullmatch(value):
            return None
        return value

    def _rank_definitions(
        self,
        indexes: dict[str, FileIndex],
        focus: set[str],
    ) -> list[Definition]:
        defines: dict[str, set[str]] = defaultdict(set)
        definitions: dict[tuple[str, str], list[Definition]] = defaultdict(list)
        references: dict[str, Counter[str]] = defaultdict(Counter)

        for path, index in indexes.items():
            for definition in index.definitions:
                defines[definition.name].add(path)
                definitions[(path, definition.name)].append(definition)
            for identifier, count in index.references:
                references[identifier][path] += count

        if not defines:
            return []

        edges: list[_Edge] = []
        identifiers = set(defines).intersection(references)

        # Match Aider's useful weighting heuristics: distinctive identifiers
        # and explicitly-focused symbols carry more graph weight, private and
        # overly-ambiguous names carry less, and repeated references are sqrt
        # damped so generated/high-frequency code does not dominate.
        for identifier in identifiers:
            definers = defines[identifier]
            multiplier = 1.0
            is_snake = "_" in identifier and any(ch.isalpha() for ch in identifier)
            is_kebab = "-" in identifier and any(ch.isalpha() for ch in identifier)
            is_camel = any(ch.isupper() for ch in identifier) and any(
                ch.islower() for ch in identifier
            )

            if identifier in focus:
                multiplier *= 10.0
            if (is_snake or is_kebab or is_camel) and len(identifier) >= 8:
                multiplier *= 10.0
            if identifier.startswith("_"):
                multiplier *= 0.1
            if len(definers) > 5:
                multiplier *= 0.1

            for referencer, count in references[identifier].items():
                for definer in definers:
                    weight = multiplier * math.sqrt(count)
                    if self._path_matches_focus(referencer, focus):
                        weight *= 50.0
                    edges.append(
                        _Edge(
                            source=referencer,
                            destination=definer,
                            identifier=identifier,
                            weight=weight,
                        )
                    )

        referenced_identifiers = set(references)
        for identifier, definers in defines.items():
            if identifier in referenced_identifiers:
                continue
            for definer in definers:
                edges.append(
                    _Edge(
                        source=definer,
                        destination=definer,
                        identifier=identifier,
                        weight=0.1,
                    )
                )

        nodes = set(indexes)
        personalization = {
            path: 1.0
            for path in nodes
            if self._path_matches_focus(path, focus)
        }
        ranks = self._pagerank(nodes, edges, personalization)

        ranked_definitions: Counter[tuple[str, str]] = Counter()
        outgoing: dict[str, list[_Edge]] = defaultdict(list)
        for edge in edges:
            outgoing[edge.source].append(edge)

        for source, source_edges in outgoing.items():
            total = sum(edge.weight for edge in source_edges)
            if total <= 0:
                continue
            source_rank = ranks.get(source, 0.0)
            for edge in source_edges:
                ranked_definitions.update(
                    {
                        (edge.destination, edge.identifier): (
                            source_rank * edge.weight / total
                        )
                    }
                )

        result: list[Definition] = []
        for key, _score in ranked_definitions.most_common():
            result.extend(definitions.get(key, ()))

        # Ensure definitions in disconnected files can still enter the map.
        included = {(item.path, item.line, item.name) for item in result}
        remaining = [
            definition
            for index in indexes.values()
            for definition in index.definitions
            if (definition.path, definition.line, definition.name) not in included
        ]
        remaining.sort(
            key=lambda item: (
                not self._path_matches_focus(item.path, focus),
                item.name not in focus,
                item.path,
                item.line,
            )
        )
        result.extend(remaining)
        return result

    @staticmethod
    def _path_matches_focus(path: str, focus: set[str]) -> bool:
        if not focus:
            return False
        pure = PurePosixPath(path)
        components = set(pure.parts)
        components.add(pure.name)
        components.add(pure.stem)
        return bool(components.intersection(focus)) or any(
            term in path for term in focus if "/" in term
        )

    @staticmethod
    def _pagerank(
        nodes: set[str],
        edges: list[_Edge],
        personalization: dict[str, float],
        *,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
    ) -> dict[str, float]:
        if not nodes:
            return {}

        ordered = tuple(sorted(nodes))
        size = len(ordered)
        uniform = 1.0 / size
        rank = {node: uniform for node in ordered}

        if personalization:
            total_personalization = sum(personalization.values())
            teleport = {
                node: personalization.get(node, 0.0) / total_personalization
                for node in ordered
            }
        else:
            teleport = {node: uniform for node in ordered}

        outgoing: dict[str, list[tuple[str, float]]] = defaultdict(list)
        totals: Counter[str] = Counter()
        for edge in edges:
            if edge.weight <= 0:
                continue
            outgoing[edge.source].append((edge.destination, edge.weight))
            totals.update({edge.source: edge.weight})

        for _ in range(max_iterations):
            next_rank = {
                node: (1.0 - damping) * teleport[node]
                for node in ordered
            }
            dangling = sum(
                rank[node]
                for node in ordered
                if totals[node] <= 0
            )
            if dangling:
                for node in ordered:
                    next_rank[node] += damping * dangling * teleport[node]

            for source, destinations in outgoing.items():
                total = totals[source]
                contribution = damping * rank[source] / total
                for destination, weight in destinations:
                    next_rank[destination] += contribution * weight

            delta = sum(
                abs(next_rank[node] - rank[node])
                for node in ordered
            )
            rank = next_rank
            if delta <= tolerance:
                break

        return rank

    def _fit_ranked_map(
        self,
        ranked: list[Definition],
        physical_paths: dict[str, Path],
        *,
        model_id: str,
        max_tokens: int,
    ) -> str:
        low = 1
        high = len(ranked)
        best = ""

        while low <= high:
            middle = (low + high) // 2
            rendered = self._render_definitions(
                ranked[:middle],
                physical_paths,
            )
            count = tokenize(model_id, rendered)
            if count <= max_tokens:
                best = rendered
                low = middle + 1
            else:
                high = middle - 1

        if best:
            return best.rstrip()

        # A single TreeContext chunk can exceed a very small budget. Return a
        # minimal location instead of letting the generic Tool truncator cut a
        # structural block in half.
        first = ranked[0]
        fallback = f"{first.path}:{first.line + 1}: {first.name}"
        return fallback

    def _render_definitions(
        self,
        definitions: list[Definition],
        physical_paths: dict[str, Path],
    ) -> str:
        _filename_to_lang, TreeContext, _get_parser = self._grep_ast()
        lines_by_file: dict[str, set[int]] = defaultdict(set)
        for definition in definitions:
            lines_by_file[definition.path].add(definition.line)

        parts: list[str] = []
        for relative in sorted(lines_by_file):
            physical = physical_paths.get(relative)
            if physical is None:
                continue
            try:
                code = physical.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError:
                continue
            if not code.endswith("\n"):
                code += "\n"

            try:
                context = TreeContext(
                    relative,
                    code,
                    color=False,
                    line_number=False,
                    child_context=False,
                    last_line=False,
                    margin=0,
                    mark_lois=False,
                    loi_pad=0,
                    show_top_of_file_parent_scope=False,
                )
                context.lines_of_interest = set()
                context.add_lines_of_interest(lines_by_file[relative])
                context.add_context()
                rendered = context.format().rstrip()
            except Exception:
                rendered = self._render_source_lines(
                    code,
                    lines_by_file[relative],
                )

            rendered = "\n".join(
                line[:_MAX_RENDERED_LINE_LENGTH]
                for line in rendered.splitlines()
            )
            parts.append(f"{relative}:\n{rendered}")

        return "\n\n".join(parts) + ("\n" if parts else "")

    @staticmethod
    def _render_source_lines(code: str, lines: set[int]) -> str:
        source_lines = code.splitlines()
        result: list[str] = []
        for line in sorted(lines):
            if 0 <= line < len(source_lines):
                result.append(source_lines[line])
        return "\n".join(result)

    @staticmethod
    def _fallback_file_map(
        indexes: dict[str, FileIndex],
        *,
        model_id: str,
        max_tokens: int,
    ) -> str:
        result: list[str] = []
        for path in sorted(indexes):
            candidate = "\n".join((*result, path))
            if tokenize(model_id, candidate) > max_tokens:
                break
            result.append(path)
        return "\n".join(result)

    @staticmethod
    def _grep_ast() -> tuple[Any, Any, Any]:
        try:
            from grep_ast import TreeContext, filename_to_lang
            from grep_ast.tsl import get_parser
        except ImportError as error:
            raise RuntimeError(
                "Repository maps require the Aider grep-ast dependency. "
                "Add 'grep-ast>=0.8.1' to Citra's Python dependencies."
            ) from error
        return filename_to_lang, TreeContext, get_parser
