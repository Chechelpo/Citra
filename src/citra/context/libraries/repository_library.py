from __future__ import annotations

import hashlib
import json
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol
from urllib.parse import urlparse
from ...utils.git_utility import GitRepositoryUtility,ResolvedGitRevision

_SEPARATOR = "// " + "~" * 72
_GITHUB_SCP_RE = re.compile(
    r"^(?:git@)?github\.com:(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


@dataclass(frozen=True)
class RepoFileEdit:
    """
    Represents one line-based edit over an existing documentation file.

        1. path: relative path to the file inside the documented repository.
        2. content: new content to insert.
        3. at_line: first line affected by the edit. Lines are 1-indexed.
        4. to_line:
            - None: inserts content at at_line without removing following lines.
            - int: replaces the inclusive range [at_line, to_line] with content.
    """

    path: Path
    content: str
    at_line: int
    to_line: int | None = None


@dataclass(frozen=True)
class DocumentedRepository:
    """
    Represents the documentation of ONE repository at ONE specific commit.

    Extracted from repo.toml and revision.toml while scanning the library.

    A documented repository follows the structure:

        ./.citra/library/repos/<name>-<hash>/
                                |- repo.toml
                                |
                                |- commits/
                                    |- <commit1>/
                                        |- revision.toml
                                        |- index.md
                                        |
                                        |- <folder1>/
                                            |- index.md
                                            |- <doc_name>.md
                                            |- <doc_name1>.md
                                            ...
                                        |
                                        |- <folder2>/
                                            |- index.md
                                            ...
                                    |
                                    |- <commit2>/
                                        |- revision.toml
                                        |- index.md
                                        ...

    repo.toml holds information identifying the repository itself:
        - url
        - name
        - author
        - preferred_commit

    revision.toml holds information regarding this documentation snapshot:
        - branch
        - commit

    root points directly to:

        ./.citra/library/repos/<name>-<hash>/commits/<commit>/

    repo_root points to:

        ./.citra/library/repos/<name>-<hash>/

    All paths passed to document operations MUST be relative to root.

    A DocumentedRepository never accesses files outside root.
    """

    url: str
    repo_root: Path
    root: Path

    name: str
    author: str

    branch: str
    commit: str

    _PROTECTED_NAMES: ClassVar[frozenset[str]] = frozenset({"revision.toml"})

    def tree_docs(self, path: Path | None = None) -> str:
        """
        Returns the documentation tree of this repository.

            1. path:
                - None: returns the complete tree from this documentation root.
                - Path: returns the tree starting from this RELATIVE directory.

        Only documentation directories and Markdown files are shown.
        revision.toml is intentionally hidden.

        Returns either:
            1. A structured tree.
            2. A diagnostic string if path does not exist or is invalid.
        """
        relative = Path(".") if path is None else Path(path)

        try:
            target = self._resolve_path(relative)
        except ValueError as exc:
            return f"Invalid documentation path '{relative}': {exc}"

        if not target.exists():
            return f"Documentation path not found: {relative.as_posix()}"

        if target.is_file():
            return (
                f"Documentation path is a file: {relative.as_posix()}\n"
                "Use view_file() to read it."
            )

        label = "." if relative == Path(".") else relative.as_posix()
        lines = [label]
        self._append_tree(target, lines, prefix="")
        return "\n".join(lines)

    def view_file(
        self,
        path: Path,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """
        Returns an existing documentation file.

            1. path:
                Relative path inside this repository's documentation root.

            2. start_line:
                Optional first line to return. Lines are 1-indexed.

            3. end_line:
                Optional final line to return, inclusive.

        If no line range is specified, returns the entire file.

        If path points to a directory, returns a diagnostic containing the
        directory tree instead.
        """
        relative = Path(path)
        try:
            target = self._resolve_doc_file(relative, must_exist=False)
        except ValueError as exc:
            return f"Invalid documentation path '{relative}': {exc}"

        if not target.exists():
            return self._file_not_found(relative)

        if target.is_dir():
            return (
                f"Documentation path is a directory: {relative.as_posix()}\n\n"
                f"{self.tree_docs(relative)}"
            )

        if not target.is_file():
            return f"Documentation path is not a regular file: {relative.as_posix()}"

        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            return f"Failed to read {relative.as_posix()}: {exc}"

        lines = text.splitlines(keepends=True)
        total_lines = len(lines)

        if total_lines == 0:
            if start_line not in (None, 1) or end_line not in (None, 0, 1):
                return f"File is empty: {relative.as_posix()}"
            return self._format_file(relative, "", 0, 0)

        start = 1 if start_line is None else start_line
        end = total_lines if end_line is None else end_line

        if start < 1:
            return f"start_line must be >= 1; got {start}."
        if end < start:
            return f"end_line must be >= start_line; got {end} < {start}."
        if start > total_lines:
            return (
                f"start_line {start} is beyond the end of {relative.as_posix()} "
                f"({total_lines} lines)."
            )

        end = min(end, total_lines)
        content = "".join(lines[start - 1 : end])
        return self._format_file(relative, content, start, end)

    def search_docs(
        self,
        query: str,
        path: Path | None = None,
        max_results: int = 50,
        case_sensitive: bool = False,
    ) -> str:
        """
        Searches text across this repository's documentation.

            1. query: text to search for.
            2. path: optional RELATIVE file/directory search scope.
            3. max_results: maximum number of matching lines returned.
            4. case_sensitive: whether matching should preserve case.

        Each match contains:
            - Relative file path.
            - Line number.
            - Matching line.

        This searches documentation ONLY. It never searches source code.
        """
        if not query:
            return "Search query cannot be empty."
        if max_results < 1:
            return f"max_results must be >= 1; got {max_results}."

        relative = Path(".") if path is None else Path(path)
        try:
            target = self._resolve_path(relative)
        except ValueError as exc:
            return f"Invalid documentation path '{relative}': {exc}"

        if not target.exists():
            return f"Documentation path not found: {relative.as_posix()}"

        candidates: list[Path]
        if target.is_file():
            if target.suffix.lower() != ".md":
                return f"Not a Markdown documentation file: {relative.as_posix()}"
            candidates = [target]
        else:
            candidates = sorted(
                (
                    candidate
                    for candidate in target.rglob("*.md")
                    if candidate.is_file() and self._is_inside_root(candidate)
                ),
                key=lambda candidate: candidate.relative_to(self.root).as_posix(),
            )

        needle = query if case_sensitive else query.casefold()
        matches: list[str] = []
        truncated = False

        for candidate in candidates:
            try:
                candidate_lines = candidate.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue

            relative_file = candidate.relative_to(self.root).as_posix()
            for line_number, line in enumerate(candidate_lines, start=1):
                haystack = line if case_sensitive else line.casefold()
                if needle in haystack:
                    matches.append(f"{relative_file}:{line_number}: {line}")
                    if len(matches) >= max_results:
                        truncated = True
                        break
            if truncated:
                break

        if not matches:
            return f"No documentation matches for: {query}"

        header = f"Matches for '{query}' ({len(matches)} returned)"
        if truncated:
            header += f" [limited to {max_results}]"
        return header + "\n" + "\n".join(matches)

    def add_file(self, path: Path, content: str) -> str:
        """
        Creates a new Markdown document inside this repository's documentation.

        Parent directories are created automatically. Existing files are never
        silently overwritten.
        """
        relative = Path(path)
        try:
            target = self._resolve_doc_file(relative, must_exist=False)
            if target.exists():
                return f"File already exists: {relative.as_posix()}"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8", newline="") as file:
                file.write(content)
        except (ValueError, OSError) as exc:
            return f"Failed to create {relative.as_posix()}: {exc}"

        return f"Created documentation file: {relative.as_posix()}"

    def replace_file(self, path: Path, content: str) -> str:
        """
        Completely replaces the contents of an EXISTING Markdown document.
        """
        relative = Path(path)
        try:
            target = self._resolve_doc_file(relative, must_exist=True)
            if not target.is_file():
                return f"Documentation path is not a file: {relative.as_posix()}"
            self._write_text_atomic(target, content)
        except (ValueError, OSError) as exc:
            return f"Failed to replace {relative.as_posix()}: {exc}"

        return f"Replaced documentation file: {relative.as_posix()}"

    def edit_file(
        self,
        path: Path,
        content: str,
        at_line: int,
        to_line: int | None = None,
    ) -> str:
        """
        Edits an EXISTING documentation file using a line range.

            - to_line=None inserts content before at_line.
            - to_line=int replaces inclusive range [at_line, to_line].
            - at_line == line_count + 1 is allowed for insertion at EOF.
        """
        relative = Path(path)
        try:
            target = self._resolve_doc_file(relative, must_exist=True)
            if not target.is_file():
                return f"Documentation path is not a file: {relative.as_posix()}"

            original = target.read_text(encoding="utf-8")
            lines = original.splitlines(keepends=True)
            line_count = len(lines)

            if at_line < 1:
                return f"at_line must be >= 1; got {at_line}."

            if to_line is None:
                if at_line > line_count + 1:
                    return (
                        f"at_line {at_line} is beyond the valid insertion range "
                        f"1..{line_count + 1}."
                    )
                insertion = self._prepare_splice_content(
                    content,
                    followed_by_existing_line=at_line <= line_count,
                )
                updated = lines[: at_line - 1] + insertion + lines[at_line - 1 :]
            else:
                if line_count == 0:
                    return f"Cannot replace a line range in empty file: {relative.as_posix()}"
                if to_line < at_line:
                    return f"to_line must be >= at_line; got {to_line} < {at_line}."
                if at_line > line_count:
                    return (
                        f"at_line {at_line} is beyond the end of {relative.as_posix()} "
                        f"({line_count} lines)."
                    )
                if to_line > line_count:
                    return (
                        f"to_line {to_line} is beyond the end of {relative.as_posix()} "
                        f"({line_count} lines)."
                    )

                replacement = self._prepare_splice_content(
                    content,
                    followed_by_existing_line=to_line < line_count,
                )
                updated = lines[: at_line - 1] + replacement + lines[to_line:]

            self._write_text_atomic(target, "".join(updated))
        except (ValueError, OSError) as exc:
            return f"Failed to edit {relative.as_posix()}: {exc}"

        if to_line is None:
            return f"Inserted content at line {at_line} in {relative.as_posix()}"
        return (
            f"Replaced lines {at_line}-{to_line} in "
            f"{relative.as_posix()}"
        )

    def move_path(self, source: Path, destination: Path) -> str:
        """
        Moves or renames a documentation file/folder.

        Parent directories of destination are created automatically.
        The operation fails if destination already exists.
        """
        source_relative = Path(source)
        destination_relative = Path(destination)

        try:
            source_target = self._resolve_path(source_relative)
            destination_target = self._resolve_path(destination_relative)
            self._assert_mutable_path(source_relative)
            self._assert_mutable_path(destination_relative)

            if source_relative == Path(".") or destination_relative == Path("."):
                raise ValueError("the documentation root cannot be moved")
            if not source_target.exists():
                return f"Source path not found: {source_relative.as_posix()}"
            if destination_target.exists():
                return f"Destination already exists: {destination_relative.as_posix()}"
            if source_target.is_file():
                if source_target.suffix.lower() != ".md":
                    raise ValueError("only Markdown documentation files may be moved")
                if destination_target.suffix.lower() != ".md":
                    raise ValueError("Markdown files must keep a .md destination")
            elif source_target.is_dir():
                try:
                    destination_target.relative_to(source_target)
                except ValueError:
                    pass
                else:
                    raise ValueError("a directory cannot be moved inside itself")

            destination_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_target), str(destination_target))
        except (ValueError, OSError, shutil.Error) as exc:
            return (
                f"Failed to move {source_relative.as_posix()} -> "
                f"{destination_relative.as_posix()}: {exc}"
            )

        return (
            f"Moved documentation path: {source_relative.as_posix()} -> "
            f"{destination_relative.as_posix()}"
        )

    def delete_path(self, path: Path, recursive: bool = False) -> str:
        """
        Deletes a file or folder inside this repository's documentation.

        recursive=True is required to delete a non-empty directory.
        Protected metadata cannot be deleted through this operation.
        """
        relative = Path(path)
        try:
            target = self._resolve_path(relative)
            self._assert_mutable_path(relative)

            if relative == Path("."):
                raise ValueError("the documentation root cannot be deleted")
            if not target.exists():
                return f"Documentation path not found: {relative.as_posix()}"

            if target.is_dir():
                if recursive:
                    shutil.rmtree(target)
                else:
                    try:
                        target.rmdir()
                    except OSError:
                        return (
                            f"Directory is not empty: {relative.as_posix()}. "
                            "Use recursive=True to delete it."
                        )
            else:
                if target.suffix.lower() != ".md":
                    raise ValueError("only Markdown documentation files may be deleted")
                target.unlink()
        except (ValueError, OSError) as exc:
            return f"Failed to delete {relative.as_posix()}: {exc}"

        return f"Deleted documentation path: {relative.as_posix()}"

    def path_exists(self, path: Path) -> bool:
        """
        Returns whether a RELATIVE documentation path exists.

        Invalid/escaping paths return False.
        """
        try:
            return self._resolve_path(Path(path)).exists()
        except ValueError:
            return False

    def _resolve_path(self, path: Path) -> Path:
        """
        Resolves a RELATIVE agent-provided path against root.

        This is the security boundary for every documentation filesystem
        operation. Absolute paths and any path resolving outside root are
        rejected, including traversal through symlinks.
        """
        relative = Path(path)
        if relative.is_absolute():
            raise ValueError("absolute paths are not allowed")

        root = self.root.resolve(strict=False)
        candidate = (root / relative).resolve(strict=False)

        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("path escapes the documentation root") from exc

        return candidate

    def _resolve_doc_file(self, path: Path, *, must_exist: bool) -> Path:
        target = self._resolve_path(path)
        self._assert_mutable_path(path)
        if Path(path).suffix.lower() != ".md":
            raise ValueError("documentation files must use the .md extension")
        if must_exist and not target.exists():
            raise ValueError("file does not exist")
        return target

    def _assert_mutable_path(self, path: Path) -> None:
        parts = Path(path).parts
        if not parts:
            return
        if any(part in self._PROTECTED_NAMES for part in parts):
            raise ValueError("path targets protected repository metadata")

    def _is_inside_root(self, path: Path) -> bool:
        try:
            path.resolve(strict=False).relative_to(self.root.resolve(strict=False))
            return True
        except ValueError:
            return False

    def _append_tree(self, directory: Path, lines: list[str], prefix: str) -> None:
        entries: list[Path] = []
        try:
            for entry in directory.iterdir():
                if entry.name in self._PROTECTED_NAMES:
                    continue
                if not self._is_inside_root(entry):
                    continue
                if entry.is_dir() or (entry.is_file() and entry.suffix.lower() == ".md"):
                    entries.append(entry)
        except OSError:
            return

        entries.sort(key=lambda entry: (not entry.is_dir(), entry.name.casefold()))

        for index, entry in enumerate(entries):
            last = index == len(entries) - 1
            connector = "└── " if last else "├── "
            lines.append(prefix + connector + entry.name)
            if entry.is_dir():
                child_prefix = prefix + ("    " if last else "│   ")
                self._append_tree(entry, lines, child_prefix)

    @staticmethod
    def _prepare_splice_content(
        content: str,
        *,
        followed_by_existing_line: bool,
    ) -> list[str]:
        if not content:
            return []
        normalized = content
        if followed_by_existing_line and not normalized.endswith(("\n", "\r")):
            normalized += "\n"
        return normalized.splitlines(keepends=True)

    @staticmethod
    def _write_text_atomic(path: Path, content: str) -> None:
        temp = path.with_name(path.name + ".tmp")
        try:
            temp.write_text(content, encoding="utf-8", newline="")
            temp.replace(path)
        finally:
            if temp.exists():
                temp.unlink(missing_ok=True)

    @staticmethod
    def _file_not_found(path: Path) -> str:
        return (
            f"{_SEPARATOR}\n"
            f"// File not found: {path.as_posix()}\n"
            f"{_SEPARATOR}"
        )

    @staticmethod
    def _format_file(path: Path, content: str, start: int, end: int) -> str:
        body = content
        if body and not body.endswith("\n"):
            body += "\n"
        return (
            f"{_SEPARATOR}\n"
            f"// {path.as_posix()}\n"
            f"{_SEPARATOR}\n"
            f"{body}"
            f"{_SEPARATOR}\n"
            f"// End (lines {start}/{end})\n"
            f"{_SEPARATOR}"
        )


class RepositoryLibrary:
    """
    Holds and gives access to DocumentedRepositories.

    Provides repository-level and batched operations for:

        1. Repository discovery.
        2. Repository creation.
        3. Commit-version management.
        4. Documentation exploration.
        5. Documentation creation.
        6. Documentation updates.
        7. Documentation deletion.

    Documentation survives agent turns and the Citra runtime, storing it under:

        ./.citra/library/repos/

    RepositoryLibrary acts as the agent-facing facade.

    RepositoryLibrary itself performs no network or Git execution. Git-aware
    behavior is available only through the optional GitRepositoryUtility hook.
    """

    def __init__(
        self,
        root: Path
    ):
        """
        Initializes the repository documentation library.

            1. root:
                Directory under which all documented repositories are stored. Will start storing at root/repos

            2. git_utility:
                Optional integration hook capable of resolving Git revisions
                and source diffs. RepositoryLibrary never creates one itself.

        Creates root if necessary and scans all existing repository metadata.
        Invalid/corrupted entries do not prevent the remaining library loading.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.git_utility = GitRepositoryUtility(root=root)
        self.repositories: list[DocumentedRepository] = []
        self.scan_diagnostics: list[str] = []
        self._preferred_commits: dict[str, str] = {}
        self._scan_repositories()

    def reload(self) -> None:
        """
        Clears the in-memory registry and scans the library again.
        """
        self.repositories.clear()
        self.scan_diagnostics.clear()
        self._preferred_commits.clear()
        self._scan_repositories()

    def _scan_repositories(self) -> None:
        """
        Scans ./.citra/library/repos/ and instantiates one
        DocumentedRepository for every valid repository commit discovered.

        Malformed repository/version entries are skipped and recorded in
        scan_diagnostics rather than aborting the complete scan.
        """
        self.root.mkdir(parents=True, exist_ok=True)

        for repo_root in sorted(self.root.iterdir(), key=lambda path: path.name.casefold()):
            if not repo_root.is_dir():
                continue

            repo_metadata_path = repo_root / "repo.toml"
            if not repo_metadata_path.is_file():
                self.scan_diagnostics.append(
                    f"Skipped {repo_root}: missing repo.toml"
                )
                continue

            try:
                metadata = self._read_toml(repo_metadata_path)
                url = self._canonicalize_repo_url(self._require_str(metadata, "url"))
                name = self._require_str(metadata, "name")
                author = self._require_str(metadata, "author")
                expected_author, expected_name = self._parse_repo_identity(url)
                if name.casefold() != expected_name.casefold() or author.casefold() != expected_author.casefold():
                    raise ValueError("repo.toml identity does not match its canonical URL")

                expected_folder = self._get_repo_folder_name(url)
                if repo_root.name != expected_folder:
                    raise ValueError(
                        f"repository folder must be '{expected_folder}', got '{repo_root.name}'"
                    )

                preferred_commit = metadata.get("preferred_commit")
                if preferred_commit:
                    try:
                        self._preferred_commits[url] = self._normalize_commit(
                            str(preferred_commit)
                        )
                    except ValueError as exc:
                        self.scan_diagnostics.append(
                            f"Ignored invalid preferred_commit in {repo_metadata_path}: {exc}"
                        )

                commits_root = repo_root / "commits"
                if not commits_root.is_dir():
                    continue

                for revision_root in sorted(
                    commits_root.iterdir(), key=lambda path: path.name.casefold()
                ):
                    if not revision_root.is_dir():
                        continue
                    try:
                        revision_metadata = self._read_toml(revision_root / "revision.toml")
                        branch = self._validate_branch(
                            self._require_str(revision_metadata, "branch")
                        )
                        commit = self._normalize_commit(
                            self._require_str(revision_metadata, "commit")
                        )
                        if revision_root.name.casefold() != commit.casefold():
                            raise ValueError(
                                "revision directory name does not match revision.toml commit"
                            )

                        self.repositories.append(
                            DocumentedRepository(
                                url=url,
                                repo_root=repo_root,
                                root=revision_root,
                                name=name,
                                author=author,
                                branch=branch,
                                commit=commit,
                            )
                        )
                    except Exception as exc:  # malformed revision must not abort scan
                        self.scan_diagnostics.append(
                            f"Skipped revision {revision_root}: {exc}"
                        )
            except Exception as exc:  # malformed repo must not abort scan
                self.scan_diagnostics.append(f"Skipped repository {repo_root}: {exc}")

        self.repositories.sort(
            key=lambda repo: (repo.url.casefold(), repo.commit.casefold())
        )

        # Repair invalid/missing preferred commits in memory without mutating disk.
        for url in {repo.url for repo in self.repositories}:
            versions = self._find_repo_versions(url)
            preferred = self._preferred_commits.get(url)
            if preferred is None or all(repo.commit != preferred for repo in versions):
                if versions:
                    self._preferred_commits[url] = versions[-1].commit

    # -------------------------------------------------------------------------
    # Repository discovery
    # -------------------------------------------------------------------------

    def list_repositories(self) -> str:
        """
        Returns a compact list of repository identities and documented versions.
        Full documentation trees are intentionally omitted.
        """
        grouped: dict[str, list[DocumentedRepository]] = {}
        for repo in self.repositories:
            grouped.setdefault(repo.url, []).append(repo)

        if not grouped:
            return "No documented repositories."

        output = ["Documented repositories:"]
        for url in sorted(grouped, key=str.casefold):
            versions = sorted(grouped[url], key=lambda repo: repo.commit)
            sample = versions[0]
            preferred = self._preferred_commits.get(url)
            output.append(f"\n{sample.author}/{sample.name}")
            output.append(f"  URL: {url}")
            output.append(f"  Preferred commit: {preferred or '<none>'}")
            output.append("  Versions:")
            for version in versions:
                marker = " *" if version.commit == preferred else ""
                output.append(
                    f"    - {version.commit} [{version.branch}]{marker}"
                )
        return "\n".join(output)

    def get_repo(self, repo_url: str, commit: str | None = None) -> str:
        """
        Returns repository information, selected documentation tree and root
        index.md contents when available.
        """
        repo, diagnostic = self._resolve_repo_or_diagnostic(repo_url, commit)
        if repo is None:
            return diagnostic

        tree = repo.tree_docs()
        index = repo.view_file(Path("index.md")) if repo.path_exists(Path("index.md")) else "Root index.md does not exist."
        preferred = self._preferred_commits.get(repo.url)

        return (
            f"Repository: {repo.author}/{repo.name}\n"
            f"URL: {repo.url}\n"
            f"Branch: {repo.branch}\n"
            f"Commit: {repo.commit}\n"
            f"Preferred: {'yes' if repo.commit == preferred else 'no'}\n\n"
            f"Documentation tree:\n{tree}\n\n"
            f"Root index:\n{index}"
        )

    def list_versions(self, repo_url: str) -> str:
        """
        Lists all documentation versions available for a repository.
        """
        try:
            canonical = self._canonicalize_repo_url(repo_url)
        except ValueError as exc:
            return f"Invalid repository URL: {exc}"

        versions = self._find_repo_versions(canonical)
        if not versions:
            return f"Repository is not documented: {canonical}"

        preferred = self._preferred_commits.get(canonical)
        output = [f"Documented versions for {canonical}:"]
        for repo in sorted(versions, key=lambda item: item.commit):
            marker = " [preferred]" if repo.commit == preferred else ""
            index_state = "index.md" if repo.path_exists(Path("index.md")) else "no index.md"
            output.append(
                f"- {repo.commit} | branch={repo.branch} | {index_state}{marker}"
            )
        return "\n".join(output)

    def tree_docs(
        self,
        repo_url: str,
        path: Path | None = None,
        commit: str | None = None,
    ) -> str:
        """
        Returns the documentation tree for one repository version.
        """
        repo, diagnostic = self._resolve_repo_or_diagnostic(repo_url, commit)
        return repo.tree_docs(path) if repo is not None else diagnostic

    def search_docs(
        self,
        repo_url: str,
        query: str,
        path: Path | None = None,
        commit: str | None = None,
        max_results: int = 50,
        case_sensitive: bool = False,
    ) -> str:
        """
        Searches documentation belonging to one repository version.
        """
        repo, diagnostic = self._resolve_repo_or_diagnostic(repo_url, commit)
        if repo is None:
            return diagnostic
        return repo.search_docs(query, path, max_results, case_sensitive)

    def read_files(
        self,
        repo_url: str,
        files: list[Path],
        commit: str | None = None,
    ) -> str:
        """
        Reads several documentation files. Missing files do not prevent other
        requested files from being returned.
        """
        repo, diagnostic = self._resolve_repo_or_diagnostic(repo_url, commit)
        if repo is None:
            return diagnostic
        if not files:
            return "No documentation files requested."
        return "\n\n".join(repo.view_file(path) for path in files)

    # -------------------------------------------------------------------------
    # Repository creation / deletion
    # -------------------------------------------------------------------------

    def create_repo(self, repo_url: str, branch: str, commit: str) -> str:
        """
        Creates a new documented repository and its initial commit version.

        This operation performs no Git resolution. branch and commit are
        treated as already-resolved inputs.
        """
        canonical: str | None = None
        repo_root: Path | None = None
        revision_root: Path | None = None
        documented: DocumentedRepository | None = None
        created_repo_root = False

        try:
            canonical = self._canonicalize_repo_url(repo_url)
            branch = self._validate_branch(branch)
            commit = self._normalize_commit(commit)

            if self._repo_exists(canonical):
                return f"Repository is already documented: {canonical}"

            author, name = self._parse_repo_identity(canonical)
            repo_root = self.root / self._get_repo_folder_name(canonical)

            if repo_root.exists():
                return (
                    "Cannot create repository because its library folder "
                    f"already exists: {repo_root}"
                )

            revision_root = repo_root / "commits" / commit
            revision_root.mkdir(parents=True, exist_ok=False)
            created_repo_root = True

            self._write_repo_metadata(
                repo_root,
                url=canonical,
                name=name,
                author=author,
                preferred_commit=commit,
            )

            self._write_revision_metadata(
                revision_root,
                branch=branch,
                commit=commit,
            )

            (revision_root / "index.md").write_text(
                self._initial_index(author, name, branch, commit),
                encoding="utf-8",
                newline="",
            )

            documented = DocumentedRepository(
                url=canonical,
                repo_root=repo_root,
                root=revision_root,
                name=name,
                author=author,
                branch=branch,
                commit=commit,
            )

            self.repositories.append(documented)
            self.repositories.sort(key=lambda repo: (repo.url, repo.commit))
            self._preferred_commits[canonical] = commit

        except Exception as exc:
            # Roll back in-memory state if it was already registered.
            if documented is not None:
                try:
                    self.repositories.remove(documented)
                except ValueError:
                    pass

            if canonical is not None:
                self._preferred_commits.pop(canonical, None)

            # Delete only a directory created by this invocation.
            if created_repo_root and repo_root is not None:
                shutil.rmtree(repo_root, ignore_errors=True)

            return f"Failed to create documented repository: {exc}"

        return (
            f"Created documented repository: {canonical}\n"
            f"Branch: {branch}\n"
            f"Commit: {commit}\n"
            f"Root: {revision_root}"
        )

    def create_repo_from_git(
        self,
        repo_url: str,
        branch: str | None = None,
        commit: str | None = None,
    ) -> str:
        """
        Git utility hook for creating a repository from an unresolved Git ref.

        Requires git_utility. RepositoryLibrary canonicalizes the URL, asks the
        injected utility for a concrete revision, then delegates to create_repo().
        """
        try:
            canonical = self._canonicalize_repo_url(repo_url)
            revision = self._resolve_git_revision(canonical, branch, commit)
        except (ValueError, RuntimeError) as exc:
            return f"Failed to resolve repository revision: {exc}"
        return self.create_repo(canonical, revision.branch, revision.commit)

    def delete_repo(self, repo_url: str) -> str:
        """
        Deletes a complete repository and every documented commit.
        """
        try:
            canonical = self._canonicalize_repo_url(repo_url)
        except ValueError as exc:
            return f"Invalid repository URL: {exc}"

        versions = self._find_repo_versions(canonical)
        if not versions:
            return f"Repository is not documented: {canonical}"

        repo_root = versions[0].repo_root
        try:
            shutil.rmtree(repo_root)
        except OSError as exc:
            return f"Failed to delete repository {canonical}: {exc}"

        self.repositories = [repo for repo in self.repositories if repo.url != canonical]
        self._preferred_commits.pop(canonical, None)
        return f"Deleted documented repository: {canonical}"

    # -------------------------------------------------------------------------
    # Commit/version management
    # -------------------------------------------------------------------------

    def create_version(
        self,
        repo_url: str,
        branch: str,
        commit: str,
        copy_from_commit: str | None = None,
    ) -> str:
        """
        Creates documentation storage for another commit of an existing repo.

        If copy_from_commit is provided, documentation is copied before the new
        revision metadata is written. Copied documentation is not implicitly
        considered verified against the new source commit.
        """
        try:
            canonical = self._canonicalize_repo_url(repo_url)
            branch = self._validate_branch(branch)
            commit = self._normalize_commit(commit)
        except ValueError as exc:
            return f"Invalid repository version: {exc}"

        versions = self._find_repo_versions(canonical)
        if not versions:
            return f"Repository is not documented: {canonical}"
        if self._version_exists(canonical, commit):
            return f"Commit is already documented: {commit}"

        repo_root = versions[0].repo_root
        revision_root = repo_root / "commits" / commit

        try:
            if copy_from_commit is None:
                revision_root.mkdir(parents=True, exist_ok=False)
                (revision_root / "index.md").write_text(
                    self._initial_index(versions[0].author, versions[0].name, branch, commit),
                    encoding="utf-8",
                    newline="",
                )
            else:
                source_commit = self._normalize_commit(copy_from_commit)
                source = self._find_repo(canonical, source_commit)
                if source is None:
                    return (
                        f"Cannot copy documentation from unavailable commit: "
                        f"{source_commit}"
                    )
                shutil.copytree(source.root, revision_root)

            self._write_revision_metadata(revision_root, branch=branch, commit=commit)
            documented = DocumentedRepository(
                url=canonical,
                repo_root=repo_root,
                root=revision_root,
                name=versions[0].name,
                author=versions[0].author,
                branch=branch,
                commit=commit,
            )
            self.repositories.append(documented)
            self.repositories.sort(key=lambda repo: (repo.url, repo.commit))
            self._set_preferred_commit(canonical, commit)
        except Exception as exc:
            if revision_root.exists():
                shutil.rmtree(revision_root, ignore_errors=True)
            return f"Failed to create documentation version {commit}: {exc}"

        copied = f" copied from {copy_from_commit}" if copy_from_commit else ""
        return f"Created documentation version {commit} on branch {branch}{copied}."

    def create_version_from_git(
        self,
        repo_url: str,
        branch: str | None = None,
        commit: str | None = None,
        copy_from_commit: str | None = None,
    ) -> str:
        """
        Git utility hook for creating a new documentation version.
        """
        try:
            canonical = self._canonicalize_repo_url(repo_url)
            revision = self._resolve_git_revision(canonical, branch, commit)
        except (ValueError, RuntimeError) as exc:
            return f"Failed to resolve repository revision: {exc}"
        return self.create_version(
            canonical,
            revision.branch,
            revision.commit,
            copy_from_commit=copy_from_commit,
        )

    def set_preferred_version(self, repo_url: str, commit: str) -> str:
        """
        Marks an existing documented commit as the default version selected when
        callers omit commit=.
        """
        try:
            canonical = self._canonicalize_repo_url(repo_url)
            commit = self._normalize_commit(commit)
        except ValueError as exc:
            return f"Invalid repository version: {exc}"

        if not self._version_exists(canonical, commit):
            return f"Commit is not documented: {commit}"

        try:
            self._set_preferred_commit(canonical, commit)
        except OSError as exc:
            return f"Failed to update preferred commit: {exc}"
        return f"Preferred documentation version set to: {commit}"

    def delete_version(self, repo_url: str, commit: str) -> str:
        """
        Deletes one documented commit while preserving the repository entry and
        all other versions.
        """
        try:
            canonical = self._canonicalize_repo_url(repo_url)
            commit = self._normalize_commit(commit)
        except ValueError as exc:
            return f"Invalid repository version: {exc}"

        repo = self._find_repo(canonical, commit)
        if repo is None:
            return f"Commit is not documented: {commit}"

        try:
            shutil.rmtree(repo.root)
            self.repositories = [
                version
                for version in self.repositories
                if not (version.url == canonical and version.commit == commit)
            ]

            remaining = self._find_repo_versions(canonical)
            preferred = self._preferred_commits.get(canonical)
            if preferred == commit:
                replacement = remaining[-1].commit if remaining else ""
                self._set_preferred_commit(canonical, replacement)
        except OSError as exc:
            return f"Failed to delete documentation version {commit}: {exc}"

        return f"Deleted documentation version: {commit}"

    def changed_source_files(
        self,
        repo_url: str,
        from_commit: str,
        to_commit: str,
    ) -> str:
        """
        Git utility hook returning source-repository paths changed between two
        commits. This does NOT inspect or modify documentation.

        Intended future use:
            1. copy old documentation into a new version,
            2. inspect source paths changed between commits,
            3. selectively refresh potentially stale documentation.
        """
        try:
            canonical = self._canonicalize_repo_url(repo_url)
            from_commit = self._normalize_commit(from_commit)
            to_commit = self._normalize_commit(to_commit)
            if self.git_utility is None:
                raise RuntimeError("no GitRepositoryUtility is configured")
            paths = self.git_utility.changed_files(canonical, from_commit, to_commit)
        except (ValueError, RuntimeError, OSError) as exc:
            return f"Failed to obtain Git diff paths: {exc}"

        if not paths:
            return f"No changed source files between {from_commit} and {to_commit}."
        return (
            f"Changed source files between {from_commit} and {to_commit}:\n"
            + "\n".join(f"- {Path(path).as_posix()}" for path in paths)
        )

    # -------------------------------------------------------------------------
    # Documentation creation / updates
    # -------------------------------------------------------------------------

    def add_files(
        self,
        repo_url: str,
        files: dict[Path, str],
        commit: str | None = None,
    ) -> str:
        """
        Creates several documentation files. Each file is processed
        independently; existing files are never overwritten.
        """
        repo, diagnostic = self._resolve_repo_or_diagnostic(repo_url, commit)
        if repo is None:
            return diagnostic
        if not files:
            return "No documentation files requested."
        return self._batch_results(
            "Add documentation files",
            ((path, repo.add_file(path, content)) for path, content in files.items()),
        )

    def replace_files(
        self,
        repo_url: str,
        files: dict[Path, str],
        commit: str | None = None,
    ) -> str:
        """
        Completely replaces several EXISTING documentation files.
        Missing files fail individually and are never implicitly created.
        """
        repo, diagnostic = self._resolve_repo_or_diagnostic(repo_url, commit)
        if repo is None:
            return diagnostic
        if not files:
            return "No documentation files requested."
        return self._batch_results(
            "Replace documentation files",
            ((path, repo.replace_file(path, content)) for path, content in files.items()),
        )

    def edit_file(
        self,
        repo_url: str,
        path: Path,
        content: str,
        at_line: int,
        to_line: int | None = None,
        commit: str | None = None,
    ) -> str:
        """
        Performs one line-based edit over an existing documentation file.
        """
        repo, diagnostic = self._resolve_repo_or_diagnostic(repo_url, commit)
        if repo is None:
            return diagnostic
        return repo.edit_file(path, content, at_line, to_line)

    def edit_files(
        self,
        repo_url: str,
        edits: list[RepoFileEdit],
        commit: str | None = None,
    ) -> str:
        """
        Performs several line-based edits in order. Later line numbers refer to
        the already-updated file when multiple edits target the same path.
        """
        repo, diagnostic = self._resolve_repo_or_diagnostic(repo_url, commit)
        if repo is None:
            return diagnostic
        if not edits:
            return "No documentation edits requested."
        return self._batch_results(
            "Edit documentation files",
            (
                (
                    edit.path,
                    repo.edit_file(
                        edit.path,
                        edit.content,
                        edit.at_line,
                        edit.to_line,
                    ),
                )
                for edit in edits
            ),
        )

    def move_paths(
        self,
        repo_url: str,
        paths: dict[Path, Path],
        commit: str | None = None,
    ) -> str:
        """
        Moves or renames documentation paths in insertion order.
        """
        repo, diagnostic = self._resolve_repo_or_diagnostic(repo_url, commit)
        if repo is None:
            return diagnostic
        if not paths:
            return "No documentation paths requested."
        return self._batch_results(
            "Move documentation paths",
            (
                (
                    Path(f"{source.as_posix()} -> {destination.as_posix()}"),
                    repo.move_path(source, destination),
                )
                for source, destination in paths.items()
            ),
        )

    def delete_paths(
        self,
        repo_url: str,
        paths: list[Path],
        commit: str | None = None,
        recursive: bool = False,
    ) -> str:
        """
        Deletes several files/directories from one documentation version.
        Each deletion is processed independently.
        """
        repo, diagnostic = self._resolve_repo_or_diagnostic(repo_url, commit)
        if repo is None:
            return diagnostic
        if not paths:
            return "No documentation paths requested."
        return self._batch_results(
            "Delete documentation paths",
            ((path, repo.delete_path(path, recursive)) for path in paths),
        )

    # -------------------------------------------------------------------------
    # Internal repository resolution
    # -------------------------------------------------------------------------

    def _find_repo_versions(self, repo_url: str) -> list[DocumentedRepository]:
        """
        Returns every loaded documentation version belonging to repo_url.
        URL comparison uses canonical repository URLs.
        """
        try:
            canonical = self._canonicalize_repo_url(repo_url)
        except ValueError:
            return []
        return sorted(
            (repo for repo in self.repositories if repo.url == canonical),
            key=lambda repo: repo.commit,
        )

    def _find_repo(
        self,
        repo_url: str,
        commit: str | None = None,
    ) -> DocumentedRepository | None:
        """
        Resolves one DocumentedRepository.

        If commit is omitted, repo.toml preferred_commit is used. If metadata is
        absent/corrupt in-memory fallback is deterministic: lexicographically
        greatest documented commit.
        """
        try:
            canonical = self._canonicalize_repo_url(repo_url)
        except ValueError:
            return None

        versions = self._find_repo_versions(canonical)
        if not versions:
            return None

        if commit is not None:
            try:
                normalized = self._normalize_commit(commit)
            except ValueError:
                return None
            return next((repo for repo in versions if repo.commit == normalized), None)

        preferred = self._preferred_commits.get(canonical)
        if preferred:
            selected = next((repo for repo in versions if repo.commit == preferred), None)
            if selected is not None:
                return selected
        return versions[-1]

    def _repo_exists(self, repo_url: str) -> bool:
        """
        Returns whether at least one documented version exists for repo_url.
        """
        return bool(self._find_repo_versions(repo_url))

    def _version_exists(self, repo_url: str, commit: str) -> bool:
        """
        Returns whether an exact repository + commit snapshot exists.
        """
        return self._find_repo(repo_url, commit) is not None

    def _canonicalize_repo_url(self, repo_url: str) -> str:
        """
        Converts equivalent GitHub repository references into one stable form.

        Supported examples:
            https://github.com/foo/bar
            https://github.com/foo/bar.git
            git@github.com:foo/bar.git
            ssh://git@github.com/foo/bar.git
            github.com/foo/bar

        All normalize to:
            https://github.com/foo/bar

        Owner and repository names are case-folded for stable identity.
        """
        raw = repo_url.strip()
        if not raw:
            raise ValueError("repository URL cannot be empty")

        scp_match = _GITHUB_SCP_RE.fullmatch(raw)
        if scp_match:
            owner = scp_match.group("owner")
            name = scp_match.group("repo")
            return self._canonical_github_url(owner, name)

        candidate = raw
        if candidate.lower().startswith("github.com/"):
            candidate = "https://" + candidate

        parsed = urlparse(candidate)
        if parsed.scheme.lower() not in {"http", "https", "ssh", "git"}:
            raise ValueError("only GitHub HTTP(S), SSH and git URLs are supported")

        hostname = (parsed.hostname or "").casefold()
        if hostname != "github.com":
            raise ValueError("only github.com repositories are currently supported")
        if parsed.query or parsed.fragment:
            raise ValueError("repository URL must not contain query parameters or fragments")

        path = parsed.path.strip("/")
        if path.endswith(".git"):
            path = path[:-4]
        parts = path.split("/") if path else []
        if len(parts) != 2 or not all(parts):
            raise ValueError("expected a GitHub repository URL of the form owner/repository")

        return self._canonical_github_url(parts[0], parts[1])

    def _get_repo_folder_name(self, repo_url: str) -> str:
        """
        Returns <repository-name>-<stable-short-hash> where the hash derives
        from the canonical repository URL.
        """
        canonical = self._canonicalize_repo_url(repo_url)
        _, name = self._parse_repo_identity(canonical)
        digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=6).hexdigest()
        return f"{name}-{digest}"

    def _parse_repo_identity(self, repo_url: str) -> tuple[str, str]:
        """
        Extracts (author, repository_name) from a supported canonical GitHub URL.
        """
        canonical = self._canonicalize_repo_url(repo_url)
        parsed = urlparse(canonical)
        author, name = parsed.path.strip("/").split("/", maxsplit=1)
        return author, name

    # -------------------------------------------------------------------------
    # Internal persistence / Git hooks
    # -------------------------------------------------------------------------

    def _resolve_repo_or_diagnostic(
        self,
        repo_url: str,
        commit: str | None,
    ) -> tuple[DocumentedRepository | None, str]:
        try:
            canonical = self._canonicalize_repo_url(repo_url)
        except ValueError as exc:
            return None, f"Invalid repository URL: {exc}"

        versions = self._find_repo_versions(canonical)
        if not versions:
            return None, f"Repository is not documented: {canonical}"

        if commit is not None:
            try:
                normalized = self._normalize_commit(commit)
            except ValueError as exc:
                return None, f"Invalid commit: {exc}"
            repo = self._find_repo(canonical, normalized)
            if repo is None:
                available = ", ".join(version.commit for version in versions)
                return (
                    None,
                    f"Commit is not documented: {normalized}\n"
                    f"Available commits: {available}",
                )
            return repo, ""

        repo = self._find_repo(canonical)
        assert repo is not None
        return repo, ""

    def _resolve_git_revision(
        self,
        canonical_repo_url: str,
        branch: str | None,
        commit: str | None,
    ) -> ResolvedGitRevision:
        if self.git_utility is None:
            raise RuntimeError("no GitRepositoryUtility is configured")
        resolved = self.git_utility.resolve_revision(
            canonical_repo_url,
            branch=branch,
            commit=commit,
        )
        return ResolvedGitRevision(
            branch=self._validate_branch(resolved.branch),
            commit=self._normalize_commit(resolved.commit),
        )

    def _set_preferred_commit(self, repo_url: str, commit: str) -> None:
        canonical = self._canonicalize_repo_url(repo_url)
        versions = self._find_repo_versions(canonical)
        if not versions and commit:
            raise OSError("cannot set preferred commit for undocumented repository")

        repo_root = versions[0].repo_root if versions else self.root / self._get_repo_folder_name(canonical)
        author, name = self._parse_repo_identity(canonical)
        self._write_repo_metadata(
            repo_root,
            url=canonical,
            name=name,
            author=author,
            preferred_commit=commit,
        )
        if commit:
            self._preferred_commits[canonical] = commit
        else:
            self._preferred_commits.pop(canonical, None)

    @staticmethod
    def _normalize_commit(commit: str) -> str:
        normalized = commit.strip().lower()
        if not _COMMIT_RE.fullmatch(normalized):
            raise ValueError(
                "commit must be a 7-64 character hexadecimal Git object ID"
            )
        return normalized

    @staticmethod
    def _validate_branch(branch: str) -> str:
        normalized = branch.strip()
        if not normalized:
            raise ValueError("branch cannot be empty")
        if "\x00" in normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("branch contains invalid control characters")
        return normalized

    @staticmethod
    def _canonical_github_url(owner: str, name: str) -> str:
        owner = owner.strip().casefold()
        name = name.strip().casefold()
        if not owner or not name:
            raise ValueError("GitHub owner and repository name cannot be empty")
        if owner in {".", ".."} or name in {".", ".."}:
            raise ValueError("invalid GitHub repository identity")
        if any(char in owner + name for char in "\\\x00\n\r"):
            raise ValueError("invalid characters in GitHub repository identity")
        return f"https://github.com/{owner}/{name}"

    @staticmethod
    def _read_toml(path: Path) -> dict[str, object]:
        with path.open("rb") as file:
            return tomllib.load(file)

    @staticmethod
    def _require_str(data: dict[str, object], key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing or invalid string field: {key}")
        return value

    def _write_repo_metadata(
        self,
        repo_root: Path,
        *,
        url: str,
        name: str,
        author: str,
        preferred_commit: str,
    ) -> None:
        content = (
            f"url = {self._toml_string(url)}\n"
            f"name = {self._toml_string(name)}\n"
            f"author = {self._toml_string(author)}\n"
            f"preferred_commit = {self._toml_string(preferred_commit)}\n"
        )
        repo_root.mkdir(parents=True, exist_ok=True)
        self._write_text_atomic(repo_root / "repo.toml", content)

    def _write_revision_metadata(
        self,
        revision_root: Path,
        *,
        branch: str,
        commit: str,
    ) -> None:
        content = (
            f"branch = {self._toml_string(branch)}\n"
            f"commit = {self._toml_string(commit)}\n"
        )
        revision_root.mkdir(parents=True, exist_ok=True)
        self._write_text_atomic(revision_root / "revision.toml", content)

    @staticmethod
    def _write_text_atomic(path: Path, content: str) -> None:
        temp = path.with_name(path.name + ".tmp")
        try:
            temp.write_text(content, encoding="utf-8", newline="")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def _toml_string(value: str) -> str:
        # JSON basic string escaping is compatible with TOML basic strings for
        # the characters produced here.
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _initial_index(author: str, name: str, branch: str, commit: str) -> str:
        return (
            f"# {author}/{name}\n\n"
            f"Documentation snapshot for branch `{branch}` at commit `{commit}`.\n\n"
            "Use this index to describe the documentation folders and files "
            "available in this revision.\n"
        )

    @staticmethod
    def _batch_results(
        title: str,
        results,
    ) -> str:
        output = [f"{title}:"]
        for path, result in results:
            output.append(f"\n[{Path(path).as_posix()}]\n{result}")
        return "\n".join(output)
