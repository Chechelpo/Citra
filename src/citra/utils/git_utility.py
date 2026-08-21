from __future__ import annotations
from typing import Generator

import hashlib
import os
import re
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, Sequence
from urllib.parse import unquote, urlparse


_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
_GITHUB_PART_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class GitUtilityError(RuntimeError):
    """
    Base error for failures produced by GitRepositoryUtility.

    The utility raises Python exceptions instead of returning agent-facing
    diagnostic strings. RepositoryLibrary owns translation of these failures
    into its public string-based result format.
    """


class GitExecutableNotFound(GitUtilityError):
    """
    Raised when the configured git executable cannot be started.
    """


class GitCommandError(GitUtilityError):
    """
    Raised when an explicitly allowed Git command exits unsuccessfully.

        1. command: exact argv passed to subprocess. No shell is involved.
        2. returncode: process exit status.
        3. stderr: stderr emitted by Git, trimmed for diagnostics.
    """

    def __init__(
        self,
        command: Sequence[str],
        returncode: int,
        stderr: str,
    ) -> None:
        self.command = tuple(command)
        self.returncode = returncode
        self.stderr = stderr.strip()

        detail = self.stderr or "git returned no diagnostic output"
        super().__init__(
            f"git command failed with exit code {returncode}: {detail}"
        )

@dataclass(frozen=True)
class ResolvedGitRevision:
    """
    Result expected from the optional Git utility hook.

        1. branch: concrete branch associated with the resolved revision.
        2. commit: concrete hexadecimal Git object ID.

    RepositoryLibrary deliberately does not know how these values are obtained.
    A future Git utility may resolve them using a local checkout, a constrained
    git process, or a provider-specific client.
    """

    branch: str
    commit: str

@dataclass(frozen=True)
class CachedGitRepository:
    """
    Describes one local bare Git cache.

        1. url: canonical remote repository URL.
        2. root: local bare repository path.

    The cache contains Git metadata/objects only. It never checks out source
    files into Citra's workspace.
    """

    url: str
    root: Path


class GitRepositoryUtility:
    """
    Concrete Git implementation used by RepositoryLibrary.

    Its purpose is deliberately narrow:

        1. Resolve a GitHub repository branch/commit into a concrete revision.
        2. Determine which source paths changed between two commits.
        3. Maintain a bare object cache so repeated agent turns do not fetch the
           complete repository history again.

    Repository storage follows:

        ./.citra/git/repos/<name>-<hash>.git/

    The hash is derived from the canonical repository URL using the same
    BLAKE2b scheme as RepositoryLibrary.

    Security properties:

        - subprocess is invoked directly; shell=True is never used.
        - production remotes are restricted to HTTPS GitHub repository URLs.
        - Git terminal prompting is disabled.
        - system/global Git configuration is ignored by default.
        - transport protocols are restricted.
        - no working tree is created.
        - source-repository hooks are never executed.

    Local file:// remotes can optionally be enabled for tests and local tooling.
    They are disabled by default.
    """

    def __init__(
        self,
        root: Path = Path(".citra/git/repos"),
        *,
        git_executable: str = "git",
        command_timeout_seconds: float = 30.0,
        network_timeout_seconds: float = 180.0,
        lock_timeout_seconds: float = 30.0,
        stale_lock_seconds: float = 600.0,
        allow_user_git_config: bool = False,
        allow_file_urls: bool = False,
    ) -> None:
        """
        Initializes the Git utility.

            1. root:
                Persistent directory containing bare repository caches.

            2. git_executable:
                Executable used for every Git invocation.

            3. command_timeout_seconds:
                Timeout for local Git operations.

            4. network_timeout_seconds:
                Timeout for fetch/ls-remote operations.

            5. lock_timeout_seconds:
                Maximum time to contend for the per-repository cache lock.

            6. stale_lock_seconds:
                Age after which an abandoned lock file may be reclaimed.

            7. allow_user_git_config:
                Whether Git may read the caller's global/system configuration.
                False is safer and is the default.

            8. allow_file_urls:
                Allows file:// repositories in addition to GitHub HTTPS URLs.
                Intended for tests/local utilities. False by default.
        """
        if command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be > 0")
        if network_timeout_seconds <= 0:
            raise ValueError("network_timeout_seconds must be > 0")
        if lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be > 0")
        if stale_lock_seconds <= 0:
            raise ValueError("stale_lock_seconds must be > 0")

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.git_executable = git_executable
        self.command_timeout_seconds = command_timeout_seconds
        self.network_timeout_seconds = network_timeout_seconds
        self.lock_timeout_seconds = lock_timeout_seconds
        self.stale_lock_seconds = stale_lock_seconds
        self.allow_user_git_config = allow_user_git_config
        self.allow_file_urls = allow_file_urls

    # ---------------------------------------------------------------------
    # RepositoryLibrary protocol
    # ---------------------------------------------------------------------

    def resolve_revision(
        self,
        repo_url: str,
        branch: str | None = None,
        commit: str | None = None,
    ) -> ResolvedGitRevision:
        """
        Resolves a repository reference to one concrete branch + full commit ID.

            1. repo_url:
                Canonical repository URL supplied by RepositoryLibrary.

            2. branch:
                Optional branch associated with the requested revision.

            3. commit:
                Optional 7-64 character hexadecimal Git commit identifier.

        Resolution rules:

            branch=None, commit=None
                -> default remote branch + its current tip.

            branch=<branch>, commit=None
                -> requested branch + its current tip.

            branch=None, commit=<commit>
                -> resolves the commit and associates it with a remote branch
                   containing it, preferring the default branch.

            branch=<branch>, commit=<commit>
                -> resolves both and verifies that commit is reachable from the
                   requested remote branch.

        Always returns the complete commit object ID produced by Git.
        """
        canonical = self._canonicalize_remote_url(repo_url)
        requested_branch = (
            self._validate_branch(branch) if branch is not None else None
        )
        requested_commit = (
            self._normalize_commit(commit) if commit is not None else None
        )

        with self._repository_lock(canonical):
            repository = self._ensure_cache(canonical)
            self._fetch(repository)

            default_branch: str | None = None
            if requested_branch is None:
                default_branch = self._default_branch(canonical)

            if requested_branch is not None:
                branch_commit = self._resolve_branch_commit(
                    repository.root,
                    requested_branch,
                )

                if requested_commit is None:
                    return ResolvedGitRevision(
                        branch=requested_branch,
                        commit=branch_commit,
                    )

                concrete_commit = self._ensure_commit(
                    repository.root,
                    requested_commit,
                )
                if not self._is_ancestor(
                    repository.root,
                    concrete_commit,
                    branch_commit,
                ):
                    raise GitUtilityError(
                        f"commit {concrete_commit} is not reachable from "
                        f"branch '{requested_branch}'"
                    )

                return ResolvedGitRevision(
                    branch=requested_branch,
                    commit=concrete_commit,
                )

            assert default_branch is not None

            if requested_commit is None:
                return ResolvedGitRevision(
                    branch=default_branch,
                    commit=self._resolve_branch_commit(
                        repository.root,
                        default_branch,
                    ),
                )

            concrete_commit = self._ensure_commit(
                repository.root,
                requested_commit,
            )
            containing_branches = self._branches_containing_commit(
                repository.root,
                concrete_commit,
            )

            if not containing_branches:
                raise GitUtilityError(
                    f"commit {concrete_commit} is not reachable from any "
                    "fetched remote branch; specify a branch containing it or "
                    "ensure the commit is still reachable upstream"
                )

            selected_branch = (
                default_branch
                if default_branch in containing_branches
                else containing_branches[0]
            )
            return ResolvedGitRevision(
                branch=selected_branch,
                commit=concrete_commit,
            )

    def changed_files(
        self,
        repo_url: str,
        from_commit: str,
        to_commit: str,
    ) -> list[Path]:
        """
        Returns source-repository paths changed between two concrete commits.

            1. repo_url:
                Canonical repository URL supplied by RepositoryLibrary.

            2. from_commit:
                Older/base commit.

            3. to_commit:
                Newer/target commit.

        Rename/copy operations return BOTH their old and new paths. This is
        intentional for documentation invalidation: knowledge may reference
        either side of a moved source file.

        Results are de-duplicated while preserving Git's diff order.
        """
        canonical = self._canonicalize_remote_url(repo_url)
        normalized_from = self._normalize_commit(from_commit)
        normalized_to = self._normalize_commit(to_commit)

        with self._repository_lock(canonical):
            repository = self._ensure_cache(canonical)
            self._fetch(repository)

            concrete_from = self._ensure_commit(
                repository.root,
                normalized_from,
            )
            concrete_to = self._ensure_commit(
                repository.root,
                normalized_to,
            )

            result = self._run(
                [
                    "-C",
                    str(repository.root),
                    "diff",
                    "--name-status",
                    "-z",
                    "--find-renames",
                    "--find-copies",
                    concrete_from,
                    concrete_to,
                    "--",
                ],
            )

        return self._parse_changed_paths(result.stdout)

    # ---------------------------------------------------------------------
    # Cache operations
    # ---------------------------------------------------------------------

    def cache_path(self, repo_url: str) -> Path:
        """
        Returns the deterministic bare-cache path for repo_url.

        This operation does not access the network and does not require the
        cache to exist.
        """
        canonical = self._canonicalize_remote_url(repo_url)
        name = self._repo_name(canonical)
        digest = hashlib.blake2b(
            canonical.encode("utf-8"),
            digest_size=6,
        ).hexdigest()
        return self.root / f"{name}-{digest}.git"

    def is_cached(self, repo_url: str) -> bool:
        """
        Returns whether a valid-looking cache directory exists for repo_url.

        No fetch/network request is performed. A corrupt existing cache may
        still fail later validation when used.
        """
        return self.cache_path(repo_url).is_dir()

    def refresh(self, repo_url: str) -> CachedGitRepository:
        """
        Ensures repo_url has a bare cache and refreshes its remote branch/tag
        references.

        Returns the refreshed cache descriptor.
        """
        canonical = self._canonicalize_remote_url(repo_url)
        with self._repository_lock(canonical):
            repository = self._ensure_cache(canonical)
            self._fetch(repository)
            return repository

    def remove_cache(self, repo_url: str) -> None:
        """
        Removes the persistent bare cache for repo_url.

        This does not affect RepositoryLibrary documentation.
        """
        canonical = self._canonicalize_remote_url(repo_url)
        with self._repository_lock(canonical):
            path = self.cache_path(canonical)
            if path.exists():
                shutil.rmtree(path)

    # ---------------------------------------------------------------------
    # Cache creation / refresh
    # ---------------------------------------------------------------------

    def _ensure_cache(self, canonical_repo_url: str) -> CachedGitRepository:
        cache = self.cache_path(canonical_repo_url)

        if cache.exists():
            self._validate_existing_cache(cache, canonical_repo_url)
            return CachedGitRepository(canonical_repo_url, cache)

        temp = self.root / f".{cache.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            self._run(["init", "--bare", "--quiet", str(temp)])
            self._run(
                [
                    "-C",
                    str(temp),
                    "remote",
                    "add",
                    "origin",
                    canonical_repo_url,
                ]
            )
            temporary = CachedGitRepository(canonical_repo_url, temp)
            self._fetch(temporary)

            try:
                temp.replace(cache)
            except FileExistsError:
                # Another process that does not share this utility's lock may
                # have populated the deterministic cache concurrently.
                shutil.rmtree(temp, ignore_errors=True)
                self._validate_existing_cache(cache, canonical_repo_url)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise

        return CachedGitRepository(canonical_repo_url, cache)

    def _validate_existing_cache(
        self,
        cache: Path,
        canonical_repo_url: str,
    ) -> None:
        if not cache.is_dir():
            raise GitUtilityError(
                f"Git cache path exists but is not a directory: {cache}"
            )

        bare = self._run(
            ["-C", str(cache), "rev-parse", "--is-bare-repository"]
        ).stdout.strip()
        if bare != "true":
            raise GitUtilityError(f"Git cache is not a bare repository: {cache}")

        remote = self._run(
            ["-C", str(cache), "remote", "get-url", "origin"]
        ).stdout.strip()
        try:
            remote_canonical = self._canonicalize_remote_url(remote)
        except ValueError as exc:
            raise GitUtilityError(
                f"Git cache origin is invalid for {cache}: {exc}"
            ) from exc

        if remote_canonical != canonical_repo_url:
            raise GitUtilityError(
                f"Git cache origin mismatch for {cache}: expected "
                f"{canonical_repo_url}, got {remote_canonical}"
            )

    def _fetch(self, repository: CachedGitRepository) -> None:
        """
        Refreshes all remote branches and tags into the bare cache.

        --filter=blob:none keeps source file bodies out of the cache unless Git
        needs them later. Commit/tree data remains available for revision
        resolution and path-only diffs.
        """
        self._run(
            [
                "-C",
                str(repository.root),
                "fetch",
                "--quiet",
                "--prune",
                "--force",
                "--filter=blob:none",
                "origin",
                "+refs/heads/*:refs/remotes/origin/*",
                "+refs/tags/*:refs/tags/*",
            ],
            timeout=self.network_timeout_seconds,
        )

    # ---------------------------------------------------------------------
    # Revision resolution
    # ---------------------------------------------------------------------

    def _default_branch(self, canonical_repo_url: str) -> str:
        result = self._run(
            ["ls-remote", "--symref", canonical_repo_url, "HEAD"],
            timeout=self.network_timeout_seconds,
        )

        for line in result.stdout.splitlines():
            if not line.startswith("ref: "):
                continue
            try:
                ref, target = line.split("\t", maxsplit=1)
            except ValueError:
                continue
            if target != "HEAD":
                continue

            ref_name = ref.removeprefix("ref: ")
            prefix = "refs/heads/"
            if not ref_name.startswith(prefix):
                continue
            return self._validate_branch(ref_name[len(prefix) :])

        raise GitUtilityError(
            f"remote repository does not advertise a symbolic default branch: "
            f"{canonical_repo_url}"
        )

    def _resolve_branch_commit(self, repository: Path, branch: str) -> str:
        ref = f"refs/remotes/origin/{self._validate_branch(branch)}"
        try:
            return self._rev_parse_commit(repository, ref)
        except GitCommandError as exc:
            raise GitUtilityError(
                f"remote branch does not exist: {branch}"
            ) from exc

    def _ensure_commit(self, repository: Path, commit: str) -> str:
        commit = self._normalize_commit(commit)

        try:
            return self._rev_parse_commit(repository, commit)
        except GitCommandError:
            # A commit may be valid but no longer reachable from the currently
            # fetched branch/tag tips. GitHub may still allow direct SHA fetch.
            try:
                self._run(
                    [
                        "-C",
                        str(repository),
                        "fetch",
                        "--quiet",
                        "--filter=blob:none",
                        "origin",
                        commit,
                    ],
                    timeout=self.network_timeout_seconds,
                )
            except GitCommandError as fetch_error:
                raise GitUtilityError(
                    f"commit is unavailable from remote repository: {commit}"
                ) from fetch_error

        try:
            return self._rev_parse_commit(repository, commit)
        except GitCommandError as exc:
            raise GitUtilityError(
                f"Git object is not a commit: {commit}"
            ) from exc

    def _rev_parse_commit(self, repository: Path, revision: str) -> str:
        result = self._run(
            [
                "-C",
                str(repository),
                "rev-parse",
                "--verify",
                f"{revision}^{{commit}}",
            ]
        )
        commit = result.stdout.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
            raise GitUtilityError(
                f"git returned an invalid concrete commit ID: {commit!r}"
            )
        return commit

    def _is_ancestor(
        self,
        repository: Path,
        ancestor: str,
        descendant: str,
    ) -> bool:
        result = self._run(
            [
                "-C",
                str(repository),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            allowed_returncodes=(0, 1),
        )
        return result.returncode == 0

    def _branches_containing_commit(
        self,
        repository: Path,
        commit: str,
    ) -> list[str]:
        result = self._run(
            [
                "-C",
                str(repository),
                "for-each-ref",
                f"--contains={commit}",
                "--format=%(refname:strip=3)",
                "refs/remotes/origin/",
            ]
        )

        branches = {
            self._validate_branch(line.strip())
            for line in result.stdout.splitlines()
            if line.strip() and line.strip() != "HEAD"
        }
        return sorted(branches, key=str.casefold)

    # ---------------------------------------------------------------------
    # Changed path parsing
    # ---------------------------------------------------------------------

    @staticmethod
    def _parse_changed_paths(output: str) -> list[Path]:
        """
        Parses:

            git diff --name-status -z --find-renames --find-copies ...

        For R* and C* statuses, Git emits two paths. Both are returned.
        """
        if not output:
            return []

        fields = output.split("\x00")
        if fields and fields[-1] == "":
            fields.pop()

        changed: list[Path] = []
        seen: set[str] = set()
        index = 0

        while index < len(fields):
            status = fields[index]
            index += 1
            if not status:
                raise GitUtilityError("malformed NUL-delimited Git diff output")
            if index >= len(fields):
                raise GitUtilityError("Git diff output ended before its path")

            first_path = fields[index]
            index += 1
            paths = [first_path]

            if status[0] in {"R", "C"}:
                if index >= len(fields):
                    raise GitUtilityError(
                        "Git rename/copy output ended before destination path"
                    )
                paths.append(fields[index])
                index += 1

            for raw_path in paths:
                normalized = GitRepositoryUtility._validate_git_path(raw_path)
                key = normalized.as_posix()
                if key not in seen:
                    seen.add(key)
                    changed.append(normalized)

        return changed

    @staticmethod
    def _validate_git_path(path: str) -> Path:
        if not path or "\x00" in path:
            raise GitUtilityError("Git returned an invalid empty/NUL source path")

        posix = PurePosixPath(path)
        if posix.is_absolute() or ".." in posix.parts:
            raise GitUtilityError(f"Git returned an unsafe source path: {path!r}")

        # Git path output uses '/' regardless of host OS. Path conversion is
        # intentionally delayed until after POSIX traversal validation.
        return Path(*posix.parts)

    # ---------------------------------------------------------------------
    # URL / branch / commit validation
    # ---------------------------------------------------------------------

    def _canonicalize_remote_url(self, repo_url: str) -> str:
        """
        Independently validates the remote even though RepositoryLibrary already
        canonicalizes GitHub URLs. The Git process boundary never trusts its
        caller to have applied that validation correctly.
        """
        value = repo_url.strip()
        if not value:
            raise ValueError("repository URL cannot be empty")

        parsed = urlparse(value)

        if parsed.scheme.casefold() == "file":
            if not self.allow_file_urls:
                raise ValueError("file:// Git remotes are disabled")
            if parsed.netloc not in {"", "localhost"}:
                raise ValueError("file:// remote must reference the local host")
            if not parsed.path:
                raise ValueError("file:// remote path cannot be empty")
            return value

        if parsed.scheme.casefold() != "https":
            raise ValueError("only HTTPS GitHub repository URLs are allowed")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("repository URLs cannot contain credentials")
        if parsed.port is not None:
            raise ValueError("repository URLs cannot specify a custom port")
        if (parsed.hostname or "").casefold() != "github.com":
            raise ValueError("only github.com repositories are allowed")
        if parsed.query or parsed.fragment or parsed.params:
            raise ValueError("repository URL cannot contain query/fragment data")

        parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
        if len(parts) != 2:
            raise ValueError(
                "GitHub repository URL must have exactly owner/repository"
            )

        owner, name = parts
        if name.casefold().endswith(".git"):
            name = name[:-4]

        if not owner or not name:
            raise ValueError("GitHub owner and repository name cannot be empty")
        if not _GITHUB_PART_RE.fullmatch(owner):
            raise ValueError("invalid GitHub owner")
        if not _GITHUB_PART_RE.fullmatch(name):
            raise ValueError("invalid GitHub repository name")
        if owner in {".", ".."} or name in {".", ".."}:
            raise ValueError("invalid GitHub repository identity")

        return f"https://github.com/{owner.casefold()}/{name.casefold()}"

    def _repo_name(self, canonical_repo_url: str) -> str:
        parsed = urlparse(canonical_repo_url)
        if parsed.scheme.casefold() == "file":
            name = Path(unquote(parsed.path)).name
            if name.casefold().endswith(".git"):
                name = name[:-4]
            return name or "repository"

        name = parsed.path.rstrip("/").split("/")[-1]
        return name[:-4] if name.casefold().endswith(".git") else name

    def _validate_branch(self, branch: str) -> str:
        normalized = branch.strip()
        if not normalized:
            raise ValueError("branch cannot be empty")
        if normalized.startswith("-"):
            raise ValueError("branch cannot begin with '-'")
        if "\x00" in normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("branch contains invalid control characters")

        try:
            self._run(["check-ref-format", "--branch", normalized])
        except GitCommandError as exc:
            raise ValueError(f"invalid Git branch name: {normalized}") from exc
        return normalized

    @staticmethod
    def _normalize_commit(commit: str) -> str:
        normalized = commit.strip().lower()
        if not _COMMIT_RE.fullmatch(normalized):
            raise ValueError(
                "commit must be a 7-64 character hexadecimal Git object ID"
            )
        return normalized

    # ---------------------------------------------------------------------
    # Process boundary
    # ---------------------------------------------------------------------

    def _run(
        self,
        arguments: Sequence[str],
        *,
        timeout: float | None = None,
        allowed_returncodes: Sequence[int] = (0,),
    ) -> subprocess.CompletedProcess[str]:
        """
        Executes one direct Git command.

        No command string is constructed and no shell is started.
        """
        command = [self.git_executable, *arguments]
        environment = os.environ.copy()
        environment.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_PROTOCOL_FROM_USER": "0",
                "GIT_ALLOW_PROTOCOL": (
                    "https:http:file" if self.allow_file_urls else "https:http"
                ),
                "LC_ALL": "C",
            }
        )

        if not self.allow_user_git_config:
            environment.update(
                {
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_SYSTEM": os.devnull,
                    "GIT_CONFIG_GLOBAL": os.devnull,
                }
            )

        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=(
                    self.command_timeout_seconds if timeout is None else timeout
                ),
                env=environment,
                shell=False,
                check=False,
            )
        except FileNotFoundError as exc:
            raise GitExecutableNotFound(
                f"git executable was not found: {self.git_executable}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise GitUtilityError(
                f"git command exceeded its configured timeout: "
                f"{arguments[0] if arguments else '<unknown>'}"
            ) from exc
        except OSError as exc:
            raise GitUtilityError(f"failed to execute git: {exc}") from exc

        if result.returncode not in allowed_returncodes:
            raise GitCommandError(command, result.returncode, result.stderr)
        return result

    # ---------------------------------------------------------------------
    # Cross-agent/process cache lock
    # ---------------------------------------------------------------------

    @contextmanager
    def _repository_lock(self, canonical_repo_url: str) -> Generator[None]:
        """
        Provides a small cross-process lock around mutations of one bare cache.

        The lock uses atomic O_EXCL creation and therefore does not require an
        external locking dependency. Stale lock files can be reclaimed after
        stale_lock_seconds.
        """
        cache = self.cache_path(canonical_repo_url)
        lock_path = self.root / f".{cache.name}.lock"
        token = f"{os.getpid()}:{uuid.uuid4().hex}"
        deadline = time.monotonic() + self.lock_timeout_seconds

        while True:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                try:
                    os.write(descriptor, token.encode("utf-8"))
                finally:
                    os.close(descriptor)
                break
            except FileExistsError:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age >= self.stale_lock_seconds:
                        lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue

                if time.monotonic() >= deadline:
                    raise GitUtilityError(
                        f"timed out acquiring Git cache lock for "
                        f"{canonical_repo_url}"
                    )
                time.sleep(0.05)

        try:
            yield
        finally:
            try:
                if lock_path.read_text(encoding="utf-8") == token:
                    lock_path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            except OSError:
                # Lock cleanup failure must not hide the operation result. A
                # later caller can reclaim it as stale.
                pass
