"""Tests for per-tool format_call_log / format_result_log overrides."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from citra.tools.session_memory import (
    CheckpointTool,
    ConstraintTool,
    DecisionTool,
    FactTool,
    TodoTool,
)
from citra.tools.transient import (
    Browser,
    Commit,
    Curl,
    Edit,
    Glob,
    Git,
    Grep,
    Lsp,
    Materialize,
    PromptUser,
    Read,
    RepoLibrary,
    SkillTool,
    Subprocess,
    Tree,
    WebSearch,
    Write,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx() -> SimpleNamespace:
    """Minimal stand-in for ExecutionContext."""

    return SimpleNamespace()


def _session_ctx() -> SimpleNamespace:
    """Minimal context + session for memory tools."""

    return SimpleNamespace(session=SimpleNamespace(turn_number=1))


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

class TestRead:
    def test_call_log_single_path(self) -> None:
        log = Read(_ctx()).format_call_log(
            {"path": "src/main.py"}
        )
        assert "path=src/main.py" in log

    def test_call_log_with_offset_limit(self) -> None:
        log = Read(_ctx()).format_call_log(
            {"path": "src/main.py", "offset": 10, "limit": 50}
        )
        assert "offset=10" in log
        assert "limit=50" in log

    def test_call_log_batch(self) -> None:
        log = Read(_ctx()).format_call_log(
            {"requests": [{"path": "a.py"}, {"path": "b.py"}]}
        )
        assert "batch=2" in log

    def test_result_log_content(self) -> None:
        result = "===== a.py =====\nline 1\nline 2\n"
        log = Read(_ctx()).format_result_log(result)
        assert "lines" in log
        assert "chars" in log

    def test_result_log_empty(self) -> None:
        log = Read(_ctx()).format_result_log("")
        assert "empty" in log


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------

class TestEdit:
    def test_call_log_replacement(self) -> None:
        log = Edit(_ctx()).format_call_log(
            {"path": "a.py", "old": "x", "new": "y"}
        )
        assert "path=a.py" in log
        assert "old=" in log
        assert "new=" in log

    def test_call_log_insert(self) -> None:
        log = Edit(_ctx()).format_call_log(
            {"path": "a.py", "line": 5, "new": "inserted"}
        )
        assert "insert@line=5" in log

    def test_call_log_all_flag(self) -> None:
        log = Edit(_ctx()).format_call_log(
            {"path": "a.py", "old": "x", "new": "y", "all": True}
        )
        assert "all=true" in log

    def test_result_log_ok(self) -> None:
        assert Edit(_ctx()).format_result_log("ok") == "ok"

    def test_result_log_error(self) -> None:
        log = Edit(_ctx()).format_result_log("error: old_string not found")
        assert "error" in log


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

class TestWrite:
    def test_call_log(self) -> None:
        log = Write(_ctx()).format_call_log(
            {"path": "a.py", "content": "hello\n"}
        )
        assert "path=a.py" in log
        assert "6 chars" in log

    def test_result_log(self) -> None:
        assert Write(_ctx()).format_result_log("ok") == "ok"


# ---------------------------------------------------------------------------
# Glob
# ---------------------------------------------------------------------------

class TestGlob:
    def test_call_log_pattern_only(self) -> None:
        log = Glob(_ctx()).format_call_log(
            {"pat": "**/*.py"}
        )
        assert "pat=**/*.py" in log
        assert "path=" not in log

    def test_call_log_with_path(self) -> None:
        log = Glob(_ctx()).format_call_log(
            {"pat": "*.py", "path": "src"}
        )
        assert "path=src" in log

    def test_result_log_matches(self) -> None:
        log = Glob(_ctx()).format_result_log("a.py\nb.py\nc.py")
        assert "3 match" in log

    def test_result_log_no_matches(self) -> None:
        assert "no matches" in Glob(_ctx()).format_result_log("none")


# ---------------------------------------------------------------------------
# Grep
# ---------------------------------------------------------------------------

class TestGrep:
    def test_call_log(self) -> None:
        log = Grep(_ctx()).format_call_log(
            {"pat": "TODO|FIXME", "path": "src"}
        )
        assert "pat=" in log
        assert "path=src" in log

    def test_call_log_truncation(self) -> None:
        long_pat = "x" * 200
        log = Grep(_ctx()).format_call_log({"pat": long_pat})
        assert "..." in log

    def test_result_log_matches(self) -> None:
        log = Grep(_ctx()).format_result_log("a.py:1:line\nb.py:2:line")
        assert "2 match" in log

    def test_result_log_no_matches(self) -> None:
        assert "no matches" in Grep(_ctx()).format_result_log("none")


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------

class TestTree:
    def test_call_log_defaults(self) -> None:
        log = Tree(_ctx()).format_call_log({"path": "."})
        assert "path=." in log

    def test_call_log_with_options(self) -> None:
        log = Tree(_ctx()).format_call_log(
            {"path": "src", "max_depth": 5, "directories_only": True}
        )
        assert "depth=5" in log
        assert "dirs-only=true" in log

    def test_call_log_with_skip(self) -> None:
        log = Tree(_ctx()).format_call_log(
            {"path": "src", "skip": ["node_modules", ".git"]}
        )
        assert "skip=2" in log

    def test_result_log_summary(self) -> None:
        text = "src\n  file.py\n\n3 directories, 5 files"
        log = Tree(_ctx()).format_result_log(text)
        assert "lines" in log


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

class TestGit:
    def test_call_log_status(self) -> None:
        log = Git(_ctx()).format_call_log(
            {"action": "status", "path": "@source"}
        )
        assert "action=status" in log
        assert "path=@source" in log

    def test_call_log_log_with_ref(self) -> None:
        log = Git(_ctx()).format_call_log(
            {"action": "log", "ref": "HEAD", "limit": 10}
        )
        assert "action=log" in log
        assert "ref=HEAD" in log
        assert "limit=10" in log

    def test_call_log_clone(self) -> None:
        log = Git(_ctx()).format_call_log(
            {"action": "clone", "url": "https://github.com/a/b"}
        )
        assert "action=clone" in log
        assert "url=" in log

    def test_result_log_normal(self) -> None:
        log = Git(_ctx()).format_result_log("line1\nline2\nline3")
        assert "3 lines" in log
        assert "chars" in log

    def test_result_log_error(self) -> None:
        log = Git(_ctx()).format_result_log(
            "error: git exited with code 1\nsome error"
        )
        assert "exit=1" in log

    def test_result_log_timeout(self) -> None:
        log = Git(_ctx()).format_result_log(
            "output\n(timed out after 30s)"
        )
        assert "timed-out" in log


# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------

class TestCommit:
    def test_call_log_status(self) -> None:
        log = Commit(_ctx()).format_call_log({"action": "status"})
        assert "action=status" in log

    def test_call_log_stage_with_paths(self) -> None:
        log = Commit(_ctx()).format_call_log(
            {"action": "stage", "paths": ["a.py", "b.py", "c.py"]}
        )
        assert "action=stage" in log
        assert "paths=" in log

    def test_call_log_many_paths(self) -> None:
        paths = [f"file{i}.py" for i in range(10)]
        log = Commit(_ctx()).format_call_log(
            {"action": "stage", "paths": paths}
        )
        assert "+7 more" in log

    def test_call_log_stage_patch(self) -> None:
        log = Commit(_ctx()).format_call_log(
            {"action": "stage_patch", "patch": "--- diff ---"}
        )
        assert "action=stage_patch" in log
        assert "patch=<redacted>" in log

    def test_result_log(self) -> None:
        log = Commit(_ctx()).format_result_log("staged 3 files\n")
        assert "lines" in log


# ---------------------------------------------------------------------------
# Curl
# ---------------------------------------------------------------------------

class TestCurl:
    def test_call_log_get(self) -> None:
        log = Curl(_ctx()).format_call_log(
            {"url": "https://example.com/api", "method": "GET"}
        )
        assert "GET" in log
        assert "https://example.com/api" in log

    def test_call_log_post_with_body(self) -> None:
        log = Curl(_ctx()).format_call_log(
            {"url": "https://example.com", "method": "POST", "data": "body"}
        )
        assert "POST" in log
        assert "body=" in log

    def test_call_log_with_headers(self) -> None:
        log = Curl(_ctx()).format_call_log(
            {
                "url": "https://example.com",
                "method": "GET",
                "headers": ["Accept: json", "Auth: token"],
            }
        )
        assert "headers=2" in log

    def test_result_log_download(self) -> None:
        log = Curl(_ctx()).format_result_log(
            "Downloaded to @tmp/file.bin"
        )
        assert "Downloaded to" in log

    def test_result_log_permission_denied(self) -> None:
        log = Curl(_ctx()).format_result_log(
            "permission-denied: curl request was not executed."
        )
        assert "permission-denied" in log

    def test_result_log_normal(self) -> None:
        log = Curl(_ctx()).format_result_log("response body\n")
        assert "lines" in log


# ---------------------------------------------------------------------------
# WebSearch
# ---------------------------------------------------------------------------

class TestWebSearch:
    def test_call_log_basic(self) -> None:
        log = WebSearch(_ctx()).format_call_log(
            {"query": "python asyncio best practices"}
        )
        assert "query=" in log

    def test_call_log_truncation(self) -> None:
        long_query = "x" * 200
        log = WebSearch(_ctx()).format_call_log({"query": long_query})
        assert "..." in log

    def test_call_log_with_categories(self) -> None:
        log = WebSearch(_ctx()).format_call_log(
            {"query": "test", "categories": ["general", "news"]}
        )
        assert "categories=general,news" in log

    def test_result_log_dict(self) -> None:
        result = {
            "query": "test",
            "page": 1,
            "returned_results": 5,
            "results": [{"title": "a"}, {"title": "b"}],
            "answers": [],
            "suggestions": [],
            "corrections": [],
        }
        log = WebSearch(_ctx()).format_result_log(result)
        assert "5 result" in log

    def test_result_log_with_answers(self) -> None:
        result = {
            "query": "test",
            "page": 1,
            "returned_results": 0,
            "results": [],
            "answers": ["answer1"],
            "suggestions": [],
            "corrections": [],
        }
        log = WebSearch(_ctx()).format_result_log(result)
        assert "1 answer" in log


# ---------------------------------------------------------------------------
# Lsp
# ---------------------------------------------------------------------------

class TestLsp:
    def test_call_log_diagnostics(self) -> None:
        log = Lsp(_ctx()).format_call_log(
            {"action": "diagnostics", "path": "src/app.py"}
        )
        assert "action=diagnostics" in log
        assert "path=src/app.py" in log

    def test_call_log_hover(self) -> None:
        log = Lsp(_ctx()).format_call_log(
            {"action": "hover", "path": "src/app.py", "line": 10, "character": 5}
        )
        assert "pos=10:5" in log

    def test_result_log_none(self) -> None:
        assert Lsp(_ctx()).format_result_log("none") == "none"

    def test_result_log_content(self) -> None:
        log = Lsp(_ctx()).format_result_log("line1\nline2\nline3")
        assert "3 line" in log


# ---------------------------------------------------------------------------
# PromptUser
# ---------------------------------------------------------------------------

class TestPromptUser:
    def test_call_log_plain_text(self) -> None:
        log = PromptUser(_ctx()).format_call_log(
            {"question": "What should I do?"}
        )
        assert "mode=plain-text" in log
        assert "q=" in log

    def test_call_log_option_list(self) -> None:
        log = PromptUser(_ctx()).format_call_log(
            {"question": "Choose", "options": ["A", "B"]}
        )
        assert "mode=option-list" in log
        assert "options=2" in log

    def test_result_log_answer(self) -> None:
        log = PromptUser(_ctx()).format_result_log("some answer")
        assert "some answer" in log

    def test_result_log_truncation(self) -> None:
        long_answer = "x" * 200
        log = PromptUser(_ctx()).format_result_log(long_answer)
        assert "..." in log

    def test_result_log_user_unavailable(self) -> None:
        from citra.tools.transient.prompt_user import USER_UNAVAILABLE_MESSAGE

        log = PromptUser(_ctx()).format_result_log(USER_UNAVAILABLE_MESSAGE)
        assert "user-unavailable" in log


# ---------------------------------------------------------------------------
# Materialize
# ---------------------------------------------------------------------------

class TestMaterialize:
    def test_call_log_copy(self) -> None:
        log = Materialize(_ctx()).format_call_log(
            {"action": "copy", "paths": ["src/app.py"]}
        )
        assert "action=copy" in log
        assert "paths=" in log

    def test_call_log_preview(self) -> None:
        log = Materialize(_ctx()).format_call_log(
            {"action": "preview", "paths": ["src/**/*.py"]}
        )
        assert "action=preview" in log

    def test_result_log(self) -> None:
        log = Materialize(_ctx()).format_result_log("copied 3 files\n")
        assert "lines" in log


# ---------------------------------------------------------------------------
# Subprocess
# ---------------------------------------------------------------------------

class TestSubprocess:
    def test_call_log_start(self) -> None:
        log = Subprocess(_ctx()).format_call_log(
            {"action": "start", "cmd": "python -m http.server"}
        )
        assert "action=start" in log
        assert "$ python -m http.server" in log

    def test_call_log_poll(self) -> None:
        log = Subprocess(_ctx()).format_call_log(
            {"action": "poll", "process_id": 3}
        )
        assert "action=poll" in log
        assert "pid=3" in log

    def test_call_log_write(self) -> None:
        log = Subprocess(_ctx()).format_call_log(
            {"action": "write", "process_id": 1, "input": "data"}
        )
        assert "input=4 chars" in log

    def test_result_log_ok(self) -> None:
        assert Subprocess(_ctx()).format_result_log("ok") == "ok"

    def test_result_log_started(self) -> None:
        log = Subprocess(_ctx()).format_result_log("Started subprocess 1.")
        assert "Started" in log


# ---------------------------------------------------------------------------
# Browser
# ---------------------------------------------------------------------------

class TestBrowser:
    def test_call_log_open(self) -> None:
        log = Browser(_ctx()).format_call_log(
            {"action": "open", "url": "https://example.com"}
        )
        assert "action=open" in log
        assert "url=https://example.com" in log

    def test_call_log_click(self) -> None:
        log = Browser(_ctx()).format_call_log(
            {"action": "click", "ref": "abc123"}
        )
        assert "action=click" in log
        assert "ref=abc123" in log

    def test_call_log_fill(self) -> None:
        log = Browser(_ctx()).format_call_log(
            {"action": "fill", "ref": "x", "value": "text"}
        )
        assert "value=text" in log

    def test_result_log_closed(self) -> None:
        assert Browser(_ctx()).format_result_log(
            "Browser session closed."
        ) == "closed"


# ---------------------------------------------------------------------------
# RepoLibrary
# ---------------------------------------------------------------------------

class TestRepoLibrary:
    def test_call_log_list(self) -> None:
        log = RepoLibrary(_ctx()).format_call_log({"action": "list"})
        assert "action=list" in log

    def test_call_log_search(self) -> None:
        log = RepoLibrary(_ctx()).format_call_log(
            {"action": "search", "repo_url": "https://github.com/a/b", "query": "error"}
        )
        assert "action=search" in log
        assert "repo=https://github.com/a/b" in log
        assert "query=error" in log

    def test_call_log_add_documents(self) -> None:
        log = RepoLibrary(_ctx()).format_call_log(
            {
                "action": "add",
                "repo_url": "https://github.com/a/b",
                "documents": [{"path": "a.md", "content": "x"}],
            }
        )
        assert "documents=1" in log

    def test_result_log(self) -> None:
        log = RepoLibrary(_ctx()).format_result_log("some output\nline2")
        assert "lines" in log


# ---------------------------------------------------------------------------
# SkillTool
# ---------------------------------------------------------------------------

class TestSkillTool:
    def test_call_log(self) -> None:
        log = SkillTool(_ctx()).format_call_log({"name": "coding-conventions"})
        assert "skill=coding-conventions" in log

    def test_result_log(self) -> None:
        log = SkillTool(_ctx()).format_result_log("line1\nline2\nline3")
        assert "3 lines" in log


# ---------------------------------------------------------------------------
# TodoTool
# ---------------------------------------------------------------------------

class TestTodoTool:
    def test_call_log_add(self) -> None:
        log = TodoTool(_ctx(), _session_ctx().session).format_call_log(
            {"action": "add", "content": "Implement feature X"}
        )
        assert "action=add" in log
        assert "content=Implement feature X" in log

    def test_call_log_add_batch(self) -> None:
        log = TodoTool(_ctx(), _session_ctx().session).format_call_log(
            {"action": "add", "contents": ["task1", "task2"]}
        )
        assert "batch=2" in log

    def test_call_log_check(self) -> None:
        log = TodoTool(_ctx(), _session_ctx().session).format_call_log(
            {"action": "check", "id": 3}
        )
        assert "action=check" in log
        assert "ids=[3]" in log

    def test_call_log_insert(self) -> None:
        log = TodoTool(_ctx(), _session_ctx().session).format_call_log(
            {"action": "insert", "content": "new task", "index": 2, "parent_id": 1}
        )
        assert "index=2" in log
        assert "parent=1" in log


# ---------------------------------------------------------------------------
# FactTool
# ---------------------------------------------------------------------------

class TestFactTool:
    def test_call_log_add(self) -> None:
        log = FactTool(_ctx(), _session_ctx().session).format_call_log(
            {"action": "add", "content": "The sky is blue"}
        )
        assert "action=add" in log
        assert "content=" in log

    def test_call_log_add_batch(self) -> None:
        log = FactTool(_ctx(), _session_ctx().session).format_call_log(
            {"action": "add", "facts": [{"content": "a"}, {"content": "b"}]}
        )
        assert "batch=2" in log

    def test_call_log_remove(self) -> None:
        log = FactTool(_ctx(), _session_ctx().session).format_call_log(
            {"action": "remove", "ids": [1, 2]}
        )
        assert "action=remove" in log
        assert "ids=[1, 2]" in log


# ---------------------------------------------------------------------------
# DecisionTool
# ---------------------------------------------------------------------------

class TestDecisionTool:
    def test_call_log_add(self) -> None:
        log = DecisionTool(_ctx(), _session_ctx().session).format_call_log(
            {"action": "add", "content": "Use approach A"}
        )
        assert "action=add" in log
        assert "content=" in log

    def test_call_log_add_batch(self) -> None:
        log = DecisionTool(_ctx(), _session_ctx().session).format_call_log(
            {"action": "add", "contents": ["d1", "d2"]}
        )
        assert "batch=2" in log


# ---------------------------------------------------------------------------
# ConstraintTool
# ---------------------------------------------------------------------------

class TestConstraintTool:
    def test_call_log_add(self) -> None:
        log = ConstraintTool(_ctx(), _session_ctx().session).format_call_log(
            {"action": "add", "content": "Must respect X"}
        )
        assert "action=add" in log
        assert "content=" in log


# ---------------------------------------------------------------------------
# CheckpointTool
# ---------------------------------------------------------------------------

class TestCheckpointTool:
    def test_call_log_set(self) -> None:
        log = CheckpointTool(_ctx(), _session_ctx().session).format_call_log(
            {"action": "set", "content": "State description"}
        )
        assert "action=set" in log
        assert "content=" in log

    def test_call_log_clear(self) -> None:
        log = CheckpointTool(_ctx(), _session_ctx().session).format_call_log(
            {"action": "clear"}
        )
        assert "action=clear" in log
