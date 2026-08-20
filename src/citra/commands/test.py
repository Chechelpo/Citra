# src/citra/commands/test.py

"""
``/test`` — verify that the current Citra setup is working.

The command runs a short sequence of checks and reports a pass/fail
summary for each. It exercises:

1. Configuration loading (already guaranteed by ExecutionContext, but
   reported explicitly).
2. Model API connectivity — sends a minimal chat-completions request
   and checks for a valid ``choices`` array.
3. Web-search (SearXNG) connectivity — issues a trivial query and
   checks for an HTTP 200 / JSON response.
4. Bash availability — confirms ``bash`` is on PATH.
5. Workspace access — confirms the configured workspace exists and is
   readable.

Each check is independent: a failure in one does not prevent the
others from running.
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..utils.api import chat_completions_url
from ..utils.terminal import BOLD, DIM, GREEN, RED, RESET
from .command import Command, CommandResult


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CheckRunner:
    results: list[CheckResult] = field(default_factory=list)

    def run(self, name: str, fn: Any) -> None:
        try:
            detail = fn() or ""
            self.results.append(
                CheckResult(name=name, passed=True, detail=detail)
            )
        except Exception as error:  # noqa: BLE001
            self.results.append(
                CheckResult(name=name, passed=False, detail=str(error))
            )

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def render(self) -> str:
        lines: list[str] = []

        for result in self.results:
            icon = f"{GREEN}✓{RESET}" if result.passed else f"{RED}✗{RESET}"
            detail = f" {DIM}— {result.detail}{RESET}" if result.detail else ""
            lines.append(f"  {icon} {result.name}{detail}")

        lines.append("")

        if self.all_passed:
            lines.append(f"{GREEN}⏺ All checks passed.{RESET}")
        else:
            failed = sum(1 for r in self.results if not r.passed)
            lines.append(
                f"{RED}⏺ {failed} check(s) failed.{RESET}"
            )

        return "\n".join(lines)


class TestCommand(Command):
    """Test the current setup and report whether it is working."""

    id = "test"
    description = "Run diagnostics on the current Citra setup."

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_config(self) -> str:
        model = self.context.model_config
        web = self.context.web_search_config

        return (
            f"model={model.id}, "
            f"searxng={web.host_url}"
        )

    def _check_model_api(self) -> str:
        model = self.context.model_config

        payload: dict[str, Any] = {
            "model": model.id,
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with the single word: ok",
                }
            ],
        }

        request = urllib.request.Request(
            chat_completions_url(model.host),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {model.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))

        choices = body.get("choices", [])

        if not isinstance(choices, list) or not choices:
            raise RuntimeError(
                "Model API returned no choices in the response."
            )

        message = choices[0].get("message", {})
        content = message.get("content", "")

        text = str(content).strip() if content else "(empty)"

        if len(text) > 60:
            text = text[:60] + "..."

        return f"replied: {text!r}"

    def _check_web_search(self) -> str:
        from urllib.parse import urlencode

        host = self.context.web_search_config.host_url.rstrip("/")
        params = urlencode(
            {
                "q": "test",
                "format": "json",
                "pageno": 1,
            }
        )

        request = urllib.request.Request(
            f"{host}/search?{params}",
            headers={
                "Accept": "application/json",
                "User-Agent": "Citra/1.0",
            },
            method="GET",
        )

        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read()

        payload = json.loads(raw)

        if not isinstance(payload, dict):
            raise RuntimeError("SearXNG returned a non-object JSON response.")

        results = payload.get("results", [])

        return f"{len(results)} result(s)"

    def _check_bash(self) -> str:
        path = shutil.which("bash")

        if not path:
            raise RuntimeError("bash not found on PATH.")

        return path

    def _check_workspace(self) -> str:
        import os

        workspace = self.context.workspace

        if not os.path.isdir(workspace):
            raise RuntimeError(
                f"Workspace directory does not exist: {workspace}"
            )

        if not os.access(workspace, os.R_OK):
            raise RuntimeError(
                f"Workspace is not readable: {workspace}"
            )

        return workspace

    # ------------------------------------------------------------------
    # Command entry point
    # ------------------------------------------------------------------

    def _run(self, args: str) -> CommandResult:
        runner = CheckRunner()

        runner.run("Config", self._check_config)
        runner.run("Model API", self._check_model_api)
        runner.run("Web Search (SearXNG)", self._check_web_search)
        runner.run("Bash", self._check_bash)
        runner.run("Workspace", self._check_workspace)

        header = f"{BOLD}Running diagnostics…{RESET}\n\n"

        return CommandResult(
            output=header + runner.render(),
        )
