"""User-facing /lsp lifecycle, status, and explicit installer command."""

from __future__ import annotations

import shlex
from typing import Any

from citra.utils.lsp import LspManager
from .command import Command, CommandResult


class LspCommand(Command):
    id = "lsp"
    description = "Inspect, install, restart, or stop optional language servers."

    def _run(self, args: str) -> CommandResult:
        manager = self.context.lsp_manager
        if not isinstance(manager, LspManager):
            return CommandResult(output="LSP services are unavailable in this execution context.")

        tokens = shlex.split(args)
        if not tokens or tokens == ["status"]:
            return CommandResult(output=self._format_status(manager.status()))

        action = tokens[0].casefold()
        if action == "install":
            return CommandResult(output=self._install(manager, tokens[1:]))
        if action in {"restart", "stop"}:
            if len(tokens) > 2:
                raise ValueError(f"Usage: /lsp {action} [server]")
            target = tokens[1] if len(tokens) == 2 else None
            count = manager.restart(target) if action == "restart" else manager.stop(target)
            verb = "restarted" if action == "restart" else "stopped"
            return CommandResult(output=f"{verb}: {count} language-server instance(s)")

        raise ValueError(
            "Usage: /lsp [status|install <server|language|missing|all> [--dry-run]|"
            "restart [server]|stop [server]]"
        )

    def _install(self, manager: LspManager, tokens: list[str]) -> str:
        if not tokens:
            raise ValueError("Usage: /lsp install <server|language|missing|all> [--dry-run]")
        dry_run = False
        targets: list[str] = []
        for token in tokens:
            if token == "--dry-run":
                dry_run = True
            elif token.startswith("-"):
                raise ValueError(f"Unknown /lsp install option: {token}")
            else:
                targets.append(token)
        if len(targets) != 1:
            raise ValueError("/lsp install requires exactly one target.")

        results = manager.install(targets[0], dry_run=dry_run)
        lines: list[str] = []
        skipped: list[str] = []
        for result in results:
            if result.command is None:
                if result.returncode == 0:
                    lines.append(f"{result.server_id}: {result.output}")
                else:
                    skipped.append(f"- {result.server_id}: {result.output}")
                continue
            lines.append(f"{result.server_id}: $ {' '.join(result.command)}")
            if result.dry_run:
                lines.append("  dry-run: not executed")
                continue
            lines.append(f"  exit: {result.returncode}")
            if result.output:
                lines.extend(f"  {line}" for line in result.output.splitlines()[-20:])
            if result.executable_found:
                lines.append(f"  executable: {result.executable_found}")
            else:
                lines.append("  executable: still missing after installation")
        if skipped:
            if lines:
                lines.append("")
            lines.append("Skipped:")
            lines.extend(skipped)
        return "\n".join(lines) or "No installable language servers selected."

    @staticmethod
    def _format_status(status: dict[str, Any]) -> str:
        if not status.get("enabled", True):
            prefix = "LSP: disabled\n"
        else:
            prefix = "LSP: enabled\n"
        lines = [prefix.rstrip()]
        servers = status.get("servers", [])
        if not isinstance(servers, list):
            return prefix.rstrip()
        for item in servers:
            if not isinstance(item, dict):
                continue
            languages = "/".join(str(value) for value in item.get("languages", []))
            installed = "installed" if item.get("installed") else "missing"
            executable = item.get("executable") or "-"
            running = int(item.get("running", 0) or 0)
            method = item.get("installation_method")
            line = (
                f"{str(item.get('id', '?')):<14} {languages:<24} {installed:<9} "
                f"{executable}   running: {running}"
            )
            if method:
                line += f"   install: {method}"
            lines.append(line)
            optional = item.get("optional_dependencies")
            if isinstance(optional, dict):
                requirements = optional.get("requirements")
                if isinstance(requirements, dict):
                    missing = [name for name, path in requirements.items() if not path]
                    if missing:
                        lines.append("  missing dependencies: " + ", ".join(missing))
                if item.get("id") == "vue":
                    if not optional.get("typescript_bridge"):
                        lines.append("  Vue bridge: typescript-language-server missing")
                    if not optional.get("vue_typescript_plugin"):
                        lines.append("  Vue bridge: @vue/typescript-plugin missing")
                if item.get("id") == "jdtls" and optional.get("java_compatible") is False:
                    major = optional.get("java_major")
                    detail = f"Java {major}" if isinstance(major, int) else "unknown Java version"
                    lines.append(f"  JDTLS runtime: {detail}; Java 21+ required")
        return "\n".join(lines)
