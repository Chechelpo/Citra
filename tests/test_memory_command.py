"""Tests for on-demand conversation-memory inspection."""

from types import SimpleNamespace

from citra.agent import AgentSession
from citra.commands.default_registry import COMMAND_REGISTRY
from citra.commands.memory import MemoryCommand


def test_memory_command_is_registered() -> None:
    assert COMMAND_REGISTRY.contains("memory")


def test_memory_command_reports_empty_session() -> None:
    context = SimpleNamespace(session=AgentSession())
    result = MemoryCommand(context).run("")
    assert "# Conversation Memory" in result.output
    assert "(empty)" in result.output


def test_memory_command_rejects_unknown_arguments() -> None:
    context = SimpleNamespace(session=AgentSession())
    result = MemoryCommand(context).run("clear")
    assert "Unknown memory action" in result.output
    assert result.usage == (MemoryCommand.usage,)
