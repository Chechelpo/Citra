from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from citra.agent import AgentSession
from citra.agent.conversation_memory import ConversationMemory
from citra.application import CitraApplication
from citra.tools.default_registry import memory_tools
from citra.tools.session_memory import (
    AcceptanceCriteriaTool,
    RequirementTool,
    VerificationTool,
)
from citra.tools.tool import Tool
from citra.tools.tool_registry import ToolRegistry
from citra.utils.chat_completions_api import build_memory_context
from citra.workflows import SerialRolesWorkflow


def _context(workflow: SerialRolesWorkflow) -> SimpleNamespace:
    """Build the minimum execution context required by memory tools."""
    return SimpleNamespace(
        config=SimpleNamespace(
            memory=SimpleNamespace(enabled=True),
            model=lambda: SimpleNamespace(id="test-model"),
        ),
        workflow_runtime=SimpleNamespace(workflow=workflow),
    )


def _phase_tools(
    workflow: SerialRolesWorkflow,
    phase: str,
    memory: ConversationMemory,
) -> tuple[AgentSession, dict[str, Tool]]:
    """Instantiate one fresh role session over shared task memory."""
    run = workflow.create_run("task")
    route = ("explore", "plan", "implement", "test", "review")
    for next_step in route[1 : route.index(phase) + 1]:
        run.begin_step()
        run.submit_handoff(summary=f"Advance to {next_step}", next_step=next_step)
        run.advance()
    session = AgentSession(memory=memory, memory_enabled=True)
    memory_ids = {tool.TOOL_ID for tool in memory_tools()}
    tools = ToolRegistry(run.current_step.workflow.tool_set).instantiate(
        cast(Any, _context(workflow)),
        session,
        tool_ids=memory_ids,
    )
    return session, tools


def test_serial_memory_carries_contract_changes_evidence_and_findings() -> None:
    """Exercise the R-A-TODO-CH-V/I handoff across fresh role sessions."""
    workflow = SerialRolesWorkflow()
    memory = ConversationMemory()

    explore_session, explore = _phase_tools(workflow, "explore", memory)
    explore["requirement"].execute(
        {"action": "add", "content": "Export every matching transaction"}
    )
    explore["acceptance_criteria"].execute(
        {
            "action": "add",
            "content": "The export contains every matching transaction",
            "requirement_ids": [1],
        }
    )

    plan_session, plan = _phase_tools(workflow, "plan", memory)
    assert plan_session is not explore_session
    assert "requirement" not in plan
    plan_context = build_memory_context(plan_session.memory.values())
    assert plan_context is not None
    assert "[R1] Export every matching transaction" in plan_context
    assert "[A1] The export contains every matching transaction" in plan_context
    plan["todo"].execute(
        {
            "action": "add",
            "content": "Implement and test complete export selection",
            "requirement_ids": [1],
            "acceptance_criterion_ids": [1],
        }
    )

    _implement_session, implement = _phase_tools(workflow, "implement", memory)
    implement["todo"].execute({"action": "check", "id": 1})
    implement["change"].execute(
        {
            "action": "record",
            "summary": "Export now retains all matching transactions",
            "paths": ["src/export.py", "tests/test_export.py"],
            "todo_ids": [1],
            "requirement_ids": [1],
            "acceptance_criterion_ids": [1],
        }
    )

    test_session, test = _phase_tools(workflow, "test", memory)
    test["issue"].execute(
        {
            "action": "add",
            "kind": "defect",
            "severity": "high",
            "content": "Duplicate identifiers drop one matching transaction",
            "route": "implement",
            "evidence": "Focused export test failed for duplicate identifiers",
            "requirement_ids": [1],
            "acceptance_criterion_ids": [1],
            "change_ids": [1],
        }
    )
    test["verification"].execute(
        {
            "action": "record",
            "check": "Complete export selection",
            "status": "failed",
            "evidence": "Expected 2 transactions, received 1",
            "command": "pytest tests/test_export.py -q",
            "requirement_ids": [1],
            "acceptance_criterion_ids": [1],
            "change_ids": [1],
            "issue_ids": [1],
        }
    )

    _review_session, review = _phase_tools(workflow, "review", memory)
    review["requirement"].execute(
        {"action": "satisfy", "id": 1, "evidence": "Reviewed behavior"}
    )
    review["acceptance_criteria"].execute(
        {"action": "satisfy", "id": 1, "evidence": "Reviewed behavior"}
    )
    application = CitraApplication.__new__(CitraApplication)
    application.session = test_session
    assert "verification" in application._memory_completion_error()

    _implement_session, implement = _phase_tools(workflow, "implement", memory)
    implement["change"].execute(
        {
            "action": "update",
            "id": 1,
            "notes": "Duplicate identifiers now remain distinct",
        }
    )
    _test_session, test = _phase_tools(workflow, "test", memory)
    test["verification"].execute(
        {
            "action": "record",
            "check": "Complete export selection",
            "status": "passed",
            "evidence": "Focused export suite passed",
            "command": "pytest tests/test_export.py -q",
            "requirement_ids": [1],
            "acceptance_criterion_ids": [1],
            "change_ids": [1],
            "issue_ids": [1],
            "supersedes_ids": [1],
        }
    )
    assert "blocking issues" in application._memory_completion_error()

    _implement_session, implement = _phase_tools(workflow, "implement", memory)
    implement["issue"].execute(
        {
            "action": "resolve",
            "id": 1,
            "resolution": "Selection now preserves duplicate identifiers",
        }
    )

    assert application._memory_completion_error() is None
    final_context = build_memory_context(memory.values())
    assert final_context is not None
    assert "[CH1@r2]" in final_context
    assert "[RESOLVED] [I1]" in final_context
    assert "[PASSED/ACTIVE] [V2]" in final_context
    assert "CH1@r2" in final_context


def test_cross_memory_links_reject_dangling_ids() -> None:
    """Prevent later roles from silently linking records that do not exist."""
    workflow = SerialRolesWorkflow()
    memory = ConversationMemory()
    _session, explore = _phase_tools(workflow, "explore", memory)

    with pytest.raises(ValueError, match="missing requirement IDs: 99"):
        explore["acceptance_criteria"].execute(
            {
                "action": "add",
                "content": "Observable result",
                "requirement_ids": [99],
            }
        )


def test_reviewer_reopens_invalidated_contract_evidence() -> None:
    """Keep satisfaction state synchronized when review evidence changes."""
    workflow = SerialRolesWorkflow()
    memory = ConversationMemory()
    _session, explore = _phase_tools(workflow, "explore", memory)
    explore["requirement"].execute({"action": "add", "content": "Preserve API"})
    explore["acceptance_criteria"].execute(
        {
            "action": "add",
            "content": "Compatibility suite passes",
            "requirement_ids": [1],
        }
    )
    _session, review = _phase_tools(workflow, "review", memory)
    review["requirement"].execute(
        {"action": "satisfy", "id": 1, "evidence": "Suite passed"}
    )
    review["acceptance_criteria"].execute(
        {"action": "satisfy", "id": 1, "evidence": "Suite passed"}
    )

    review["requirement"].execute({"action": "reopen", "id": 1})
    review["acceptance_criteria"].execute({"action": "reopen", "id": 1})

    requirements = memory.get(RequirementTool.TOOL_ID)
    criteria = memory.get(AcceptanceCriteriaTool.TOOL_ID)
    assert isinstance(requirements, RequirementTool)
    assert isinstance(criteria, AcceptanceCriteriaTool)
    assert requirements.has_unsatisfied_requirements()
    assert criteria.has_unsatisfied_criteria()
    assert requirements.get_extracts()[0].evidence is None
    assert criteria.get_extracts()[0].evidence is None


def test_change_revision_makes_linked_verification_stale() -> None:
    """Require a retest when implementation changes after passing evidence."""
    workflow = SerialRolesWorkflow()
    memory = ConversationMemory()
    _session, implement = _phase_tools(workflow, "implement", memory)
    implement["change"].execute(
        {
            "action": "record",
            "summary": "Implement export",
            "paths": ["src/export.py"],
        }
    )
    _session, test = _phase_tools(workflow, "test", memory)
    test["verification"].execute(
        {
            "action": "record",
            "check": "Export behavior",
            "status": "passed",
            "evidence": "Focused test passed",
            "change_ids": [1],
        }
    )
    verification = memory.get(VerificationTool.TOOL_ID)
    assert isinstance(verification, VerificationTool)
    assert not verification.has_blocking_results()

    _session, implement = _phase_tools(workflow, "implement", memory)
    implement["change"].execute(
        {"action": "update", "id": 1, "notes": "Adjusted edge-case handling"}
    )

    assert verification.has_blocking_results()
    assert "[PASSED/STALE-CHANGE]" in verification.format_for_llm()
