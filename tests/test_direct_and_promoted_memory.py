import unittest
from types import SimpleNamespace

from citra.agent import AgentSession
from citra.tools.session_memory import (
    CheckpointTool,
    ConstraintTool,
    DecisionTool,
    FactTool,
    TodoTool,
    WorkingStateTool,
)


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = AgentSession()
        self.context = SimpleNamespace(
            config=SimpleNamespace(
                model=lambda: SimpleNamespace(id="test-model")
            )
        )
        self.working = WorkingStateTool(self.context, self.session)
        self.facts = FactTool(self.context, self.session)
        self.decisions = DecisionTool(self.context, self.session)
        self.constraints = ConstraintTool(self.context, self.session)
        self.todos = TodoTool(self.context, self.session)
        self.checkpoint = CheckpointTool(self.context, self.session)

    def test_direct_constraint_single_and_batch(self) -> None:
        self.constraints._execute({"action": "add", "content": "Keep API stable"})
        self.constraints._execute(
            {"action": "add", "contents": ["Python 3.12", "No network"]}
        )
        items = self.constraints.get_extracts()
        self.assertEqual([x.id for x in items], [1, 2, 3])
        self.assertTrue(all(x.working_state_id is None for x in items))
        self.assertNotIn("working state", self.constraints.format_extract(items[0]))

    def test_direct_decision(self) -> None:
        result = self.decisions._execute(
            {"action": "add", "content": "Keep the existing registry design"}
        )
        self.assertIn("Added DECISION [1]", result)
        self.assertIsNone(self.decisions.get_extracts()[0].working_state_id)

    def test_direct_fact_with_citation_and_batch(self) -> None:
        self.facts._execute(
            {
                "action": "add",
                "content": "clear_history defaults to true",
                "citations": [
                    {"type": "file", "source": "agent/session.py", "line": 20}
                ],
            }
        )
        self.facts._execute(
            {
                "action": "add",
                "facts": [
                    {"content": "Fact B"},
                    {"content": "Fact C", "citations": []},
                ],
            }
        )
        items = self.facts.get_extracts()
        self.assertEqual(len(items), 3)
        self.assertIsNone(items[0].working_state_id)
        rendered = self.facts.format_extract(items[0])
        self.assertIn("agent/session.py:20", rendered)
        self.assertNotIn("origin: working state", rendered)

    def test_direct_fact_rejects_promotion_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "working_state_id"):
            self.facts._execute(
                {"action": "add", "content": "x", "working_state_id": 1}
            )

    def test_direct_fact_citation_validation_is_preserved(self) -> None:
        with self.assertRaisesRegex(ValueError, "line.*at least 1"):
            self.facts._execute(
                {
                    "action": "add",
                    "content": "bad citation",
                    "citations": [
                        {"type": "file", "source": "a.py", "line": 0}
                    ],
                }
            )
        self.assertEqual(self.facts.get_extracts(), [])

    def test_direct_todo_add_insert_and_hierarchy(self) -> None:
        self.todos._execute({"action": "add", "content": "Parent"})
        self.todos._execute(
            {"action": "add", "contents": ["Child A", "Child C"], "parent_id": 1}
        )
        self.todos._execute(
            {"action": "insert", "content": "Child B", "parent_id": 1, "index": 1}
        )
        items = self.todos.get_extracts()
        self.assertEqual([x.content for x in items], ["Parent", "Child A", "Child B", "Child C"])
        self.assertTrue(all(x.working_state_id is None for x in items))
        self.assertEqual([x.parent_id for x in items], [None, 1, 1, 1])

    def test_direct_todo_descendant_completion_rule(self) -> None:
        self.todos._execute({"action": "add", "content": "Parent"})
        self.todos._execute({"action": "add", "content": "Child", "parent_id": 1})
        with self.assertRaisesRegex(ValueError, "descendants remain incomplete"):
            self.todos._execute({"action": "check", "id": 1})
        self.todos._execute({"action": "check", "ids": [2, 1]})
        self.assertFalse(self.todos.has_outstanding_todos())

    def test_direct_todo_reopens_completed_parent(self) -> None:
        self.todos._execute({"action": "add", "content": "Parent"})
        self.todos._execute({"action": "check", "id": 1})
        result = self.todos._execute(
            {"action": "add", "content": "New child", "parent_id": 1}
        )
        self.assertIn("reopened", result)
        self.assertFalse(self.todos.get_extracts()[0].completed)

    def test_promotion_still_works_for_all_durable_types(self) -> None:
        self.working._execute(
            {
                "action": "create",
                "contents": ["verified fact", "chosen design", "required invariant", "required work"],
            }
        )
        self.facts._execute({"action": "promote", "working_state_id": 1})
        self.decisions._execute({"action": "promote", "working_state_id": 2})
        self.constraints._execute({"action": "promote", "working_state_id": 3})
        self.todos._execute({"action": "promote", "working_state_id": 4})

        self.assertEqual(self.facts.get_extracts()[0].working_state_id, 1)
        self.assertEqual(self.decisions.get_extracts()[0].working_state_id, 2)
        self.assertEqual(self.constraints.get_extracts()[0].working_state_id, 3)
        self.assertEqual(self.todos.get_extracts()[0].working_state_id, 4)
        for working in self.working.get_extracts():
            self.assertEqual(len(working.promotions), 1)

    def test_one_working_state_can_promote_to_multiple_types(self) -> None:
        self.working._execute({"action": "create", "content": "Investigation result"})
        self.facts._execute(
            {"action": "promote", "working_state_id": 1, "content": "Verified result"}
        )
        self.todos._execute(
            {"action": "promote", "working_state_id": 1, "content": "Follow-up work"}
        )
        state = self.working.get_extracts()[0]
        self.assertEqual([(x.kind, x.memory_id) for x in state.promotions], [("fact", 1), ("todo", 1)])
        self.working._execute({"action": "resolve", "id": 1})
        self.assertEqual(self.working.get_extracts(), [])

    def test_direct_memory_does_not_create_fake_promotion(self) -> None:
        self.working._execute({"action": "create", "content": "Unrelated hypothesis"})
        self.facts._execute({"action": "add", "content": "Independent verified fact"})
        state = self.working.get_extracts()[0]
        self.assertEqual(state.promotions, ())
        self.working._execute({"action": "discard", "id": 1})

    def test_removing_promoted_memory_unregisters_provenance(self) -> None:
        self.working._execute({"action": "create", "content": "candidate"})
        self.constraints._execute({"action": "promote", "working_state_id": 1})
        self.constraints._execute({"action": "remove", "id": 1})
        state = self.working.get_extracts()[0]
        self.assertEqual(state.promotions, ())
        self.working._execute({"action": "discard", "id": 1})

    def test_removing_direct_memory_needs_no_working_state(self) -> None:
        self.decisions._execute({"action": "add", "content": "temporary decision"})
        self.decisions._execute({"action": "remove", "id": 1})
        self.assertEqual(self.decisions.get_extracts(), [])

    def test_batch_promotion_is_atomic_when_working_state_is_invalid(self) -> None:
        self.working._execute({"action": "create", "content": "valid"})
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.constraints._execute(
                {"action": "promote", "working_state_ids": [1, 999]}
            )
        self.assertEqual(self.constraints.get_extracts(), [])
        self.assertEqual(self.working.get_extracts()[0].promotions, ())

    def test_working_state_resolve_and_discard_semantics_preserved(self) -> None:
        self.working._execute({"action": "create", "contents": ["A", "B"]})
        with self.assertRaisesRegex(ValueError, "no promotions"):
            self.working._execute({"action": "resolve", "id": 1})
        self.working._execute({"action": "discard", "id": 1})
        self.facts._execute({"action": "promote", "working_state_id": 2})
        with self.assertRaisesRegex(ValueError, "durable promotions"):
            self.working._execute({"action": "discard", "id": 2})
        self.working._execute({"action": "resolve", "id": 2})

    def test_checkpoint_remains_independent(self) -> None:
        result = self.checkpoint._execute(
            {"action": "set", "content": "Files edited", "next_step": "Run tests"}
        )
        self.assertIn("Updated", result)
        extract = self.checkpoint.get_extracts()[0]
        self.assertEqual(extract.content, "Files edited")
        self.assertEqual(extract.turn, self.session.turn_number)
        self.assertEqual(extract.revision, 1)
        self.checkpoint._execute({"action": "clear"})
        self.assertEqual(self.checkpoint.get_extracts(), [])
        self.assertEqual(self.checkpoint.revision, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
