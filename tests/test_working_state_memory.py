import unittest

from citra.agent import AgentSession
from citra.context import ExecutionContext
from citra.tools.session_memory.working_state_tool import WorkingStateTool
from citra.tools.session_memory.fact_tool import FactTool
from citra.tools.session_memory.todo_tool import TodoTool
from citra.tools.session_memory.constraint_tool import ConstraintTool
from citra.tools.session_memory.decision_tool import DecisionTool
from citra.tools.session_memory.checkpoint_tool import CheckpointTool


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.session = AgentSession()
        self.context = ExecutionContext()
        self.working = WorkingStateTool(self.context, self.session)
        self.facts = FactTool(self.context, self.session)
        self.todos = TodoTool(self.context, self.session)
        self.constraints = ConstraintTool(self.context, self.session)
        self.decisions = DecisionTool(self.context, self.session)
        self.checkpoint = CheckpointTool(self.context, self.session)

    def create(self, content):
        self.working._execute({"action": "create", "content": content})
        return self.working.get_extracts()[-1].id

    def test_durable_memory_requires_working_state(self):
        with self.assertRaises(ValueError):
            self.facts._execute({"action": "promote", "working_state_id": 999})
        with self.assertRaises(ValueError):
            self.todos._execute({"action": "promote", "working_state_id": 999})
        with self.assertRaises(ValueError):
            self.constraints._execute({"action": "promote", "working_state_id": 999})
        with self.assertRaises(ValueError):
            self.decisions._execute({"action": "promote", "working_state_id": 999})

    def test_one_working_state_can_promote_to_multiple_memory_types(self):
        wid = self.create("Command 102 appears to own displayed choices; verify runtime semantics.")
        self.facts._execute({
            "action": "promote",
            "working_state_id": wid,
            "content": "Command 102 owns the displayed choice list.",
            "citations": [{"type": "file", "source": "runtime.rb", "line": 10, "end_line": 12}],
        })
        self.todos._execute({
            "action": "promote",
            "working_state_id": wid,
            "content": "Move choice extraction to command 102.",
        })
        active = self.working.get_extracts()[0]
        self.assertEqual({(x.kind, x.memory_id) for x in active.promotions}, {("fact", 1), ("todo", 1)})
        self.working._execute({"action": "resolve", "id": wid})
        self.assertEqual(self.working.get_extracts(), [])
        with self.assertRaises(ValueError):
            self.constraints._execute({"action": "promote", "working_state_id": wid})

    def test_remove_unregisters_promotion_then_discard_is_allowed(self):
        wid = self.create("Tentative repository invariant")
        self.constraints._execute({"action": "promote", "working_state_id": wid})
        with self.assertRaises(ValueError):
            self.working._execute({"action": "discard", "id": wid})
        self.constraints._execute({"action": "remove", "id": 1})
        self.working._execute({"action": "discard", "id": wid})
        self.assertEqual(self.working.get_extracts(), [])

    def test_update_changes_default_promoted_content(self):
        wid = self.create("maybe true")
        self.session.turn_number = 2
        self.working._execute({"action": "update", "id": wid, "content": "verified behavior"})
        self.facts._execute({"action": "promote", "working_state_id": wid})
        fact = self.facts.get_extracts()[0]
        self.assertEqual(fact.content, "verified behavior")
        self.assertEqual(fact.working_state_id, wid)

    def test_resolve_without_promotion_is_rejected(self):
        wid = self.create("Unresolved hypothesis")
        with self.assertRaises(ValueError):
            self.working._execute({"action": "resolve", "id": wid})
        self.working._execute({"action": "discard", "id": wid})

    def test_todo_hierarchy_requires_descendants_completed(self):
        parent_w = self.create("Implement validator")
        self.todos._execute({"action": "promote", "working_state_id": parent_w})
        child_w = self.create("Add control-token tests")
        self.todos._execute({"action": "promote", "working_state_id": child_w, "parent_id": 1})
        with self.assertRaises(ValueError):
            self.todos._execute({"action": "check", "id": 1})
        self.todos._execute({"action": "check", "id": 2})
        self.todos._execute({"action": "check", "id": 1})
        self.assertFalse(self.todos.has_outstanding_todos())

    def test_promoting_child_reopens_completed_parent(self):
        parent_w = self.create("Audit extraction")
        self.todos._execute({"action": "promote", "working_state_id": parent_w})
        self.todos._execute({"action": "check", "id": 1})
        child_w = self.create("Audit System.rvdata2")
        result = self.todos._execute({"action": "promote", "working_state_id": child_w, "parent_id": 1})
        self.assertIn("reopened", result)
        self.assertFalse(self.todos.get_extracts()[0].completed)

    def test_batch_promotion_is_atomic_on_invalid_working_state(self):
        w1 = self.create("First")
        with self.assertRaises(ValueError):
            self.todos._execute({"action": "promote", "working_state_ids": [w1, 999]})
        self.assertEqual(self.todos.get_extracts(), [])
        self.assertEqual(self.working.get_extracts()[0].promotions, ())

    def test_remove_parent_unregisters_entire_subtree(self):
        w1 = self.create("Parent")
        w2 = self.create("Child")
        self.todos._execute({"action": "promote", "working_state_id": w1})
        self.todos._execute({"action": "promote", "working_state_id": w2, "parent_id": 1})
        self.todos._execute({"action": "remove", "id": 1})
        self.working._execute({"action": "discard", "ids": [w1, w2]})
        self.assertEqual(self.working.get_extracts(), [])

    def test_checkpoint_remains_independent(self):
        self.checkpoint._execute({"action": "set", "content": "Tests green", "next_step": "Commit"})
        item = self.checkpoint.get_extracts()[0]
        self.assertEqual(item.content, "Tests green")
        self.assertEqual(item.next_step, "Commit")
        self.assertEqual(item.revision, 1)
        self.assertEqual(self.checkpoint.revision, 1)
        self.checkpoint._execute({"action": "clear"})
        self.assertEqual(self.checkpoint.revision, 2)

    def test_direct_add_actions_are_rejected(self):
        for tool in (self.facts, self.todos, self.constraints, self.decisions):
            with self.assertRaises(ValueError):
                tool._execute({"action": "add", "content": "bypass"})

    def test_fact_citation_validation(self):
        wid = self.create("Fact")
        with self.assertRaises(ValueError):
            self.facts._execute({
                "action": "promote",
                "working_state_id": wid,
                "citations": [{"type": "url", "source": "https://example.com", "line": 2}],
            })
        self.assertEqual(self.facts.get_extracts(), [])
        self.assertEqual(self.working.get_extracts()[0].promotions, ())

    def test_batch_working_state_create_and_simple_promotions(self):
        self.working._execute({"action": "create", "contents": ["Choose A", "Require B", "Know C"]})
        ids = [item.id for item in self.working.get_extracts()]
        self.assertEqual(ids, [1, 2, 3])
        self.decisions._execute({"action": "promote", "working_state_id": 1, "content": "Use A"})
        self.constraints._execute({"action": "promote", "working_state_id": 2, "content": "B must remain true"})
        self.facts._execute({"action": "promote", "working_state_id": 3, "content": "C is verified"})
        self.assertEqual(self.decisions.get_extracts()[0].working_state_id, 1)
        self.assertEqual(self.constraints.get_extracts()[0].working_state_id, 2)
        self.assertEqual(self.facts.get_extracts()[0].working_state_id, 3)

    def test_batch_fact_promotion_validates_before_mutation(self):
        w1 = self.create("First fact")
        with self.assertRaises(ValueError):
            self.facts._execute({
                "action": "promote",
                "facts": [
                    {"working_state_id": w1},
                    {"working_state_id": 999},
                ],
            })
        self.assertEqual(self.facts.get_extracts(), [])
        self.assertEqual(self.working.get_extracts()[0].promotions, ())

    def test_working_state_cannot_be_updated_after_resolution(self):
        wid = self.create("Decision basis")
        self.decisions._execute({"action": "promote", "working_state_id": wid})
        self.working._execute({"action": "resolve", "id": wid})
        with self.assertRaises(ValueError):
            self.working._execute({"action": "update", "id": wid, "content": "changed"})

    def test_todo_single_insert_position(self):
        w1 = self.create("First")
        w2 = self.create("Second")
        w3 = self.create("Inserted")
        self.todos._execute({"action": "promote", "working_state_id": w1})
        self.todos._execute({"action": "promote", "working_state_id": w2})
        self.todos._execute({"action": "promote", "working_state_id": w3, "index": 1})
        self.assertEqual([x.content for x in self.todos.get_extracts()], ["First", "Inserted", "Second"])


if __name__ == "__main__":
    unittest.main()
