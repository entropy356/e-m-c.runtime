import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from long_term_memory.model import DynamicsConfig
from long_term_memory.providers import OfflineProvider
from long_term_memory.runtime import Runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "agent.sqlite3"
        self.provider = OfflineProvider()
        self.config = DynamicsConfig(
            decay_rate=0.1,
            stability_protection=0.8,
            recovery_rate=0.5,
            stability_gain=0.2,
            top_k=5,
        )
        self.runtime = Runtime(self.path, self.provider, self.config)

    @staticmethod
    def result(answer="test answer", used=None, memories=None):
        return {
            "answer": answer,
            "used_memory_ids": [] if used is None else used,
            "memories": [] if memories is None else memories,
        }

    def database_dump(self):
        # closing() closes the connection; the sqlite3 context manager only commits,
        # and a leaked handle makes TemporaryDirectory cleanup fail on Windows.
        with closing(sqlite3.connect(self.path)) as db:
            return list(db.iterdump())

    def test_complete_ordered_history_includes_user_and_assistant_messages(self):
        responses = [
            self.result("first assistant response"),
            self.result("second assistant response"),
        ]
        captured = []

        def cognize(messages, memories):
            captured.append((messages, memories))
            return responses[len(captured) - 1]

        with patch.object(self.provider, "cognize", side_effect=cognize):
            self.runtime.turn("first user message")
            self.runtime.turn("second user message")

        expected = [
            {"role": "user", "content": "first user message"},
            {"role": "assistant", "content": "first assistant response"},
            {"role": "user", "content": "second user message"},
        ]
        self.assertEqual(captured[1][0], expected)
        history = self.runtime.history()
        self.assertEqual([item["role"] for item in history],
                         ["user", "assistant", "user", "assistant"])
        self.assertEqual([item["seq"] for item in history], [1, 2, 3, 4])
        self.assertEqual([item["content"] for item in history], [
            "first user message", "first assistant response",
            "second user message", "second assistant response",
        ])

    def test_conversation_histories_are_isolated_but_memory_is_shared(self):
        proposal = [{"content": "User prefers concise written updates."}]
        with patch.object(self.provider, "cognize",
                          return_value=self.result(memories=proposal)):
            self.runtime.turn("Please remember that I prefer concise written updates.", "alpha")

        captured = {}

        def second_cognition(messages, memories):
            captured["messages"] = messages
            captured["memories"] = memories
            return self.result("I have the preference available.")

        with patch.object(self.provider, "cognize", side_effect=second_cognition):
            self.runtime.turn("What communication style do I prefer?", "beta")

        self.assertEqual(captured["messages"], [
            {"role": "user", "content": "What communication style do I prefer?"}
        ])
        self.assertEqual(len(captured["memories"]), 1)
        self.assertEqual(captured["memories"][0]["content"],
                         "User prefers concise written updates.")
        self.assertEqual(len(self.runtime.history("alpha")), 2)
        self.assertEqual(len(self.runtime.history("beta")), 2)

    def test_only_memories_reported_as_used_are_reinforced(self):
        proposal = [{"content": "The spare key is in the blue tin."}]
        with patch.object(self.provider, "cognize",
                          return_value=self.result(memories=proposal)):
            self.runtime.turn("Remember where the spare key is.")

        before = self.runtime.inspect()["state"]["memories"][0]["weight"]
        with patch.object(self.provider, "cognize",
                          return_value=self.result(used=[1])):
            trace = self.runtime.turn("Where is the spare key?")
        memory = self.runtime.inspect()["state"]["memories"][0]
        decayed_weight = trace["decayed"][0]["weight_after"]
        self.assertEqual(trace["activated"][0]["id"], 1)
        self.assertGreater(memory["weight"], decayed_weight)
        self.assertLess(memory["weight"], before)
        self.assertGreater(memory["stability"], 0)
        self.assertEqual(memory["activated_count"], 1)

    def test_duplicate_memory_proposal_is_not_written_twice(self):
        proposal = [{"content": "User prefers short answers."}]
        with patch.object(self.provider, "cognize",
                          return_value=self.result(memories=proposal)):
            self.runtime.turn("I prefer short answers.")
            trace = self.runtime.turn("As I said, I prefer short answers.")
        self.assertEqual(len(self.runtime.inspect()["state"]["memories"]), 1)
        self.assertEqual(trace["written_ids"], [])

    def test_failed_cognition_does_not_persist_messages_or_state(self):
        before = self.database_dump()
        with patch.object(self.provider, "cognize",
                          return_value=self.result(used=[999])):
            with self.assertRaises(ValueError):
                self.runtime.turn("This turn should fail.")
        self.assertEqual(before, self.database_dump())
        self.assertEqual(self.runtime.history(), [])

    def test_embedding_failure_after_cognition_does_not_commit(self):
        before = self.database_dump()
        original = self.provider.embed
        calls = 0

        def fail_on_memory_write(text, task=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("embedding unavailable")
            return original(text, task)

        with patch.object(self.provider, "cognize",
                          return_value=self.result(memories=[{"content": "A durable fact."}])):
            with patch.object(self.provider, "embed", side_effect=fail_on_memory_write):
                with self.assertRaises(ValueError):
                    self.runtime.turn("Remember a durable fact.")
        self.assertEqual(before, self.database_dump())
        self.assertEqual(self.runtime.history(), [])

    def test_restart_preserves_history_memories_and_trace(self):
        with patch.object(self.provider, "cognize",
                          return_value=self.result(memories=[{"content": "A persistent preference."}])):
            self.runtime.turn("I have a persistent preference.")
        expected = self.runtime.inspect()
        reopened = Runtime(self.path, self.provider, self.config).inspect()
        self.assertEqual(expected, reopened)

    def test_reading_history_and_state_does_not_advance_runtime(self):
        self.runtime.turn("Hello.")
        before = self.database_dump()
        self.runtime.history()
        self.runtime.inspect()
        self.assertEqual(before, self.database_dump())

    def test_database_rejects_changed_embedding_profile_or_dynamics(self):
        other = OfflineProvider()
        other.profile = "another-embedding-space"
        with self.assertRaises(ValueError):
            Runtime(self.path, other, self.config)
        with self.assertRaises(ValueError):
            Runtime(self.path, self.provider, DynamicsConfig(decay_rate=0.3))


if __name__ == "__main__":
    unittest.main()
