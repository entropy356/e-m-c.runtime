"""Context assembly tests (T11)."""

from __future__ import annotations

import unittest

from emc_runtime import Engine, EventBus, Event, Storage
from emc_runtime.cognition import make_cognition
from emc_runtime.modules import ModuleRegistry

from .helpers import RecordingProvider, TempDirTestCase, offline_config, spec_for


def build(config, provider):
    storage = Storage(config.database)
    engine = Engine(
        config=config,
        storage=storage,
        registry=ModuleRegistry(config.modules),
        provider=provider,
        cognition=make_cognition(config.cognition_kind, config.cognition_params),
        bus=EventBus(),
    )
    return engine, storage


class ContextOrderTest(TempDirTestCase):
    """T11: fixed context order; originals are never rewritten."""

    MODULES = [
        {
            "id": "counter",
            "entrypoint": "examples.counter_module:CounterModule",
            "scope": "agent",
            "config": {},
        },
    ]

    def test_order_and_sources(self):
        from .helpers import ChattyModule

        modules = self.MODULES + [spec_for(ChattyModule)]
        config = offline_config(self.tmp, modules=modules)
        provider = RecordingProvider()
        engine, storage = build(config, provider)
        engine.run(Event(kind="user_message", content="第一句"))
        engine.run(Event(kind="user_message", content="第二句"))

        messages = provider.calls[-1]
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["system", "system", "user", "assistant", "user"])
        # 1. base system prompt first
        self.assertIn("helpful conversational agent", messages[0]["content"])
        # 2. module data framed as data with explicit, ordered sources
        self.assertIn("reference data", messages[1]["content"])
        self.assertLess(
            messages[1]["content"].index("[source: counter]"),
            messages[1]["content"].index("[source: chatty]"),
        )
        # 3. history in original order, content unchanged
        self.assertEqual(messages[2]["content"], "第一句")
        self.assertTrue(messages[3]["content"].startswith("[offline]"))
        self.assertEqual(messages[4]["content"], "第二句")
        # 4. current input last
        self.assertEqual(messages[-1]["content"], "第二句")

    def test_no_modules_no_module_system_message(self):
        config = offline_config(self.tmp, modules=[])
        provider = RecordingProvider()
        engine, _ = build(config, provider)
        engine.run(Event(kind="user_message", content="hello"))
        messages = provider.calls[-1]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertNotIn("reference data", messages[0]["content"])  # no module message

    def test_history_integrity(self):
        config = offline_config(self.tmp, modules=self.MODULES)
        provider = RecordingProvider()
        engine, storage = build(config, provider)
        engine.run(Event(kind="user_message", content="原文 A"))
        engine.run(Event(kind="user_message", content="原文 B"))
        stored = [m.content for m in storage.history("default")]
        self.assertEqual(stored[::2], ["原文 A", "原文 B"])  # originals untouched
        context = provider.calls[-1]
        sent_history = [m["content"] for m in context][2:-1]
        self.assertEqual(sent_history, stored[:-2])  # context mirrors the stored history exactly


if __name__ == "__main__":
    unittest.main()
