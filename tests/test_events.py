"""Observation event tests (T14)."""

from __future__ import annotations

import unittest

from emc_runtime import Engine, EventBus, Event, Storage
from emc_runtime.cognition import make_cognition
from emc_runtime.modules import ModuleRegistry

from .helpers import OfflineProviderStub, TempDirTestCase, offline_config


class ObserverIsolationTest(TempDirTestCase):
    """T14: a crashing observer must not damage a committed turn."""

    def _run_once(self, bus):
        config = offline_config(self.tmp, modules=[])
        storage = Storage(config.database)
        engine = Engine(
            config=config,
            storage=storage,
            registry=ModuleRegistry(config.modules),
            provider=OfflineProviderStub(),
            cognition=make_cognition(config.cognition_kind, config.cognition_params),
            bus=bus,
        )
        return engine.run(Event(kind="user_message", content="hi")), storage

    def test_broken_observer_does_not_break_the_turn(self):
        received = []

        def broken(_event):
            raise RuntimeError("observer exploded")

        def healthy(event):
            received.append(event)

        bus = EventBus()
        bus.subscribe(broken)
        bus.subscribe(healthy)
        result, storage = self._run_once(bus)
        self.assertEqual(len(storage.history("default")), 2)  # committed anyway
        self.assertEqual(storage.revision(), 1)
        self.assertTrue(result.reply)
        # the healthy observer still ran after the broken one
        self.assertEqual(len(received), 1)  # exactly one turn_committed event
        committed = [e for e in received if e["type"] == "turn_committed"]
        self.assertEqual(len(committed), 1)
        self.assertEqual(committed[0]["conversation_id"], "default")
        self.assertEqual(committed[0]["revision"], 1)
        self.assertIn("reply", committed[0])
        self.assertEqual(committed[0]["event_kind"], "user_message")

    def test_publish_never_raises(self):
        bus = EventBus()
        bus.subscribe(lambda event: 1 / 0)
        bus.publish({"type": "turn_committed"})  # must not raise


if __name__ == "__main__":
    unittest.main()
