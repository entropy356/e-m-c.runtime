"""Module protocol tests (T03-T05, T13)."""

from __future__ import annotations

import unittest

from emc_runtime import EMCError, Engine, EventBus, ModuleStateVersionError, Storage
from emc_runtime.cognition import make_cognition
from emc_runtime.modules import ModuleRegistry, scope_key_for

from .helpers import OfflineProviderStub, TempDirTestCase, offline_config, spec_for


def make_engine(config, provider=None):
    storage = Storage(config.database)
    registry = ModuleRegistry(config.modules)
    return (
        Engine(
            config=config,
            storage=storage,
            registry=registry,
            provider=provider or OfflineProviderStub(),
            cognition=make_cognition(config.cognition_kind, config.cognition_params),
            bus=EventBus(),
        ),
        storage,
    )


class AgentScopeCounterTest(TempDirTestCase):
    """T03: agent-scope state is shared across conversations."""

    def test_counter_shared_across_conversations(self):
        config = offline_config(
            self.tmp,
            modules=[
                {
                    "id": "counter",
                    "entrypoint": "examples.counter_module:CounterModule",
                    "scope": "agent",
                    "config": {},
                }
            ],
        )
        engine, storage = make_engine(config)
        engine.run(event("user_message", "a", "conv1"))
        engine.run(event("user_message", "b", "conv2"))
        engine.run(event("user_message", "c", "conv1"))
        state = storage.module_state("counter", scope_key_for("agent", "ignored"))
        self.assertEqual(state.state, {"count": 3})
        # the same state is visible from every conversation
        for conversation in ("conv1", "conv2"):
            entry = storage.module_state("counter", scope_key_for("agent", conversation))
            self.assertEqual(entry.state["count"], 3)


class ConversationScopeCounterTest(TempDirTestCase):
    """T04: conversation-scope state is isolated per conversation."""

    def test_counter_isolated_per_conversation(self):
        config = offline_config(
            self.tmp,
            modules=[
                {
                    "id": "counter",
                    "entrypoint": "examples.counter_module:CounterModule",
                    "scope": "conversation",
                    "config": {},
                }
            ],
        )
        engine, storage = make_engine(config)
        engine.run(event("user_message", "a", "conv1"))
        engine.run(event("user_message", "b", "conv1"))
        engine.run(event("user_message", "c", "conv2"))
        self.assertEqual(
            storage.module_state("counter", "conv1").state, {"count": 2}
        )
        self.assertEqual(
            storage.module_state("counter", "conv2").state, {"count": 1}
        )


class ModuleSwapTest(TempDirTestCase):
    """T05: enabling/disabling modules is a pure configuration change."""

    def test_disable_preserves_state_and_reenable_continues(self):
        counter = {
            "id": "counter",
            "entrypoint": "examples.counter_module:CounterModule",
            "scope": "agent",
            "config": {},
        }
        with_counter = offline_config(self.tmp, modules=[counter])
        engine, _ = make_engine(with_counter)
        engine.run(event("user_message", "a"))
        engine.run(event("user_message", "b"))

        # disable: plain chat must work and keep history usable
        without = offline_config(self.tmp, modules=[])
        engine2, storage2 = make_engine(without)
        engine2.run(event("user_message", "c"))
        self.assertEqual(len(storage2.history("default")), 6)
        # counter state untouched while disabled
        self.assertEqual(
            storage2.module_state("counter", "__agent__").state, {"count": 2}
        )

        # re-enable: the count continues from the stored state
        engine3, storage3 = make_engine(with_counter)
        engine3.run(event("user_message", "d"))
        self.assertEqual(
            storage3.module_state("counter", "__agent__").state, {"count": 3}
        )


class StateVersionTest(TempDirTestCase):
    """T13: state structure changes require an explicit state_version bump."""

    def test_version_mismatch_is_a_clean_error(self):
        from .helpers import VersionedModule

        v1 = offline_config(self.tmp, modules=[spec_for(VersionedModule, config={"version": 1})])
        engine, storage = make_engine(v1)
        engine.run(event("user_message", "a"))
        self.assertEqual(storage.module_state("versioned", "default").state, {"n": 1})

        v2 = offline_config(self.tmp, modules=[spec_for(VersionedModule, config={"version": 2})])
        engine2, storage2 = make_engine(v2)
        with self.assertRaises(ModuleStateVersionError):
            engine2.run(event("user_message", "b"))
        # stored state is left untouched
        self.assertEqual(storage2.module_state("versioned", "default").state, {"n": 1})


def event(kind, content=None, conversation="default"):
    from emc_runtime import Event

    return Event(kind=kind, conversation_id=conversation, content=content)


if __name__ == "__main__":
    unittest.main()
