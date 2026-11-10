"""Engine lifecycle tests (T01, T06-T09, T12, T16 + contract enforcement)."""

from __future__ import annotations

import unittest

from emc_runtime import (
    ConcurrencyConflict,
    EMCError,
    Engine,
    Event,
    EventBus,
    ModuleContractError,
    Storage,
)
from emc_runtime.cognition import make_cognition
from emc_runtime.modules import ModuleRegistry

from .helpers import (
    BreakingFinalizeModule,
    BreakingPrepareModule,
    EmptyReplyProvider,
    FlakyProvider,
    NaNStateModule,
    RecordingProvider,
    RevisionBumpingModule,
    SpoofedBlockModule,
    TempDirTestCase,
    offline_config,
    spec_for,
)


def user_message(content: str) -> Event:
    """Build the event up front so assertRaises blocks hold a single call."""
    return Event(kind="user_message", content=content)


def make_engine(config, provider=None):
    storage = Storage(config.database)
    registry = ModuleRegistry(config.modules)
    provider = provider or RecordingProvider()
    engine = Engine(
        config=config,
        storage=storage,
        registry=registry,
        provider=provider,
        cognition=make_cognition(config.cognition_kind, config.cognition_params),
        bus=EventBus(),
    )
    return engine, storage, provider


class PureChatTest(TempDirTestCase):
    """T01: modules=[] still gives a full ordered chat experience."""

    def test_two_turns_offline(self):
        config = offline_config(self.tmp)
        engine, storage, provider = make_engine(config)
        first = engine.run(Event(kind="user_message", content="你好"))
        self.assertEqual(first.event_kind, "user_message")
        self.assertIn("messages=2", first.reply)  # system + user
        second = engine.run(Event(kind="user_message", content="第二句"))
        self.assertIn("messages=4", second.reply)  # system + 2 history + user
        history = storage.history("default")
        self.assertEqual([m.role for m in history], ["user", "assistant", "user", "assistant"])
        self.assertEqual([m.content for m in history][::2], ["你好", "第二句"])
        self.assertEqual(storage.revision(), 2)

    def test_reply_is_recorded_verbatim(self):
        config = offline_config(self.tmp)
        engine, storage, provider = make_engine(config, provider=RecordingProvider("他说"))
        result = engine.run(Event(kind="user_message", content="hi"))
        self.assertEqual(storage.history("default")[-1].content, result.reply)


class ModuleFailureIsolationTest(TempDirTestCase):
    def _counts(self, storage):
        return (
            storage.revision(),
            len(storage.history("default")),
            len(storage.recent_turns()),
            storage.module_state("breaker", "default"),
        )

    def test_t06_prepare_exception_aborts_turn(self):
        config = offline_config(self.tmp, modules=[spec_for(BreakingPrepareModule)])
        engine, storage, provider = make_engine(config)
        before = self._counts(storage)
        payload = user_message("boom")
        with self.assertRaises(EMCError):
            engine.run(payload)
        self.assertEqual(self._counts(storage), before)
        self.assertEqual(len(provider.calls), 0)  # model never called

    def test_t07_provider_failure_aborts_turn(self):
        config = offline_config(self.tmp, modules=[])
        engine, storage, _ = make_engine(config, provider=FlakyProvider(failures=1))
        before = self._counts(storage)
        payload = user_message("hi")
        with self.assertRaises(EMCError):
            engine.run(payload)
        self.assertEqual(self._counts(storage), before)

        engine2, storage2, _ = make_engine(config, provider=EmptyReplyProvider())
        with self.assertRaises(EMCError):
            engine2.run(payload)
        self.assertEqual(self._counts(storage2), before)

    def test_t08_finalize_exception_aborts_after_model_call(self):
        config = offline_config(
            self.tmp, modules=[spec_for(BreakingFinalizeModule, module_id="breaker")]
        )
        engine, storage, provider = make_engine(config)
        before = self._counts(storage)
        payload = user_message("hi")
        with self.assertRaises(EMCError):
            engine.run(payload)
        self.assertEqual(len(provider.calls), 1)  # model was paid for...
        self.assertEqual(self._counts(storage), before)  # ...but nothing was written

    def test_t09_concurrency_conflict_never_replays_model(self):
        config = offline_config(
            self.tmp,
            modules=[
                spec_for(
                    RevisionBumpingModule,
                    module_id="bumper",
                    config={"db_path": str(self.tmp / "agent.sqlite3")},
                )
            ],
        )
        engine, storage, provider = make_engine(config)
        payload = user_message("hi")
        with self.assertRaises(ConcurrencyConflict):
            engine.run(payload)
        self.assertEqual(len(provider.calls), 1)  # exactly one model request
        self.assertEqual(len(storage.history("default")), 0)
        self.assertEqual(storage.recent_turns(), [])  # no turn record either

    def test_t16_nan_state_aborts_turn(self):
        config = offline_config(self.tmp, modules=[spec_for(NaNStateModule)])
        engine, storage, provider = make_engine(config)
        before = self._counts(storage)
        payload = user_message("hi")
        with self.assertRaises(EMCError):
            engine.run(payload)
        self.assertEqual(self._counts(storage), before)


class ContractEnforcementTest(TempDirTestCase):
    def test_spoofed_context_block_source_rejected(self):
        config = offline_config(self.tmp, modules=[spec_for(SpoofedBlockModule)])
        engine, storage, _ = make_engine(config)
        payload = user_message("hi")
        with self.assertRaises(ModuleContractError):
            engine.run(payload)
        self.assertEqual(len(storage.history("default")), 0)


class InternalTriggerTest(TempDirTestCase):
    def test_t12_no_fake_user_message(self):
        config = offline_config(self.tmp, modules=[])
        engine, storage, provider = make_engine(config)
        engine.run(Event(kind="user_message", content="先聊一句"))
        trigger = engine.run(
            Event(
                kind="internal_trigger",
                conversation_id="default",
                metadata={"reason": "clock"},
            )
        )
        self.assertEqual(trigger.event_kind, "internal_trigger")
        history = storage.history("default")
        roles = [m.role for m in history]
        # only one user message: the real one
        self.assertEqual(roles, ["user", "assistant", "assistant"])
        # the trigger note reached the model as the last message...
        last_context = provider.calls[-1][-1]
        self.assertIn("internal trigger", last_context["content"])
        # ...but was never stored as user history
        self.assertNotIn("internal trigger", [m.content for m in history])
        turn = storage.recent_turns("default", limit=1)[0]
        self.assertEqual(turn["event_kind"], "internal_trigger")
        self.assertEqual(turn["trace"]["context_messages"][-1]["role"], "user")
        self.assertIn("internal trigger", turn["trace"]["context_messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
