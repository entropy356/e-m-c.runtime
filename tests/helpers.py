"""Shared fixtures: config builders and test double modules."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from emc_runtime import (
    AssistantReply,
    ContextBlock,
    Event,
    Finalized,
    ModuleContext,
    Prepared,
)
from emc_runtime.config import parse_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def offline_config(tmp_dir, modules=None, **overrides) -> dict:
    """A validated RuntimeConfig backed by a temp database."""
    data = {
        "schema_version": 1,
        "database": str(Path(tmp_dir) / "agent.sqlite3"),
        "provider": {"kind": "offline"},
        "cognition": {"kind": "direct"},
        "modules": modules or [],
    }
    data.update(overrides)
    return parse_config(data)


class RecordingProvider:
    """Offline double that records every request for order assertions."""

    kind = "recording"

    def __init__(self, reply_prefix="[offline]"):
        self.reply_prefix = reply_prefix
        self.calls: list[list[dict]] = []

    def generate(self, messages, options):
        self.calls.append([dict(message) for message in messages])
        last = messages[-1]["content"] if messages else ""
        return AssistantReply(content=f"{self.reply_prefix} messages={len(messages)} last={last}")


class FlakyProvider:
    """Fails N times then succeeds; counts total transport attempts."""

    kind = "flaky"

    def __init__(self, failures=1):
        self.failures = failures
        self.attempts = 0

    def generate(self, messages, options):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RuntimeError("simulated provider outage")
        return AssistantReply(content="recovered reply")


class EmptyReplyProvider:
    kind = "empty"

    def generate(self, messages, options):
        return AssistantReply(content="   ")


class BreakingPrepareModule:
    """A module whose prepare explodes; used for failure-isolation tests."""

    module_id = "breaker"
    scope = "conversation"
    state_version = 1

    def __init__(self, config, module_id, scope):
        self.module_id = module_id
        self.scope = scope

    def initial_state(self):
        return {"ok": True}

    def expose(self, state):
        return {"ok": state["ok"]}

    def prepare(self, event, state, ctx):
        raise RuntimeError("prepare exploded")

    def finalize(self, event, prepared, reply, ctx):
        return Finalized(state=prepared.working_state, trace={})

    def inspect(self, state):
        return state


class BreakingFinalizeModule(BreakingPrepareModule):
    """prepare succeeds; finalize explodes after the model call."""

    module_id = "finalizer"

    def prepare(self, event, state, ctx):
        return Prepared(
            working_state={"ok": True},
            context_blocks=[
                ContextBlock(source=self.module_id, content="finalize will explode")
            ],
            trace={},
        )

    def finalize(self, event, prepared, reply, ctx):
        raise RuntimeError("finalize exploded")


class RevisionBumpingModule(BreakingPrepareModule):
    """Simulates a concurrent writer by bumping the revision during finalize."""

    module_id = "bumper"

    def __init__(self, config, module_id, scope):
        super().__init__(config, module_id, scope)
        self.db_path = config["db_path"]

    def prepare(self, event, state, ctx):
        return Prepared(working_state={"ok": True}, context_blocks=[], trace={})

    def finalize(self, event, prepared, reply, ctx):
        import sqlite3

        connection = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE runtime_meta SET revision=revision+1 WHERE singleton_id=1"
            )
            connection.execute("COMMIT")
        finally:
            connection.close()
        return Finalized(state=prepared.working_state, trace={"bumped": True})


class NaNStateModule(BreakingPrepareModule):
    """Returns a non-strict-JSON state (NaN) that must abort the turn."""

    module_id = "nanster"

    def prepare(self, event, state, ctx):
        return Prepared(
            working_state={"value": float("nan")},
            context_blocks=[],
            trace={},
        )

    def finalize(self, event, prepared, reply, ctx):
        return Finalized(state=prepared.working_state, trace={})


class OfflineProviderStub:
    """Minimal well-behaved provider for module-focused tests."""

    kind = "offline"

    def __init__(self):
        from emc_runtime import AssistantReply

        self._reply_cls = AssistantReply

    def generate(self, messages, options):
        return self._reply_cls(content="ok")


class VersionedModule(BreakingPrepareModule):
    """State schema changes are guarded by a configurable state_version."""

    module_id = "versioned"
    state_version = 1

    def __init__(self, config, module_id, scope):
        super().__init__(config, module_id, scope)
        self._version = int(config.get("version", 1))

    @property
    def state_version(self):
        return self._version

    def prepare(self, event, state, ctx):
        return Prepared(
            working_state={"n": state.get("n", 0) + 1},
            context_blocks=[],
            trace={},
        )


class FixedAgentScopeModule(BreakingPrepareModule):
    """Declares agent scope and refuses to be configured otherwise."""

    module_id = "fixed"

    def __init__(self, config, module_id, scope):
        self.module_id = module_id
        self.scope = "agent"  # hardcoded: must match the configuration


class SpoofedBlockModule(BreakingPrepareModule):
    """Claims another module's identity in a ContextBlock."""

    module_id = "spoofer"

    def prepare(self, event, state, ctx):
        return Prepared(
            working_state={},
            context_blocks=[ContextBlock(source="someone-else", content="hi")],
            trace={},
        )


class ChattyModule:
    """Minimal well-behaved module used in several tests."""

    module_id = "chatty"
    scope = "conversation"
    state_version = 1

    def __init__(self, config, module_id, scope):
        self.module_id = module_id
        self.scope = scope
        self.seen_exports: list = []

    def initial_state(self):
        return {"turns": 0}

    def expose(self, state):
        return {"turns": state["turns"]}

    def prepare(self, event, state, ctx):
        self.seen_exports.append(dict(ctx.exports))
        return Prepared(
            working_state={"turns": state["turns"] + 1},
            context_blocks=[
                ContextBlock(
                    source=self.module_id, content=f"chatty has seen {state['turns']} turns"
                )
            ],
            trace={},
        )

    def finalize(self, event, prepared, reply, ctx):
        return Finalized(state=prepared.working_state, trace={})

    def inspect(self, state):
        return state


def spec_for(cls, module_id=None, scope="conversation", config=None):
    return {
        "id": module_id or cls.module_id,
        "entrypoint": f"tests.helpers:{cls.__name__}",
        "scope": scope,
        "config": config or {},
    }


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)


def write_json(path: Path, value) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path
