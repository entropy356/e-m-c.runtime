"""Single-turn orchestration.

The engine wires modules, context assembly, cognition, storage and observers
together. It imports no concrete module implementation (no examples, no
memory/emotion code) — modules arrive exclusively through configuration.

Turn lifecycle (unchangeable order):

[1] validate the event, read one consistent SQLite snapshot
[2] module expose (read-only signals) + prepare on deep-copied state
[3] assemble the LLM context deterministically
[4] run the cognition strategy through the chat provider
[5] module finalize on the same working copies
[6] one SQLite transaction: verify revision, write messages/states/turn,
    bump revision
[7] after a durable commit, publish best-effort read-only observation events
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from . import providers
from .config import RuntimeConfig
from .contracts import (
    AssistantReply,
    ChatProvider,
    CognitionStrategy,
    ContextBlock,
    ConcurrencyConflict,
    EMCError,
    Event,
    Finalized,
    Module,
    ModuleContractError,
    ModuleStateVersionError,
    Prepared,
    RunResult,
    check_json_value,
    validate_event,
)
from .context import ContextAssembler
from .cognition import make_cognition
from .events import EventBus
from .modules import ModuleRegistry, scope_key_for
from .storage import ModuleStateWrite, Snapshot, Storage


class ModuleStageError(EMCError):
    """A module raised during one lifecycle stage; the whole turn failed."""


def _require_prepared(result: Any, module: Module) -> Prepared:
    if not isinstance(result, Prepared):
        raise ModuleContractError(
            f"module {module.module_id!r} prepare must return a Prepared object"
        )
    if not isinstance(result.context_blocks, list):
        raise ModuleContractError(
            f"module {module.module_id!r} context_blocks must be a list"
        )
    for block in result.context_blocks:
        if not isinstance(block, ContextBlock):
            raise ModuleContractError(
                f"module {module.module_id!r} context_blocks must contain ContextBlock objects"
            )
        if block.source != module.module_id:
            raise ModuleContractError(
                f"module {module.module_id!r} produced a ContextBlock with spoofed "
                f"source {block.source!r}"
            )
        if not isinstance(block.content, str) or not block.content.strip():
            raise ModuleContractError(
                f"module {module.module_id!r} produced an empty ContextBlock"
            )
    if not isinstance(result.trace, dict):
        raise ModuleContractError(
            f"module {module.module_id!r} prepare trace must be a dict"
        )
    return result


def _require_finalized(result: Any, module: Module) -> Finalized:
    if not isinstance(result, Finalized):
        raise ModuleContractError(
            f"module {module.module_id!r} finalize must return a Finalized object"
        )
    if not isinstance(result.trace, dict):
        raise ModuleContractError(
            f"module {module.module_id!r} finalize trace must be a dict"
        )
    return result


class Engine:
    """Executes one event at a time; holds no per-turn mutable global state."""

    def __init__(
        self,
        *,
        config: RuntimeConfig,
        storage: Storage,
        registry: ModuleRegistry,
        provider: ChatProvider,
        cognition: CognitionStrategy,
        bus: EventBus | None = None,
    ):
        self.config = config
        self.storage = storage
        self.registry = registry
        self.provider = provider
        self.cognition = cognition
        self.bus = bus or EventBus()
        self.assembler = ContextAssembler()

    # ------------------------------------------------------------------ public

    def run(self, event: Event) -> RunResult:
        """Run one validated event through the full lifecycle."""
        try:
            return self._run(event)
        except EMCError:
            self._publish_failed(event)
            raise
        except Exception as error:
            self._publish_failed(event)
            raise EMCError(f"turn failed: {type(error).__name__}: {error}") from error

    # ----------------------------------------------------------------- private

    def _run(self, raw_event: Event) -> RunResult:
        event = validate_event(raw_event)

        # [1] snapshot ------------------------------------------------------
        snapshot = self.storage.snapshot(
            event.conversation_id, self.registry.module_keys(event.conversation_id)
        )
        working = self._working_states(event, snapshot)

        # [2] expose + prepare ----------------------------------------------
        exports = self._exports(working)
        prepared: dict[str, Prepared] = {}
        for module in self.registry.modules:
            ctx = ModuleRegistry.make_context(event, snapshot.history, exports)
            try:
                result = module.prepare(event, working[module.module_id], ctx)
            except EMCError:
                raise
            except Exception as error:
                raise ModuleStageError(
                    f"module {module.module_id!r} prepare failed: "
                    f"{type(error).__name__}: {error}"
                ) from error
            result = _require_prepared(result, module)
            check_json_value(
                result.working_state, f"module {module.module_id!r} working_state"
            )
            check_json_value(result.trace, f"module {module.module_id!r} prepare trace")
            prepared[module.module_id] = result

        # [3] context assembly ----------------------------------------------
        blocks = [
            block
            for module in self.registry.modules
            for block in prepared[module.module_id].context_blocks
        ]
        context_messages = self.assembler.assemble(
            system_prompt=self.config.system_prompt,
            blocks=blocks,
            history=snapshot.history,
            event=event,
            internal_trigger_prompt=self.config.internal_trigger_prompt,
        )

        # [4] cognition ------------------------------------------------------
        reply = self.cognition.run(
            context_messages, self.provider, self.config.cognition_params
        )

        # [5] finalize -------------------------------------------------------
        finalized: dict[str, Finalized] = {}
        for module in self.registry.modules:
            ctx = ModuleRegistry.make_context(event, snapshot.history, exports)
            try:
                result = module.finalize(event, prepared[module.module_id], reply, ctx)
            except EMCError:
                raise
            except Exception as error:
                raise ModuleStageError(
                    f"module {module.module_id!r} finalize failed: "
                    f"{type(error).__name__}: {error}"
                ) from error
            result = _require_finalized(result, module)
            check_json_value(result.state, f"module {module.module_id!r} final state")
            check_json_value(result.trace, f"module {module.module_id!r} finalize trace")
            finalized[module.module_id] = result

        # [6] atomic commit ---------------------------------------------------
        turn_id = uuid.uuid4().hex
        new_messages = self._new_messages(event, reply)
        trace = self._build_trace(event, context_messages, prepared, finalized, reply)
        new_revision = self.storage.commit_turn(
            expected_revision=snapshot.revision,
            conversation_id=event.conversation_id,
            turn_id=turn_id,
            event_id=event.event_id,
            event_kind=event.kind,
            new_messages=new_messages,
            module_writes=[
                ModuleStateWrite(
                    module_id=module.module_id,
                    scope_key=scope_key_for(module.scope, event.conversation_id),
                    state_version=module.state_version,
                    state=finalized[module.module_id].state,
                )
                for module in self.registry.modules
            ],
            trace=trace,
        )

        # [7] observation ----------------------------------------------------
        result = RunResult(
            turn_id=turn_id,
            event_id=event.event_id,
            conversation_id=event.conversation_id,
            event_kind=event.kind,
            revision=new_revision,
            reply=reply.content,
            module_traces={
                module.module_id: {
                    "prepare": prepared[module.module_id].trace,
                    "finalize": finalized[module.module_id].trace,
                }
                for module in self.registry.modules
            },
        )
        self._publish_committed(result)
        return result

    def _working_states(self, event: Event, snapshot: Snapshot) -> dict[str, Any]:
        """Deep copies of stored (or initial) state; never shared with storage."""
        working: dict[str, Any] = {}
        for module in self.registry.modules:
            entry = snapshot.module_states.get(module.module_id)
            if entry is None:
                state = copy.deepcopy(module.initial_state())
            else:
                if entry.state_version != module.state_version:
                    raise ModuleStateVersionError(
                        f"module {module.module_id!r} stored state_version is "
                        f"{entry.state_version} but code declares {module.state_version}; "
                        "state was left untouched (write an explicit migrator if needed)"
                    )
                state = copy.deepcopy(entry.state)
            check_json_value(state, f"module {module.module_id!r} loaded state")
            working[module.module_id] = state
        return working

    def _exports(self, working: dict[str, Any]) -> dict[str, Any]:
        """Read-only public signals of every module, from the same snapshot."""
        exports: dict[str, Any] = {}
        for module in self.registry.modules:
            try:
                signal = module.expose(copy.deepcopy(working[module.module_id]))
            except EMCError:
                raise
            except Exception as error:
                raise ModuleStageError(
                    f"module {module.module_id!r} expose failed: "
                    f"{type(error).__name__}: {error}"
                ) from error
            check_json_value(signal, f"module {module.module_id!r} exposed signal")
            exports[module.module_id] = copy.deepcopy(signal)
        return exports

    @staticmethod
    def _new_messages(event: Event, reply: AssistantReply) -> list[tuple[str, str]]:
        if event.kind == "user_message":
            return [("user", event.content), ("assistant", reply.content)]
        # internal_trigger never writes a user message
        return [("assistant", reply.content)]

    def _build_trace(
        self,
        event: Event,
        context_messages: list[dict],
        prepared: dict[str, Prepared],
        finalized: dict[str, Finalized],
        reply: AssistantReply,
    ) -> dict:
        return {
            "event": {
                "kind": event.kind,
                "event_id": event.event_id,
                "content": event.content,
                "metadata": dict(event.metadata),
            },
            "context_messages": [
                {"role": message["role"], "content": message["content"]}
                for message in context_messages
            ],
            "context_summary": {
                "count": len(context_messages),
                "roles": [message["role"] for message in context_messages],
                "module_blocks": [
                    {"source": block.source, "content": block.content}
                    for module in self.registry.modules
                    for block in prepared[module.module_id].context_blocks
                ],
            },
            "modules": {
                module.module_id: {
                    "prepare_trace": prepared[module.module_id].trace,
                    "finalize_trace": finalized[module.module_id].trace,
                }
                for module in self.registry.modules
            },
            "reply": reply.content,
            "provider": getattr(self.provider, "kind", type(self.provider).__name__),
            "cognition": self.config.cognition_kind,
        }

    # -------------------------------------------------------------- observers

    def _publish_committed(self, result: RunResult) -> None:
        self.bus.publish(
            {
                "type": "turn_committed",
                "turn_id": result.turn_id,
                "event_kind": result.event_kind,
                "conversation_id": result.conversation_id,
                "revision": result.revision,
                "reply": result.reply,
            }
        )

    def _publish_failed(self, event: Event) -> None:
        try:
            self.bus.publish(
                {
                    "type": "turn_failed",
                    "event_kind": event.kind,
                    "conversation_id": event.conversation_id,
                    "event_id": event.event_id,
                }
            )
        except Exception:  # noqa: BLE001 - observation must never mask the real error
            pass


def build_engine(config: RuntimeConfig, bus: EventBus | None = None) -> Engine:
    """Wire a fully configured engine from a validated RuntimeConfig."""
    provider = providers.make_provider(config.provider_params)
    cognition = make_cognition(config.cognition_kind, config.cognition_params)
    registry = ModuleRegistry(config.modules)
    storage = Storage(config.database)
    return Engine(
        config=config,
        storage=storage,
        registry=registry,
        provider=provider,
        cognition=cognition,
        bus=bus,
    )
