"""Reference module: an agent-scope turn counter.

Demonstrates the full contract with no domain logic: state is a plain dict,
prepare increments a working copy and contributes a ContextBlock, finalize
returns the same working copy for atomic commit.
"""

from __future__ import annotations

from typing import Any

from emc_runtime import AssistantReply, ContextBlock, Event, Finalized, ModuleContext, Prepared


class CounterModule:
    module_id = "counter"
    scope = "agent"
    state_version = 1

    def __init__(self, config: dict, module_id: str, scope: str):
        # The runtime passes (config, module_id, scope); this module needs no
        # configuration of its own but must accept the standard constructor.
        self.config = config
        self.module_id = module_id
        self.scope = scope

    def initial_state(self) -> Any:
        return {"count": 0}

    def expose(self, state: Any) -> Any:
        return {"count": state["count"]}

    def prepare(self, event: Event, state: Any, _ctx: ModuleContext) -> Prepared:
        working = {"count": state["count"] + 1}
        block = ContextBlock(
            source=self.module_id,
            content=f"Counter value: {working['count']}",
        )
        return Prepared(
            working_state=working,
            context_blocks=[block],
            trace={"event_kind": event.kind, "count_after_increment": working["count"]},
        )

    def finalize(
        self, event: Event, prepared: Prepared, _reply: AssistantReply, _ctx: ModuleContext
    ) -> Finalized:
        return Finalized(
            state=prepared.working_state,
            trace={"committed_count": prepared.working_state["count"]},
        )

    def inspect(self, state: Any) -> Any:
        return {"count": state["count"]}
