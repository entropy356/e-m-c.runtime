"""Context given to a research workflow for one transaction."""
from __future__ import annotations

import copy
from typing import Any

from .contracts import (AssistantReply, ContextBlock, EMCError, Finalized, Module,
                        ModuleContractError, ModuleStateVersionError, Prepared, check_json_value)
from .context import ContextAssembler
from .modules import ModuleRegistry, scope_key_for
from .services import ExperienceReader, ModelCalls, Scheduler, StateStore
from .storage import ModuleStateWrite

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


class TurnContext:
    """Research-owned execution order over runtime-owned persistence and calls.

    Mutate state through state.set(); model calls and observations persist even
    if the workflow subsequently fails. Module lifecycle helpers are optional.
    """

    def __init__(self, *, config, event, snapshot, registry, storage, resolve, attempt_id, clock, now):
        self.event = copy.deepcopy(event)
        self.now = now
        self.history = snapshot.history
        self.revision = snapshot.revision
        self.state = StateStore(snapshot.workflow_states, event.conversation_id)
        self.experiences = ExperienceReader(storage, snapshot.revision)
        self.models = ModelCalls(resolve=resolve, storage=storage, attempt_id=attempt_id, clock=clock)
        self.scheduler = Scheduler()
        self.trace = {}
        self.stage = 'cognition'
        self._config, self._registry, self._snapshot = config, registry, snapshot
        self._prepared = None
        self._exports = {}
        self._module_writes = []
        self._finalized = False
        self.module_traces = {}

    def observe(self, name: str, value):
        check_json_value(value, 'observation')
        self.trace[name] = copy.deepcopy(value)

    def assemble(self, *, blocks=(), history=None, history_limit=None, system_prompt=None):
        selected = self.history if history is None else tuple(history)
        if history_limit is not None:
            if type(history_limit) is not int or history_limit < 0:
                raise EMCError('history_limit must be a nonnegative integer')
            selected = selected[-history_limit:] if history_limit else ()
        messages = ContextAssembler().assemble(
            system_prompt=system_prompt or self._config.system_prompt,
            blocks=blocks, history=selected, event=self.event,
            internal_trigger_prompt=self._config.internal_trigger_prompt,
        )
        self.trace['context_messages'] = copy.deepcopy(messages)
        self.trace['history_sources'] = [message.message_id for message in selected]
        return messages

    def prepare_modules(self):
        if self._prepared is not None:
            raise ModuleContractError('prepare_modules may be called once per turn')
        self._prepared = {}
        working = {}
        for module in self._registry.modules:
            self.stage = f'module:{module.module_id}:load'
            entry = self._snapshot.module_states.get(module.module_id)
            if entry is not None and entry.state_version != module.state_version:
                raise ModuleStateVersionError(f'module {module.module_id!r} state version mismatch')
            state = copy.deepcopy(entry.state if entry is not None else module.initial_state())
            check_json_value(state, 'module state')
            working[module.module_id] = state
            self.stage = f'module:{module.module_id}:expose'
            signal = module.expose(copy.deepcopy(state))
            check_json_value(signal, 'module exports')
            self._exports[module.module_id] = copy.deepcopy(signal)
        blocks = []
        for module in self._registry.modules:
            self.stage = f'module:{module.module_id}:prepare'
            ctx = ModuleRegistry.make_context(self.event, self.history, self._exports)
            result = _require_prepared(module.prepare(ctx.event, working[module.module_id], ctx), module)
            check_json_value(result.working_state, 'module working state')
            check_json_value(result.trace, 'module prepare trace')
            self._prepared[module.module_id] = result
            self.module_traces[module.module_id] = {'prepare': copy.deepcopy(result.trace)}
            blocks.extend(result.context_blocks)
        self.stage = 'cognition'
        return blocks

    def finalize_modules(self, reply: AssistantReply):
        if self._prepared is None or self._finalized:
            raise ModuleContractError('finalize_modules requires one preceding prepare_modules')
        for module in self._registry.modules:
            self.stage = f'module:{module.module_id}:finalize'
            ctx = ModuleRegistry.make_context(self.event, self.history, self._exports)
            result = _require_finalized(module.finalize(ctx.event, self._prepared[module.module_id], reply, ctx), module)
            check_json_value(result.state, 'module final state')
            check_json_value(result.trace, 'module finalize trace')
            self._module_writes.append(ModuleStateWrite(module.module_id,
                scope_key_for(module.scope, self.event.conversation_id), module.state_version, copy.deepcopy(result.state)))
            self.module_traces[module.module_id]['finalize'] = copy.deepcopy(result.trace)
        self._finalized = True
        self.stage = 'cognition'

    def validate_completion(self):
        if self._prepared is not None and not self._finalized:
            raise ModuleContractError('prepared modules must be finalized before commit')
        check_json_value(self.trace, 'workflow trace')
