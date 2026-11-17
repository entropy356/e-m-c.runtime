"""Execute a research workflow, then atomically commit its staged effects."""
from __future__ import annotations

import copy
import threading
import uuid
from dataclasses import asdict, replace

from . import providers
from .clock import SystemClock, timestamp
from .cognition import make_cognition
from .contracts import AssistantReply, EMCError, Event, ProviderError, RunResult, validate_event
from .events import EventBus
from .modules import ModuleRegistry
from .session import TurnContext
from .storage import Storage


class Engine:
    def __init__(self, *, config, storage, registry, provider, cognition, bus=None,
                 clock=None, models=None, embeddings=None):
        self.config, self.storage, self.registry = config, storage, registry
        self.provider, self.cognition = provider, cognition
        self.bus = bus or EventBus()
        self.clock = clock or SystemClock()
        self._models = dict(models or {})
        if provider is not None:
            self._models['default'] = provider
        self._embeddings = dict(embeddings or {})
        self._lock = threading.RLock()

    def _resolve(self, kind, name):
        instances = self._models if kind == 'chat' else self._embeddings
        if name not in instances:
            group = 'models' if kind == 'chat' else 'embeddings'
            params = (self.config.provider_params if kind == 'chat' and name == 'default'
                      else self.config.raw.get(group, {}).get(name))
            if params is None:
                raise ProviderError(f'unknown {kind} model {name!r}')
            if kind == 'chat':
                instances[name] = providers.make_provider(params)
            else:
                from .providers.embeddings import make_embedding_provider
                instances[name] = make_embedding_provider(params)
        return instances[name]

    def run(self, event: Event, *, _schedule_id=None) -> RunResult:
        with self._lock:
            return self._run(event, _schedule_id)

    def _run(self, raw_event, schedule_id):
        now = timestamp(self.clock.now())
        event = validate_event(raw_event)
        event = replace(event, occurred_at=timestamp(event.occurred_at or now))
        if self.storage.event_committed(event.event_id):
            raise EMCError(f'event {event.event_id!r} already committed; no model request was made')
        if self.storage.event_schedule(event.event_id) not in (None, schedule_id):
            raise EMCError('event belongs to a pending schedule; execute it with run_due')
        attempt_id = uuid.uuid4().hex
        self.storage.start_attempt(attempt_id, asdict(event), now)
        ctx = None
        try:
            snapshot = self.storage.snapshot(event.conversation_id, self.registry.module_keys(event.conversation_id))
            ctx = TurnContext(config=self.config, event=event, snapshot=snapshot, registry=self.registry,
                              storage=self.storage, resolve=self._resolve, attempt_id=attempt_id,
                              clock=self.clock, now=now)
            reply = self.cognition.run(ctx)
            if reply is not None and (not isinstance(reply, AssistantReply) or not isinstance(reply.content, str) or not reply.content.strip()):
                raise ProviderError('cognition must return a nonempty AssistantReply or None')
            ctx.validate_completion()
            ctx.stage = 'commit'
            content = reply.content if reply is not None else None
            messages = [('assistant', content)] if content is not None else []
            if event.kind == 'user_message':
                messages.insert(0, ('user', event.content))
            trace = {**copy.deepcopy(ctx.trace), 'event': asdict(event), 'attempt_id': attempt_id,
                     'modules': ctx.module_traces, 'reply': content,
                     'cognition': self.config.cognition_kind,
                     'workflow_states': [asdict(write) for write in ctx.state._writes.values()],
                     'module_states': [asdict(write) for write in ctx._module_writes]}
            revision = self.storage.commit_turn(
                expected_revision=snapshot.revision, conversation_id=event.conversation_id,
                turn_id=attempt_id, event_id=event.event_id, event_kind=event.kind,
                new_messages=messages, module_writes=ctx._module_writes,
                workflow_writes=list(ctx.state._writes.values()), trace=trace,
                scheduled=ctx.scheduler._scheduled, cancellations=ctx.scheduler._cancellations,
                consumed_schedule=schedule_id, attempt_id=attempt_id,
                committed_at=timestamp(self.clock.now()),
            )
        except Exception as error:
            try:
                self.storage.finish_failed_attempt(attempt_id, stage=ctx.stage if ctx else 'snapshot',
                    error=f'{type(error).__name__}: {error}', finished_at=timestamp(self.clock.now()))
            except Exception:
                pass  # preserve the original failure; the durable attempt remains running
            self.bus.publish({'type': 'turn_failed', 'event_id': event.event_id,
                              'conversation_id': event.conversation_id, 'event_kind': event.kind,
                              'attempt_id': attempt_id, 'stage': ctx.stage if ctx else 'snapshot'})
            if isinstance(error, EMCError):
                raise
            raise EMCError(f'turn failed: {type(error).__name__}: {error}') from error
        result = RunResult(turn_id=attempt_id, event_id=event.event_id,
                           conversation_id=event.conversation_id, event_kind=event.kind,
                           revision=revision, reply=content, module_traces=ctx.module_traces)
        self.bus.publish({'type': 'turn_committed', **result.to_json_dict()})
        return result

    def run_due(self, *, limit=100):
        """Execute a bounded batch already due. New schedules wait for the next call.

        A failure stops the batch and leaves that event pending. The caller decides
        when to try again; model requests are not automatically replayed.
        """
        results = []
        with self._lock:
            jobs = self.storage.pending_events(due_at=timestamp(self.clock.now()), limit=limit)
            for job in jobs:
                # A preceding event in this batch may have cancelled another one.
                if self.storage.schedule_pending(job['schedule_id']):
                    results.append(self._run(Event(**job['event']), job['schedule_id']))
        return results


def build_engine(config, bus=None, *, clock=None):
    return Engine(config=config, storage=Storage(config.database), registry=ModuleRegistry(config.modules),
                  provider=None,
                  cognition=make_cognition(config.cognition_kind, config.cognition_params),
                  bus=bus, clock=clock)
