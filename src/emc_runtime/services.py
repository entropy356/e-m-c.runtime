"""Per-turn research services; all mutations are staged until the turn commits."""
from __future__ import annotations

import copy
import time
import uuid
from dataclasses import asdict, replace

from .clock import timestamp
from .contracts import (AssistantReply, EMCError, Event, ModuleStateVersionError,
                        ProviderError, check_json_value, validate_event)
from .modules import scope_key_for
from .storage import ModuleStateWrite


class StateStore:
    """JSON state addressed by a researcher-defined key and persistence scope."""

    def __init__(self, snapshot, conversation_id):
        self._snapshot = snapshot
        self._conversation_id = conversation_id
        self._writes = {}

    def _key(self, key, scope, version):
        if not isinstance(key, str) or not key.strip():
            raise EMCError('state key must be nonempty text')
        if type(version) is not int or version < 1:
            raise EMCError('state version must be a positive integer')
        return key, scope_key_for(scope, self._conversation_id)

    def get(self, key, *, scope='agent', version=1, default=None):
        address = self._key(key, scope, version)
        entry = self._writes.get(address) or self._snapshot.get(address)
        if entry is None:
            return copy.deepcopy(default)
        if entry.state_version != version:
            raise ModuleStateVersionError(f'state {key!r} has version {entry.state_version}, requested {version}')
        return copy.deepcopy(entry.state)

    def set(self, key, value, *, scope='agent', version=1):
        address = self._key(key, scope, version)
        self.get(key, scope=scope, version=version)
        check_json_value(value, f'state {key!r}')
        self._writes[address] = ModuleStateWrite(*address, version, copy.deepcopy(value))


class ExperienceReader:
    def __init__(self, storage, revision):
        self._storage, self._revision = storage, revision

    def query(self, *, conversation_id=None, text=None, before_id=None, after_id=None, limit=100):
        return self._storage.experiences(revision=self._revision, conversation_id=conversation_id,
                                        text=text, before_id=before_id, after_id=after_id, limit=limit)


class Scheduler:
    """Stages explicit one-shot events; no wake-up policy is built in."""

    def __init__(self):
        self._scheduled = []
        self._cancellations = []

    def schedule(self, event: Event, *, due_at, schedule_id=None):
        due_at = timestamp(due_at)
        schedule_id = schedule_id or uuid.uuid4().hex
        if not isinstance(schedule_id, str) or not schedule_id.strip():
            raise EMCError('schedule_id must be nonempty text')
        event = validate_event(event)
        event = replace(event, occurred_at=timestamp(event.occurred_at or due_at))
        self._scheduled.append({'schedule_id': schedule_id, 'due_at': due_at, 'event': asdict(event)})
        return schedule_id

    def cancel(self, schedule_id: str):
        if not isinstance(schedule_id, str) or not schedule_id.strip():
            raise EMCError('schedule_id must be nonempty text')
        self._scheduled = [job for job in self._scheduled if job['schedule_id'] != schedule_id]
        self._cancellations.append(schedule_id)


class ModelCalls:
    """Named chat/embedding calls with durable request and outcome observations."""

    def __init__(self, *, resolve, storage, attempt_id, clock):
        self._resolve, self._storage = resolve, storage
        self._attempt_id, self._clock = attempt_id, clock
        self._seq = 0

    def _call(self, kind, name, request, purpose, invoke):
        self._seq += 1
        seq = self._seq
        record = {'kind': kind, 'name': name, 'purpose': purpose,
                  'request': copy.deepcopy(request), 'started_at': timestamp(self._clock.now()),
                  'status': 'running'}
        check_json_value(record, 'model request')
        self._storage.record_call(self._attempt_id, seq, record)
        start = time.monotonic()
        try:
            provider = self._resolve(kind, name)
            record['provider'] = getattr(provider, 'kind', type(provider).__name__)
            record['model'] = getattr(provider, 'model', None)
            if kind == 'chat':
                request['options'] = {**copy.deepcopy(getattr(provider, 'options', {})), **request['options']}
                record['request'] = copy.deepcopy(request)
            self._storage.record_call(self._attempt_id, seq, record)
            result, observed = invoke(provider, request)
            check_json_value(observed, 'model output')
            record.update(status='succeeded', output=observed)
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - start
            record['finished_at'] = timestamp(self._clock.now())
            self._storage.record_call(self._attempt_id, seq, record)
        return result

    def chat(self, messages, *, model='default', options=None, purpose=None) -> AssistantReply:
        request = {'messages': copy.deepcopy(list(messages)), 'options': copy.deepcopy(dict(options or {}))}

        def invoke(provider, request):
            reply = provider.generate(request['messages'], request['options'])
            if not isinstance(reply, AssistantReply) or not isinstance(reply.content, str) or not reply.content.strip():
                raise ProviderError('provider must return a nonempty AssistantReply')
            return reply, asdict(reply)

        return self._call('chat', model, request, purpose, invoke)

    def embed(self, text: str, *, model: str, purpose=None) -> list[float]:
        if not isinstance(text, str):
            raise ProviderError('embedding input must be text')

        def invoke(provider, request):
            vector = list(provider.embed(request['text']))
            from .providers.embeddings import unit
            unit(vector)  # validate without changing the provider's chosen representation
            return vector, {'vector': vector, 'dimensions': len(vector)}

        return self._call('embedding', model, {'text': text}, purpose, invoke)
