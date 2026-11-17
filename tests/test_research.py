"""Infrastructure behavior independent of any emotion, memory or wake policy."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest import mock

from emc_runtime import AssistantReply, EMCError, Event, ModuleStateVersionError, build_engine
from emc_runtime.modules import ModuleRegistry
from emc_runtime.providers.openai_chat import OpenAIChatProvider

from .helpers import TempDirTestCase, offline_config


class FixedClock:
    def __init__(self):
        self.value = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def now(self):
        return self.value


class FunctionWorkflow:
    def __init__(self, fn):
        self.fn = fn

    def run(self, ctx):
        return self.fn(ctx)


class EntryWorkflow:
    def __init__(self, config):
        self.text = config['text']

    def run(self, ctx):
        ctx.state.set('entry', self.text)
        return AssistantReply(self.text)


class FakeModel:
    kind = 'fake'
    model = 'fake-model'
    options = {'temperature': 0.1, 'seed': 4}

    def __init__(self):
        self.requests = []

    def generate(self, messages, options):
        self.requests.append((messages, dict(options)))
        return AssistantReply('result', usage={'total_tokens': 7}, model='served-model')


class FakeEmbedding:
    model = 'embedding-model'

    def embed(self, text):
        return [0.6, 0.8]


class ResearchTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.clock = FixedClock()
        self.engine = build_engine(offline_config(self.tmp), clock=self.clock)

    def workflow(self, fn):
        self.engine.cognition = FunctionWorkflow(fn)

    def run_turn(self, text='input', conversation='default', event_id=''):
        return self.engine.run(Event('user_message', conversation, text, event_id=event_id))

    def test_multiple_calls_dynamic_options_and_named_embedding(self):
        fake = FakeModel()
        self.engine._models['analysis'] = fake
        self.engine._embeddings['index'] = FakeEmbedding()

        def fn(ctx):
            count = ctx.state.get('count', default=0) + 1
            ctx.state.set('count', count)
            ctx.models.embed('source text', model='index', purpose='indexing')
            ctx.models.chat([{'role': 'user', 'content': 'extract'}], model='analysis',
                            options={'response_format': {'type': 'json_object'}}, purpose='extraction')
            return ctx.models.chat(ctx.assemble(history_limit=0), model='analysis',
                                   options={'temperature': count / 10}, purpose='reply')

        self.workflow(fn)
        self.run_turn()
        self.run_turn()
        calls = self.engine.storage.attempts()[0]['calls']
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]['output']['dimensions'], 2)
        self.assertEqual(calls[-1]['request']['options'], {'temperature': 0.2, 'seed': 4})
        self.assertEqual(calls[-1]['output']['usage'], {'total_tokens': 7})
        self.assertEqual(calls[-1]['output']['model'], 'served-model')
        self.assertEqual(len(fake.requests[-1][0]), 2)

    def test_failure_keeps_calls_but_rolls_back_state_and_schedule(self):
        def fn(ctx):
            ctx.state.set('candidate', {'x': 1})
            ctx.scheduler.schedule(Event('internal_trigger'), due_at=self.clock.now())
            ctx.models.chat(ctx.assemble())
            raise RuntimeError('after paid call')

        self.workflow(fn)
        with self.assertRaises(EMCError):
            self.run_turn()
        storage = self.engine.storage
        self.assertEqual(storage.revision(), 0)
        self.assertEqual(storage.history('default'), [])
        self.assertEqual(storage.pending_events(), [])
        self.assertEqual(storage.snapshot('default', []).workflow_states, {})
        attempt = storage.attempts()[0]
        self.assertEqual(attempt['status'], 'failed')
        self.assertIn('after paid call', attempt['error'])
        self.assertEqual(attempt['calls'][0]['status'], 'succeeded')
        self.assertTrue(attempt['calls'][0]['output']['content'])

    def test_provider_failure_is_recorded_before_state_commit(self):
        def fn(ctx):
            return ctx.models.chat(ctx.assemble(), model='missing')

        self.workflow(fn)
        with self.assertRaises(EMCError):
            self.run_turn()
        call = self.engine.storage.attempts()[0]['calls'][0]
        self.assertEqual(call['status'], 'failed')
        self.assertIn('missing', call['error'])
        self.assertEqual(self.engine.storage.revision(), 0)

    def test_state_scopes_and_version_guard(self):
        def fn(ctx):
            for scope in ('agent', 'conversation'):
                count = ctx.state.get('count', scope=scope, default=0)
                ctx.state.set('count', count + 1, scope=scope)
            return AssistantReply('ok')

        self.workflow(fn)
        self.run_turn(conversation='a')
        self.run_turn(conversation='b')
        snapshot = self.engine.storage.snapshot('a', []).workflow_states
        self.assertEqual(snapshot['count', '__agent__'].state, 2)
        self.assertEqual(snapshot['count', 'a'].state, 1)
        self.workflow(lambda ctx: ctx.state.get('count', version=2))
        with self.assertRaises(ModuleStateVersionError):
            self.run_turn()

    def test_cross_conversation_sources_and_bounded_view(self):
        self.run_turn('  original detail  ', 'a')
        self.run_turn('other', 'b')
        seen = []

        def fn(ctx):
            rows = ctx.experiences.query(text='original detail')
            seen.extend(rows)
            self.assertEqual(len(ctx.assemble(history_limit=0)), 2)
            return AssistantReply('ok')

        self.workflow(fn)
        self.run_turn('next', 'b')
        self.assertEqual(seen[0].content, '  original detail  ')
        self.assertEqual(seen[0].conversation_id, 'a')
        self.assertIsNotNone(seen[0].message_id)
        self.assertEqual(seen[0].created_at, '2026-01-01T00:00:00.000000+00:00')
        self.assertEqual(len(self.engine.storage.experiences(revision=1)), 2)
        first = self.engine.storage.experiences(limit=2)
        self.assertEqual(len(self.engine.storage.experiences(before_id=first[0].message_id)), 4)

    def test_experience_reader_excludes_a_later_commit(self):
        held = []
        self.workflow(lambda ctx: (held.append(ctx.experiences), AssistantReply('ok'))[1])
        self.run_turn()
        self.assertEqual(held[0].query(), [])

    def test_scheduled_event_survives_restart_and_commits_once(self):
        def schedule(ctx):
            ctx.scheduler.schedule(Event('internal_trigger', metadata={'reason': 'test'}),
                                   due_at=self.clock.now() + timedelta(hours=1), schedule_id='wake')
            return AssistantReply('scheduled')

        self.workflow(schedule)
        self.run_turn()
        restarted = build_engine(self.engine.config, clock=self.clock)
        self.assertEqual(restarted.run_due(), [])
        self.clock.value += timedelta(hours=1)
        self.assertEqual(len(restarted.run_due()), 1)
        self.assertEqual(restarted.run_due(), [])
        self.assertEqual(restarted.storage.pending_events(), [])
        self.assertEqual([m.role for m in restarted.storage.history('default')], ['user', 'assistant', 'assistant'])
        request = restarted.storage.attempts()[0]['calls'][0]['request']
        self.assertIn('test', request['messages'][-1]['content'])

    def test_failed_schedule_stays_pending_for_explicit_retry(self):
        self.workflow(lambda ctx: (ctx.scheduler.schedule(Event('internal_trigger'), due_at=self.clock.now()), AssistantReply('ok'))[1])
        self.run_turn()
        self.workflow(lambda ctx: (_ for _ in ()).throw(RuntimeError('retry explicitly')))
        with self.assertRaises(EMCError):
            self.engine.run_due()
        self.assertEqual(len(self.engine.storage.pending_events()), 1)
        self.assertEqual(self.engine.storage.revision(), 1)

    def test_cancellation_and_new_due_events_wait_for_next_batch(self):
        def seed(ctx):
            for name in ('a', 'b'):
                ctx.scheduler.schedule(Event('internal_trigger'), due_at=self.clock.now(), schedule_id=name)
            return AssistantReply('ok')

        self.workflow(seed)
        self.run_turn()

        def run(ctx):
            ctx.scheduler.cancel('b')
            ctx.scheduler.schedule(Event('internal_trigger'), due_at=self.clock.now(), schedule_id='c')
            return AssistantReply('ok')

        self.workflow(run)
        self.assertEqual(len(self.engine.run_due()), 1)
        self.assertEqual([job['schedule_id'] for job in self.engine.storage.pending_events()], ['c'])

    def test_duplicate_event_does_not_repeat_model_call(self):
        self.run_turn(event_id='same')
        with self.assertRaises(EMCError):
            self.run_turn(event_id='same')
        self.assertEqual(len(self.engine.storage.attempts()), 1)

    def test_fork_is_independent_and_never_overwrites(self):
        self.run_turn()
        fork = self.engine.storage.fork(str(self.tmp / 'branch.sqlite3'))
        self.run_turn('next')
        self.assertEqual(fork.revision(), 1)
        self.assertEqual(len(fork.history('default')), 2)
        with self.assertRaises(FileExistsError):
            self.engine.storage.fork(fork.path)
        other = build_engine(replace(self.engine.config, database=fork.path), clock=self.clock)
        other.run(Event('user_message', content='independent'))
        self.assertNotIn('independent', [m.content for m in self.engine.storage.history('default')])

    def test_python_workflow_loaded_from_config(self):
        config = offline_config(self.tmp, cognition={'kind': 'python', 'entrypoint': 'tests.test_research:EntryWorkflow', 'config': {'text': 'entry'}})
        result = build_engine(config, clock=self.clock).run(Event('user_message', content='input'))
        self.assertEqual(result.reply, 'entry')

    def test_unused_named_model_does_not_require_credentials(self):
        config = offline_config(self.tmp, models={'unused': {'kind': 'openai-compatible'}}, embeddings={'unused': {'kind': 'gemini'}})
        build_engine(config, clock=self.clock).run(Event('user_message', content='input'))

    def test_nested_module_exports_and_metadata_are_isolated(self):
        event = Event('internal_trigger', metadata={'nested': {'x': 1}})
        exports = {'other': {'values': [1]}}
        a = ModuleRegistry.make_context(event, (), exports)
        b = ModuleRegistry.make_context(event, (), exports)
        a.exports['other']['values'].append(2)
        a.event.metadata['nested']['x'] = 9
        self.assertEqual(b.exports['other']['values'], [1])
        self.assertEqual(event.metadata['nested']['x'], 1)

    def test_openai_passes_experiment_options_and_usage(self):
        captured = {}

        def transport(url, payload, *args):
            captured.update(payload)
            return {'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}],
                    'model': 'served', 'usage': {'total_tokens': 3}}

        provider = OpenAIChatProvider(base_url='https://example.com', model='configured', api_key='secret', transport=transport)
        result = provider.generate([{'role': 'user', 'content': 'input'}],
                                   {'seed': 42, 'response_format': {'type': 'json_object'}, 'max_completion_tokens': 80})
        self.assertEqual(captured['seed'], 42)
        self.assertEqual(captured['max_completion_tokens'], 80)
        self.assertEqual(result.model, 'served')
        self.assertEqual(result.usage, {'total_tokens': 3})

    def test_read_commands_do_not_require_model_credentials(self):
        self.run_turn()
        config = dict(self.engine.config.raw)
        config['provider'] = {'kind': 'openai-compatible', 'base_url': 'https://example.com',
                              'model': 'model', 'api_key_env': 'EMC_ABSENT_KEY'}
        path = self.tmp / 'config.json'
        path.write_text(json.dumps(config))
        env = {key: value for key, value in os.environ.items() if key != 'EMC_ABSENT_KEY'}
        env['PYTHONPATH'] = str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src')
        for command in ('history', 'inspect', 'attempts', 'schedules'):
            result = subprocess.run([sys.executable, '-m', 'emc_runtime', '--config', str(path), command],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_internal_event_can_update_state_without_speaking(self):
        self.workflow(lambda ctx: ctx.state.set('background', {'updated': True}))
        result = self.engine.run(Event('internal_trigger', metadata={'reason': 'background'}))
        self.assertIsNone(result.reply)
        self.assertEqual(self.engine.storage.history('default'), [])
        self.assertEqual(self.engine.storage.revision(), 1)
        self.assertEqual(self.engine.storage.attempts()[0]['calls'], [])
        self.assertEqual(self.engine.storage.snapshot('default', []).workflow_states['background', '__agent__'].state,
                         {'updated': True})

    def test_parameter_changes_do_not_mutate_provider_defaults(self):
        fake = FakeModel()
        self.engine._models['analysis'] = fake
        self.workflow(lambda ctx: ctx.models.chat(ctx.assemble(), model='analysis', options={'temperature': 0.9}))
        self.run_turn()
        self.assertEqual(fake.options['temperature'], 0.1)

    def test_bad_time_is_rejected_before_a_model_call(self):
        with self.assertRaises(EMCError):
            self.engine.run(Event('user_message', content='input', occurred_at='2026-01-01T00:00:00'))
        self.assertEqual(self.engine.storage.attempts(), [])

    def test_schedule_consumption_rolls_back_on_revision_conflict(self):
        self.workflow(lambda ctx: (ctx.scheduler.schedule(Event('internal_trigger'), due_at=self.clock.now()), AssistantReply('ok'))[1])
        self.run_turn()

        def conflict(ctx):
            import sqlite3
            with closing(sqlite3.connect(self.engine.storage.path)) as db, db:
                db.execute('UPDATE runtime_meta SET revision=revision+1')
            ctx.state.set('failed', True)
            return ctx.models.chat(ctx.assemble())

        self.workflow(conflict)
        with self.assertRaises(EMCError):
            self.engine.run_due()
        self.assertEqual(len(self.engine.storage.pending_events()), 1)
        self.assertEqual(self.engine.storage.snapshot('default', []).workflow_states, {})
        self.assertEqual(len(self.engine.storage.history('default')), 2)
        self.assertEqual(self.engine.storage.attempts()[0]['calls'][0]['status'], 'succeeded')

    def test_duplicate_scheduled_event_ids_abort_whole_turn(self):
        event = Event('internal_trigger', event_id='repeated')

        def fn(ctx):
            ctx.scheduler.schedule(event, due_at=self.clock.now(), schedule_id='first')
            ctx.scheduler.schedule(event, due_at=self.clock.now(), schedule_id='second')
            return AssistantReply('candidate')

        self.workflow(fn)
        with self.assertRaises(EMCError):
            self.run_turn()
        self.assertEqual(self.engine.storage.pending_events(), [])
        self.assertEqual(self.engine.storage.revision(), 0)

    def test_scheduled_event_cannot_be_executed_early_as_normal_input(self):
        event = Event('internal_trigger', event_id='reserved')
        self.workflow(lambda ctx: (ctx.scheduler.schedule(event, due_at=self.clock.now() + timedelta(days=1)), AssistantReply('ok'))[1])
        self.run_turn()
        with self.assertRaises(EMCError):
            self.engine.run(event)
        self.assertEqual(len(self.engine.storage.attempts()), 1)
        self.assertEqual(len(self.engine.storage.pending_events()), 1)

    def test_silent_workflow_does_not_require_unused_default_credentials(self):
        config = offline_config(self.tmp, provider={'kind': 'openai-compatible', 'base_url': 'https://example.com',
                                                    'model': 'model', 'api_key_env': 'EMC_ABSENT_KEY'})
        engine = build_engine(config, clock=self.clock)
        engine.cognition = FunctionWorkflow(lambda ctx: None)
        with mock.patch.dict(os.environ, {}, clear=True):
            result = engine.run(Event('internal_trigger'))
        self.assertIsNone(result.reply)

    def test_schema_mismatch_does_not_touch_database(self):
        import sqlite3
        path = str(self.tmp / 'old.sqlite3')
        with closing(sqlite3.connect(path)) as db, db:
            db.execute('CREATE TABLE runtime_meta(singleton_id INTEGER, schema_version INTEGER)')
            db.execute('INSERT INTO runtime_meta VALUES (1,1)')
        before = __import__('pathlib').Path(path).read_bytes()
        from emc_runtime import Storage
        with self.assertRaises(EMCError):
            Storage(path)
        self.assertEqual(__import__('pathlib').Path(path).read_bytes(), before)
