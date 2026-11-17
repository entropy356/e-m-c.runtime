"""Persistence tests (T02, T10)."""

from __future__ import annotations

import sqlite3
import unittest
from unittest import mock

from emc_runtime import EMCError, Engine, EventBus, Storage
from emc_runtime.cognition import make_cognition
from emc_runtime.modules import ModuleRegistry
from emc_runtime.paths import PathSecurityError
from emc_runtime.storage import _connect

from .helpers import OfflineProviderStub, TempDirTestCase, offline_config


def build(config):
    storage = Storage(config.database)
    engine = Engine(
        config=config,
        storage=storage,
        registry=ModuleRegistry(config.modules),
        provider=OfflineProviderStub(),
        cognition=make_cognition(config.cognition_kind, config.cognition_params),
        bus=EventBus(),
    )
    return engine, storage


class RestartTest(TempDirTestCase):
    """T02: process restart loses nothing."""

    def test_state_and_history_survive_restart(self):
        from emc_runtime import Event

        counter = {
            "id": "counter",
            "entrypoint": "examples.counter_module:CounterModule",
            "scope": "agent",
            "config": {},
        }
        config = offline_config(self.tmp, modules=[counter])
        engine, storage = build(config)
        engine.run(Event(kind="user_message", content="一"))
        engine.run(Event(kind="user_message", content="二"))
        old_revision = storage.revision()
        old_history = [m.content for m in storage.history("default")]

        # "restart": brand-new storage and engine on the same database file
        engine2, storage2 = build(config)
        self.assertEqual(storage2.revision(), old_revision)
        self.assertEqual(
            [m.content for m in storage2.history("default")], old_history
        )
        engine2.run(Event(kind="user_message", content="三"))
        self.assertEqual(
            storage2.module_state("counter", "__agent__").state, {"count": 3}
        )
        self.assertEqual(storage2.revision(), old_revision + 1)


class ReadOnlyQueriesTest(TempDirTestCase):
    """T10: history/inspect/turn queries never mutate the database."""

    def test_read_only_queries_do_not_write(self):
        from emc_runtime import Event

        config = offline_config(self.tmp, modules=[])
        engine, storage = build(config)
        engine.run(Event(kind="user_message", content="一"))
        engine.run(Event(kind="user_message", content="二", conversation_id="other"))
        revision = storage.revision()

        self.assertEqual(len(storage.history("default")), 2)
        self.assertEqual(len(storage.recent_turns()), 2)
        self.assertEqual(len(storage.recent_turns("default")), 1)
        self.assertIsNone(storage.module_state("ghost", "default"))

        reopened = Storage(config.database)
        self.assertEqual(reopened.revision(), revision)
        raw = sqlite3.connect(config.database)
        try:
            rows = raw.execute("SELECT COUNT(*) FROM module_states").fetchone()[0]
            self.assertEqual(rows, 0)  # no state rows were invented
            conv_rows = raw.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            self.assertEqual(conv_rows, 2)
        finally:
            raw.close()


class StoragePathTest(TempDirTestCase):
    def test_connection_uses_absolute_literal_filename(self):
        path = self.tmp / 'nested' / '..' / 'experiment#mode=memory.sqlite3'
        with mock.patch('emc_runtime.storage.sqlite3.connect') as connect:
            with _connect(str(path)):
                connect.assert_called_once_with(
                    str(path.resolve()), timeout=10, isolation_level=None, uri=False)
            connect.return_value.close.assert_called_once_with()

    def test_storage_rejects_unsafe_paths_before_connecting(self):
        for path in ('', '   ', 'bad\x00name'):
            with self.subTest(path=path), mock.patch('emc_runtime.storage.sqlite3.connect') as connect:
                with self.assertRaises(PathSecurityError):
                    Storage(path)
                connect.assert_not_called()

    def test_fork_rejects_unsafe_paths_without_changing_source(self):
        storage = Storage(str(self.tmp / 'source.sqlite3'))
        for path in ('', '   ', 'bad\x00name', str(self.tmp)):
            with self.subTest(path=path), self.assertRaises(PathSecurityError):
                storage.fork(path)
        self.assertEqual(storage.revision(), 0)

    def test_fork_rejects_normalized_source_alias(self):
        storage = Storage(str(self.tmp / 'source.sqlite3'))
        with self.assertRaisesRegex(EMCError, 'must differ from source'):
            storage.fork(str(self.tmp / 'unused' / '..' / 'source.sqlite3'))
        self.assertEqual(storage.revision(), 0)

    def test_failed_fork_removes_only_its_new_file(self):
        storage = Storage(str(self.tmp / 'source.sqlite3'))
        target = self.tmp / 'branch.sqlite3'
        with mock.patch('emc_runtime.storage._connect', side_effect=RuntimeError('backup failed')):
            with self.assertRaisesRegex(RuntimeError, 'backup failed'):
                storage.fork(str(target))
        self.assertFalse(target.exists())
        self.assertEqual(storage.revision(), 0)

    def test_existing_fork_destination_is_preserved(self):
        storage = Storage(str(self.tmp / 'source.sqlite3'))
        target = self.tmp / 'existing.sqlite3'
        target.write_bytes(b'previous experiment')
        with self.assertRaises(FileExistsError):
            storage.fork(str(target))
        self.assertEqual(target.read_bytes(), b'previous experiment')


class SchemaGuardTest(TempDirTestCase):
    def test_foreign_schema_version_rejected(self):
        from emc_runtime import EMCError

        config = offline_config(self.tmp)
        engine, _ = build(config)
        from emc_runtime import Event

        engine.run(Event(kind="user_message", content="一"))
        # forge an incompatible future schema
        raw = sqlite3.connect(config.database)
        raw.execute("UPDATE runtime_meta SET schema_version=99")
        raw.commit()
        raw.close()
        with self.assertRaises(EMCError):
            Storage(config.database)


if __name__ == "__main__":
    unittest.main()
