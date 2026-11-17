"""SQLite persistence: snapshots, one-transaction commits and read-only queries.

Business writes for one turn (messages, states, schedules, the turn record and the
global revision bump) happen inside a single ``BEGIN IMMEDIATE`` transaction.
A revision mismatch between snapshot and commit raises ConcurrencyConflict;
the engine never retries the turn automatically.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .contracts import ConcurrencyConflict, EMCError, Message, dumps_json
from .paths import writable_file

SCHEMA_VERSION = 2

_SELECT_REVISION = "SELECT revision FROM runtime_meta WHERE singleton_id=1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runtime_meta (
    singleton_id    INTEGER PRIMARY KEY CHECK(singleton_id = 1),
    revision        INTEGER NOT NULL,
    schema_version  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    role            TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    turn_id         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(conversation_id, seq)
);
CREATE TABLE IF NOT EXISTS module_states (
    module_id       TEXT NOT NULL,
    scope_key       TEXT NOT NULL,
    state_version   INTEGER NOT NULL,
    state_json      TEXT NOT NULL,
    updated_revision INTEGER NOT NULL,
    PRIMARY KEY(module_id, scope_key)
);
CREATE TABLE IF NOT EXISTS turns (
    turn_id         TEXT PRIMARY KEY,
    event_id        TEXT NOT NULL UNIQUE,
    revision        INTEGER NOT NULL,
    conversation_id TEXT NOT NULL,
    event_kind      TEXT NOT NULL,
    trace_json      TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS workflow_states (
    module_id TEXT NOT NULL, scope_key TEXT NOT NULL,
    state_version INTEGER NOT NULL, state_json TEXT NOT NULL,
    updated_revision INTEGER NOT NULL, PRIMARY KEY(module_id, scope_key)
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY, event_id TEXT NOT NULL, event_json TEXT NOT NULL,
    started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
    stage TEXT, error TEXT, turn_id TEXT
);
CREATE TABLE IF NOT EXISTS model_calls (
    attempt_id TEXT NOT NULL, seq INTEGER NOT NULL, record_json TEXT NOT NULL,
    PRIMARY KEY(attempt_id, seq)
);
CREATE TABLE IF NOT EXISTS scheduled_events (
    schedule_id TEXT PRIMARY KEY, due_at TEXT NOT NULL, event_id TEXT NOT NULL UNIQUE,
    event_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', completed_turn_id TEXT
);
CREATE INDEX IF NOT EXISTS scheduled_due ON scheduled_events(status, due_at);
CREATE INDEX IF NOT EXISTS turns_conversation ON turns(conversation_id, revision);

"""


@dataclass(frozen=True)
class ModuleStateEntry:
    state_version: int
    state: Any


@dataclass(frozen=True)
class Snapshot:
    revision: int
    history: tuple[Message, ...]
    module_states: dict[str, ModuleStateEntry]
    workflow_states: dict[tuple[str, str], ModuleStateEntry] = field(default_factory=dict)


@dataclass(frozen=True)
class ModuleStateWrite:
    module_id: str
    scope_key: str
    state_version: int
    state: Any


@contextmanager
def _connect(path: str):
    # Always use a normalized literal filename, never SQLite URI parameters.
    database = writable_file(path)
    connection = sqlite3.connect(str(database), timeout=10, isolation_level=None, uri=False)
    try:
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA foreign_keys=ON")
        yield connection
    finally:
        connection.close()


class Storage:
    """Owns the database file. All methods are explicit about write intent."""

    def __init__(self, path: str):
        if str(path).strip() == ":memory:":
            raise EMCError("Storage requires a file-backed database")
        self.path = str(writable_file(path))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with _connect(self.path) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='runtime_meta'").fetchone():
                existing = db.execute("SELECT schema_version FROM runtime_meta WHERE singleton_id=1").fetchone()
                if existing is None or existing[0] != SCHEMA_VERSION:
                    raise EMCError(f"database requires schema {SCHEMA_VERSION}; use a new database path")
            db.executescript(_SCHEMA)
            db.execute(
                "INSERT OR IGNORE INTO runtime_meta(singleton_id, revision, schema_version) "
                "VALUES (1, 0, ?)",
                (SCHEMA_VERSION,),
            )
            row = db.execute(
                "SELECT schema_version FROM runtime_meta WHERE singleton_id=1"
            ).fetchone()
            if row is None or row[0] != SCHEMA_VERSION:
                raise EMCError(
                    f"database {self.path} uses schema version "
                    f"{row[0] if row else 'unknown'}, this runtime requires "
                    f"{SCHEMA_VERSION}; use a new database path"
                )

    # ------------------------------------------------------------------ reads

    def snapshot(
        self, conversation_id: str, module_keys: Sequence[tuple[str, str]]
    ) -> Snapshot:
        """One consistent read of revision, history and enabled module states."""
        with _connect(self.path) as db:
            db.execute("BEGIN")
            try:
                revision = db.execute(_SELECT_REVISION).fetchone()[0]
                history = tuple(
                    Message(
                        conversation_id=row[0],
                        seq=row[1],
                        role=row[2],
                        content=row[3],
                        turn_id=row[4],
                    message_id=row[5],
                    created_at=row[6],
                    )
                    for row in db.execute(
                        "SELECT conversation_id, seq, role, content, turn_id, id, created_at FROM messages "
                        "WHERE conversation_id=? ORDER BY seq",
                        (conversation_id,),
                    )
                )
                states: dict[str, ModuleStateEntry] = {}
                for module_id, scope_key in module_keys:
                    row = db.execute(
                        "SELECT state_version, state_json FROM module_states "
                        "WHERE module_id=? AND scope_key=?",
                        (module_id, scope_key),
                    ).fetchone()
                    if row is not None:
                        states[module_id] = ModuleStateEntry(
                            state_version=row[0],
                            state=json.loads(row[1]),
                        )
                workflow_states = {
                    (row[0], row[1]): ModuleStateEntry(row[2], json.loads(row[3]))
                    for row in db.execute(
                        "SELECT module_id, scope_key, state_version, state_json FROM workflow_states "
                        "WHERE scope_key IN (?, '__agent__')", (conversation_id,)
                    )
                }
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
        return Snapshot(revision=revision, history=history, module_states=states, workflow_states=workflow_states)

    def history(self, conversation_id: str) -> list[Message]:
        """Read-only conversation history ordered by seq."""
        with _connect(self.path) as db:
            return [
                Message(
                    conversation_id=row[0],
                    seq=row[1],
                    role=row[2],
                    content=row[3],
                    turn_id=row[4],
                    message_id=row[5],
                    created_at=row[6],
                )
                for row in db.execute(
                    "SELECT conversation_id, seq, role, content, turn_id, id, created_at FROM messages "
                    "WHERE conversation_id=? ORDER BY seq",
                    (conversation_id,),
                )
            ]

    def revision(self) -> int:
        with _connect(self.path) as db:
            return db.execute(_SELECT_REVISION).fetchone()[0]

    def module_state(self, module_id: str, scope_key: str) -> ModuleStateEntry | None:
        """Read-only module state lookup; never creates missing state."""
        with _connect(self.path) as db:
            row = db.execute(
                "SELECT state_version, state_json FROM module_states "
                "WHERE module_id=? AND scope_key=?",
                (module_id, scope_key),
            ).fetchone()
        if row is None:
            return None
        return ModuleStateEntry(state_version=row[0], state=json.loads(row[1]))

    def recent_turns(self, conversation_id: str | None = None, limit: int = 20) -> list[dict]:
        """Newest-first turn records with parsed traces; read-only."""
        with _connect(self.path) as db:
            if conversation_id is None:
                rows = db.execute(
                    "SELECT turn_id, event_id, revision, conversation_id, event_kind, "
                    "trace_json, created_at FROM turns ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT turn_id, event_id, revision, conversation_id, event_kind, "
                    "trace_json, created_at FROM turns WHERE conversation_id=? "
                    "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (conversation_id, limit),
                ).fetchall()
        return [
            {
                "turn_id": row[0],
                "event_id": row[1],
                "revision": row[2],
                "conversation_id": row[3],
                "event_kind": row[4],
                "trace": json.loads(row[5]),
                "created_at": row[6],
            }
            for row in rows
        ]

    # ----------------------------------------------------------------- writes

    def commit_turn(
        self,
        *,
        expected_revision: int,
        conversation_id: str,
        turn_id: str,
        event_id: str,
        event_kind: str,
        new_messages: Sequence[tuple[str, str]],
        module_writes: Iterable[ModuleStateWrite],
        trace: dict,
        workflow_writes: Sequence[ModuleStateWrite] = (),
        scheduled: Sequence[dict] = (),
        cancellations: Sequence[str] = (),
        consumed_schedule: str | None = None,
        committed_at: str | None = None,
    ) -> int:
        """Atomically persist one turn; returns the new revision.

        ``new_messages`` is a list of ``(role, content)`` pairs appended after
        the conversation's current last seq. Raises ConcurrencyConflict when the
        global revision or the conversation's last seq changed since the
        snapshot; nothing is written in that case.

        Engine turns use their attempt ID as ``turn_id``; committing also marks
        that attempt complete, if present.
        """
        encoded_trace = dumps_json(trace, "turn trace")
        with _connect(self.path) as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                current_revision = db.execute(_SELECT_REVISION).fetchone()[0]
                if current_revision != expected_revision:
                    raise ConcurrencyConflict(
                        f"revision changed from {expected_revision} to {current_revision} "
                        "during this turn; the model request was not replayed. "
                        "Reload and retry explicitly if you accept the cost."
                    )
                last_seq = db.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id=?",
                    (conversation_id,),
                ).fetchone()[0]
                db.execute(
                    "INSERT OR IGNORE INTO conversations(conversation_id) VALUES (?)",
                    (conversation_id,),
                )
                for offset, (role, content) in enumerate(new_messages, start=1):
                    db.execute(
                        "INSERT INTO messages(conversation_id, seq, role, content, turn_id, created_at) "
                        "VALUES (?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))",
                        (conversation_id, last_seq + offset, role, content, turn_id, committed_at),
                    )
                new_revision = current_revision + 1
                for write in module_writes:
                    db.execute(
                        "INSERT INTO module_states(module_id, scope_key, state_version, "
                        "state_json, updated_revision) VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(module_id, scope_key) DO UPDATE SET "
                        "state_version=excluded.state_version, "
                        "state_json=excluded.state_json, "
                        "updated_revision=excluded.updated_revision",
                        (
                            write.module_id,
                            write.scope_key,
                            write.state_version,
                            dumps_json(write.state, f"module {write.module_id} state"),
                            new_revision,
                        ),
                    )
                db.execute(
                    "INSERT INTO turns(turn_id, event_id, revision, conversation_id, "
                    "event_kind, trace_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        turn_id,
                        event_id,
                        new_revision,
                        conversation_id,
                        event_kind,
                        encoded_trace,
                    ),
                )
                for write in workflow_writes:
                    db.execute(
                        "INSERT INTO workflow_states VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(module_id, scope_key) DO UPDATE SET "
                        "state_version=excluded.state_version, state_json=excluded.state_json, "
                        "updated_revision=excluded.updated_revision",
                        (write.module_id, write.scope_key, write.state_version,
                         dumps_json(write.state, 'workflow state'), new_revision),
                    )
                if consumed_schedule is not None:
                    changed_schedule = db.execute(
                        "UPDATE scheduled_events SET status='completed', completed_turn_id=? "
                        "WHERE schedule_id=? AND status='pending'",
                        (turn_id, consumed_schedule),
                    ).rowcount
                    if changed_schedule != 1:
                        raise ConcurrencyConflict('scheduled event is no longer pending')
                for schedule_id in cancellations:
                    db.execute("UPDATE scheduled_events SET status='cancelled' WHERE schedule_id=? AND status='pending'", (schedule_id,))
                for job in scheduled:
                    scheduled_event_id = job['event']['event_id']
                    if db.execute("SELECT 1 FROM turns WHERE event_id=?", (scheduled_event_id,)).fetchone():
                        raise EMCError('cannot schedule an event that already committed')
                    db.execute(
                        "INSERT INTO scheduled_events(schedule_id, due_at, event_id, event_json) VALUES (?, ?, ?, ?)",
                        (job['schedule_id'], job['due_at'], scheduled_event_id,
                         dumps_json(job['event'], 'scheduled event')),
                    )
                db.execute("UPDATE attempts SET status='committed', finished_at=?, turn_id=?, stage='commit' WHERE attempt_id=?",
                           (committed_at, turn_id, turn_id))
                changed = db.execute(
                    "UPDATE runtime_meta SET revision=? WHERE singleton_id=1 AND revision=?",
                    (new_revision, current_revision),
                ).rowcount
                if changed != 1:
                    raise ConcurrencyConflict(
                        "revision changed during commit; nothing was written"
                    )
                db.execute(
                    "UPDATE conversations SET updated_at=CURRENT_TIMESTAMP "
                    "WHERE conversation_id=?",
                    (conversation_id,),
                )
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
        return new_revision

    def event_committed(self, event_id: str) -> bool:
        with _connect(self.path) as db:
            return db.execute('SELECT 1 FROM turns WHERE event_id=?', (event_id,)).fetchone() is not None

    def experiences(self, *, revision: int | None = None, conversation_id: str | None = None,
                    text: str | None = None, before_id: int | None = None,
                    after_id: int | None = None, limit: int = 100) -> list[Message]:
        """Newest matching originals, returned chronologically, with stable source IDs.

        A revision bound gives a workflow an immutable view of committed experiences.
        Text is a literal substring filter, not a particular retrieval algorithm.
        """
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise EMCError('experience limit must be an integer in [1, 10000]')
        clauses, values = [], []
        for value, clause in ((revision, 't.revision <= ?'), (conversation_id, 'm.conversation_id = ?'),
                              (text, 'instr(m.content, ?) > 0'), (before_id, 'm.id < ?'),
                              (after_id, 'm.id > ?')):
            if value is not None:
                clauses.append(clause)
                values.append(value)
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        with _connect(self.path) as db:
            rows = db.execute(
                'SELECT m.conversation_id, m.seq, m.role, m.content, m.turn_id, m.id, m.created_at '
                'FROM messages m JOIN turns t ON t.turn_id=m.turn_id' + where + ' ORDER BY m.id DESC LIMIT ?',
                (*values, limit),
            ).fetchall()
        return [Message(*row) for row in reversed(rows)]

    def start_attempt(self, attempt_id: str, event: dict, started_at: str) -> None:
        with _connect(self.path) as db:
            db.execute('INSERT INTO attempts(attempt_id,event_id,event_json,started_at,status,stage) VALUES (?,?,?,?,?,?)',
                       (attempt_id, event['event_id'], dumps_json(event, 'event'), started_at, 'running', 'snapshot'))

    def finish_failed_attempt(self, attempt_id: str, *, stage: str, error: str, finished_at: str) -> None:
        with _connect(self.path) as db:
            db.execute("UPDATE attempts SET status='failed', stage=?, error=?, finished_at=? WHERE attempt_id=? AND status='running'",
                       (stage, error, finished_at, attempt_id))

    def record_call(self, attempt_id: str, seq: int, record: dict) -> None:
        with _connect(self.path) as db:
            db.execute('INSERT INTO model_calls VALUES (?,?,?) ON CONFLICT(attempt_id,seq) DO UPDATE SET record_json=excluded.record_json',
                       (attempt_id, seq, dumps_json(record, 'model call')))

    def attempts(self, limit: int = 20) -> list[dict]:
        with _connect(self.path) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute('SELECT * FROM attempts ORDER BY rowid DESC LIMIT ?', (limit,)).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item['event'] = json.loads(item.pop('event_json'))
                item['calls'] = [json.loads(r[0]) for r in db.execute(
                    'SELECT record_json FROM model_calls WHERE attempt_id=? ORDER BY seq', (row['attempt_id'],))]
                result.append(item)
            return result

    def event_schedule(self, event_id: str) -> str | None:
        with _connect(self.path) as db:
            row = db.execute("SELECT schedule_id FROM scheduled_events WHERE event_id=? AND status='pending'", (event_id,)).fetchone()
        return row[0] if row else None

    def schedule_pending(self, schedule_id: str) -> bool:
        with _connect(self.path) as db:
            return db.execute("SELECT 1 FROM scheduled_events WHERE schedule_id=? AND status='pending'", (schedule_id,)).fetchone() is not None

    def pending_events(self, *, due_at: str | None = None, limit: int = 100) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise EMCError('schedule limit must be an integer in [1, 10000]')
        with _connect(self.path) as db:
            clause = ' AND due_at<=?' if due_at is not None else ''
            args = (due_at, limit) if due_at is not None else (limit,)
            rows = db.execute("SELECT schedule_id,due_at,event_json FROM scheduled_events WHERE status='pending'" + clause + ' ORDER BY due_at,schedule_id LIMIT ?', args).fetchall()
        return [{'schedule_id': row[0], 'due_at': row[1], 'event': json.loads(row[2])} for row in rows]

    def fork(self, destination: str) -> Storage:
        """Copy one consistent database snapshot for an independent experiment."""
        target = writable_file(destination)
        if target.resolve() == Path(self.path).resolve():
            raise EMCError('fork destination must differ from source')
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids overwriting any existing experiment.
        target.touch(exist_ok=False)
        try:
            with _connect(self.path) as source, _connect(str(target)) as dest:
                source.backup(dest)
            return Storage(str(target))
        except Exception:
            target.unlink(missing_ok=True)
            raise
