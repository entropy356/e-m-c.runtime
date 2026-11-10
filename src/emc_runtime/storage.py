"""SQLite persistence: snapshots, one-transaction commits and read-only queries.

All writes for one turn (messages, module states, the turn record and the
global revision bump) happen inside a single ``BEGIN IMMEDIATE`` transaction.
A revision mismatch between snapshot and commit raises ConcurrencyConflict;
the engine never retries the turn automatically.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .contracts import ConcurrencyConflict, EMCError, Message, dumps_json

SCHEMA_VERSION = 1

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
    event_id        TEXT NOT NULL,
    revision        INTEGER NOT NULL,
    conversation_id TEXT NOT NULL,
    event_kind      TEXT NOT NULL,
    trace_json      TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
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


@dataclass(frozen=True)
class ModuleStateWrite:
    module_id: str
    scope_key: str
    state_version: int
    state: Any


@contextmanager
def _connect(path: str):
    connection = sqlite3.connect(path, timeout=10, isolation_level=None)
    try:
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA foreign_keys=ON")
        yield connection
    finally:
        connection.close()


class Storage:
    """Owns the database file. All methods are explicit about write intent."""

    def __init__(self, path: str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with _connect(self.path) as db:
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
                    )
                    for row in db.execute(
                        "SELECT conversation_id, seq, role, content, turn_id FROM messages "
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
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
        return Snapshot(revision=revision, history=history, module_states=states)

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
                )
                for row in db.execute(
                    "SELECT conversation_id, seq, role, content, turn_id FROM messages "
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
    ) -> int:
        """Atomically persist one turn; returns the new revision.

        ``new_messages`` is a list of ``(role, content)`` pairs appended after
        the conversation's current last seq. Raises ConcurrencyConflict when the
        global revision or the conversation's last seq changed since the
        snapshot; nothing is written in that case.
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
                        "INSERT INTO messages(conversation_id, seq, role, content, turn_id) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (conversation_id, last_seq + offset, role, content, turn_id),
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
