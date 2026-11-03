"""Persistent conversational agent with ordered context and long-term memory."""

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import re
import sqlite3

from .model import DynamicsConfig, activate, decay, rank
from .providers import unit


def _text(value, field, maximum=12000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be nonempty text, at most {maximum} characters")
    return value.strip()


def _conversation_id(value):
    value = _text(value, "conversation_id", 128)
    if re.search(r"\s", value):
        raise ValueError("conversation_id cannot contain whitespace")
    return value


def validate_cognition(result, selected):
    if not isinstance(result, dict):
        raise ValueError("Cognition must return an object")
    answer = _text(result.get("answer"), "answer")
    allowed = {memory["id"] for memory in selected}
    used = result.get("used_memory_ids")
    if not isinstance(used, list) or any(type(item) is not int or item not in allowed for item in used):
        raise ValueError("Cognition cited a memory that was not provided; state was not changed")
    proposals = result.get("memories")
    if not isinstance(proposals, list) or len(proposals) > 3:
        raise ValueError("Cognition may propose at most three memories")
    memories = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("Each proposed memory must be an object")
        memories.append({"content": _text(proposal.get("content"), "memory.content")})
    return {
        "answer": answer,
        "used_memory_ids": sorted(set(used)),
        "memories": memories,
    }


def _normalized_content(value):
    return " ".join(value.casefold().split())


class Runtime:
    """One agent-wide memory store, with isolated ordered histories per conversation."""

    def __init__(self, database, provider, config=None, notifier=None):
        self.path = str(database)
        self.provider = provider
        self.config = config or DynamicsConfig()
        # Optional push sink (events.EventHub.publish). A viewer must never be
        # able to fail a turn, so every notification is fire-and-forget.
        self.notifier = notifier
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        initial = {"step": 0, "next_id": 1, "memories": []}
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS runtime ("
                "id INTEGER PRIMARY KEY CHECK(id=1), "
                "revision INTEGER NOT NULL, embedding_profile TEXT NOT NULL, "
                "config TEXT NOT NULL, state TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS conversations ("
                "id TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, "
                "seq INTEGER NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','assistant')), "
                "content TEXT NOT NULL, created_step INTEGER NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "UNIQUE(conversation_id, seq), "
                "FOREIGN KEY(conversation_id) REFERENCES conversations(id))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS rounds ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, step INTEGER NOT NULL UNIQUE, "
                "conversation_id TEXT NOT NULL, trace TEXT NOT NULL)"
            )
            db.execute(
                "INSERT OR IGNORE INTO runtime VALUES (1, 0, ?, ?, ?)",
                (
                    provider.profile,
                    json.dumps(self.config.to_dict(), sort_keys=True),
                    json.dumps(initial, ensure_ascii=False),
                ),
            )
            row = db.execute(
                "SELECT embedding_profile, config FROM runtime WHERE id=1"
            ).fetchone()
            if row[0] != provider.profile:
                raise ValueError("Embedding provider/model differs from this database; use a new --db path")
            if json.loads(row[1]) != self.config.to_dict():
                raise ValueError("Dynamics config differs from this database; use a new --db path")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            yield connection
        finally:
            connection.close()

    def _snapshot(self, conversation_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT revision, state FROM runtime WHERE id=1"
            ).fetchone()
            messages = db.execute(
                "SELECT id, seq, role, content, created_step FROM messages "
                "WHERE conversation_id=? ORDER BY seq",
                (conversation_id,),
            ).fetchall()
        history = [
            {"id": item[0], "seq": item[1], "role": item[2],
             "content": item[3], "created_step": item[4]}
            for item in messages
        ]
        return row[0], json.loads(row[1]), history

    def history(self, conversation_id="default"):
        conversation_id = _conversation_id(conversation_id)
        _, _, messages = self._snapshot(conversation_id)
        return messages

    @staticmethod
    def _dimension(state):
        if state["memories"]:
            return len(state["memories"][0]["vector"])
        return None

    def _embed(self, text, dimension=None, task=None):
        vector = unit(self.provider.embed(text, task))
        if dimension is not None and len(vector) != dimension:
            raise ValueError("Embedding dimensions changed; state was not changed")
        return vector

    def retrieve(self, text, top_k=None):
        text = _text(text, "query")
        _, state, _ = self._snapshot("default")
        if not state["memories"]:
            return []
        query = self._embed(text, self._dimension(state), "query")
        rows = rank(state["memories"], query, top_k or self.config.top_k)
        return rows

    def _commit_turn(self, revision, expected_last_seq, conversation_id, state, trace):
        encoded = json.dumps(state, ensure_ascii=False, allow_nan=False)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT revision FROM runtime WHERE id=1"
            ).fetchone()
            if row[0] != revision:
                raise ValueError("Agent state changed during this turn; retry the turn")
            current_last_seq = db.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()[0]
            if current_last_seq != expected_last_seq:
                raise ValueError("Conversation changed during this turn; reload and retry")
            db.execute(
                "INSERT OR IGNORE INTO conversations(id) VALUES (?)",
                (conversation_id,),
            )
            next_step = state["step"]
            db.execute(
                "INSERT INTO messages(conversation_id, seq, role, content, created_step) "
                "VALUES (?, ?, 'user', ?, ?)",
                (conversation_id, expected_last_seq + 1, trace["input"], next_step),
            )
            db.execute(
                "INSERT INTO messages(conversation_id, seq, role, content, created_step) "
                "VALUES (?, ?, 'assistant', ?, ?)",
                (conversation_id, expected_last_seq + 2, trace["cognition"]["answer"], next_step),
            )
            changed = db.execute(
                "UPDATE runtime SET revision=revision+1, state=? WHERE id=1 AND revision=?",
                (encoded, revision),
            ).rowcount
            if changed != 1:
                raise ValueError("Agent state changed during commit; retry the turn")
            db.execute(
                "INSERT INTO rounds(step, conversation_id, trace) VALUES (?, ?, ?)",
                (next_step, conversation_id,
                 json.dumps(trace, ensure_ascii=False, allow_nan=False)),
            )
            db.execute(
                "UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (conversation_id,),
            )
            db.execute("COMMIT")

    def _notify(self, event):
        if self.notifier is None:
            return
        try:
            self.notifier(event)
        except Exception:
            # Monitoring is best-effort; the committed turn is already durable.
            pass

    def turn(self, text, conversation_id="default"):
        try:
            return self._turn(text, conversation_id)
        except Exception as error:
            self._notify({
                "type": "error",
                "conversation_id": conversation_id,
                "error": f"{type(error).__name__}: {error}",
            })
            raise

    def _turn(self, text, conversation_id="default"):
        text = _text(text, "input")
        conversation_id = _conversation_id(conversation_id)
        revision, state, history = self._snapshot(conversation_id)
        expected_last_seq = history[-1]["seq"] if history else 0

        decayed = []
        for memory in state["memories"]:
            result = decay(memory, self.config)
            decayed.append({"id": memory["id"], **result})

        dimension = self._dimension(state)
        query = self._embed(text, dimension, "query") if state["memories"] else None
        rows = rank(state["memories"], query, self.config.top_k)
        selected = [row for row in rows if row["selected"]]

        # The provider receives every previous user/assistant message in order,
        # followed by the current user input. The current turn is committed only
        # after cognition and all embeddings succeed.
        context = [
            {"role": item["role"], "content": item["content"]}
            for item in history
        ]
        context.append({"role": "user", "content": text})
        cognition = validate_cognition(
            self.provider.cognize(context, selected), selected
        )

        activated = []
        used_ids = set(cognition["used_memory_ids"])
        for memory in state["memories"]:
            if memory["id"] in used_ids:
                result = activate(memory, self.config)
                memory["activated_count"] += 1
                memory["last_activated_step"] = state["step"] + 1
                activated.append({"id": memory["id"], **result})

        known = {_normalized_content(item["content"]) for item in state["memories"]}
        prepared = []
        for proposal in cognition["memories"]:
            normalized = _normalized_content(proposal["content"])
            if normalized in known:
                continue
            vector = self._embed(proposal["content"], dimension, "document")
            dimension = len(vector)
            prepared.append({
                **proposal,
                "id": state["next_id"] + len(prepared),
                "vector": vector,
                "weight": 1.0,
                "stability": 0.0,
                "created_step": state["step"] + 1,
                "activated_count": 0,
                "last_activated_step": None,
                "source_conversation_id": conversation_id,
            })
            known.add(normalized)

        state["memories"].extend(prepared)
        state["next_id"] += len(prepared)
        state["step"] += 1
        trace = {
            "step": state["step"],
            "conversation_id": conversation_id,
            "input": text,
            "answer": cognition["answer"],
            "mode": self.provider.mode,
            "chat_profile": getattr(self.provider, "chat_profile", None),
            "cognition_protocol": getattr(self.provider, "cognition_protocol", None),
            "embedding_profile": self.provider.profile,
            "config": self.config.to_dict(),
            "prior_message_count": len(history),
            "context_message_count": len(context),
            "prior_message_ids": [item["id"] for item in history],
            "decayed": decayed,
            "retrieval": rows,
            "cognition": cognition,
            "activated": activated,
            "written_ids": [item["id"] for item in prepared],
        }
        self._commit_turn(
            revision, expected_last_seq, conversation_id, state, trace
        )
        self._notify({
            "type": "turn",
            "step": trace["step"],
            "revision": revision + 1,
            "conversation_id": conversation_id,
            "input": trace["input"],
            "answer": trace["answer"],
            "used_memory_ids": trace["cognition"]["used_memory_ids"],
            "written_ids": trace["written_ids"],
            "activated_ids": [item["id"] for item in activated],
        })
        return trace

    def inspect(self, conversation_id="default"):
        conversation_id = _conversation_id(conversation_id)
        revision, state, messages = self._snapshot(conversation_id)
        visible_state = copy.deepcopy(state)
        for memory in visible_state["memories"]:
            memory["embedding_dimensions"] = len(memory.pop("vector"))
        with self.connect() as db:
            traces = [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT trace FROM rounds ORDER BY step DESC LIMIT 50"
                )
            ]
            conversations = [
                {"id": row[0], "message_count": row[1], "updated_at": row[2]}
                for row in db.execute(
                    "SELECT c.id, COUNT(m.id), c.updated_at "
                    "FROM conversations c LEFT JOIN messages m ON m.conversation_id=c.id "
                    "GROUP BY c.id ORDER BY c.updated_at DESC"
                )
            ]
        return {
            "revision": revision,
            "step": state["step"],
            "conversation_id": conversation_id,
            "messages": messages,
            "state": visible_state,
            "traces": traces,
            "conversations": conversations,
            "mode": self.provider.mode,
            "chat_profile": getattr(self.provider, "chat_profile", None),
            "cognition_protocol": getattr(self.provider, "cognition_protocol", None),
            "embedding_profile": self.provider.profile,
        }
