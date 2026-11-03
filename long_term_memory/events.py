"""Push channel from the agent process to read-only viewers.

The agent owns a loopback TCP socket and writes one JSON line per committed
turn. A viewer blocks on ``recv()`` and re-renders only when a line arrives, so
nothing polls: no refresh interval, no mtime checks, no queries while idle.

    agent  : python -m long_term_memory --db data/agent.sqlite3 chat --events
    viewer : python -m long_term_memory --db data/agent.sqlite3 watch

``publish()`` only enqueues, so a slow or stuck viewer can never stall a turn.
"""

import json
import os
from pathlib import Path
import queue
import socket
import threading
import time

HOST = "127.0.0.1"
SUFFIX = ".events.json"
MAX_QUEUE = 256
SEND_TIMEOUT = 5.0


def event_file(db_path):
    """Sidecar file that tells a viewer which port to connect to."""
    return Path(str(db_path) + SUFFIX)


def read_event_file(db_path):
    """Endpoint written by a running agent, or None when nobody is listening."""
    path = event_file(db_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("port"), int):
        return None
    return data


class EventHub:
    """Publish JSON events to every connected viewer, without blocking callers."""

    def __init__(self, db_path=None, host=HOST):
        self.host = host
        self.db = None if db_path is None else str(db_path)
        self.file = None if db_path is None else event_file(db_path)
        self._listeners = []
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=MAX_QUEUE)
        self._closed = False
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, 0))
        self._server.listen(8)
        self.port = self._server.getsockname()[1]
        self._threads = [
            threading.Thread(target=self._accept_loop, name="events-accept", daemon=True),
            threading.Thread(target=self._send_loop, name="events-send", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    @property
    def endpoint(self):
        return f"{self.host}:{self.port}"

    @property
    def listener_count(self):
        with self._lock:
            return len(self._listeners)

    def _accept_loop(self):
        while not self._closed:
            try:
                connection, _ = self._server.accept()
            except OSError:
                return
            connection.settimeout(SEND_TIMEOUT)
            with self._lock:
                if self._closed:
                    connection.close()
                    return
                self._listeners.append(connection)

    def _send_loop(self):
        while True:
            event = self._queue.get()
            if event is None:
                return
            line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
            with self._lock:
                listeners = list(self._listeners)
            for connection in listeners:
                try:
                    connection.sendall(line)
                except OSError:
                    # A paused or dead viewer must never wedge the agent.
                    with self._lock:
                        if connection in self._listeners:
                            self._listeners.remove(connection)
                    try:
                        connection.close()
                    except OSError:
                        pass

    def publish(self, event):
        """Queue one event. Never raises, never blocks."""
        if self._closed:
            return False
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Keep the newest events: a viewer that fell far behind only needs
            # the current state, which it re-reads from the database anyway.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except (queue.Empty, queue.Full):
                return False
        return True

    def write_port_file(self):
        if self.file is None:
            return None
        self.file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "host": self.host,
            "port": self.port,
            "pid": os.getpid(),
            "db": self.db,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        return self.file

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass
        try:
            self._server.close()
        except OSError:
            pass
        with self._lock:
            listeners, self._listeners = self._listeners, []
        for connection in listeners:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass
        self._remove_port_file()

    def _remove_port_file(self):
        if self.file is None or not self.file.exists():
            return
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
            if data.get("port") == self.port:
                self.file.unlink()
        except (OSError, ValueError):
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


def connect(port, host=HOST, timeout=3.0):
    """Open a viewer connection; raises OSError when the agent is not listening."""
    connection = socket.create_connection((host, int(port)), timeout=timeout)
    connection.settimeout(None)  # blocking reads: the viewer waits for pushes
    return connection


class EventStream:
    """One viewer connection. Connects eagerly, then blocks until an event lands.

    Several streams can attach to the same hub: every listener gets every event.
    """

    def __init__(self, port, host=HOST, timeout=3.0):
        self._connection = connect(port, host, timeout)
        self._reader = None

    def __iter__(self):
        self._reader = self._connection.makefile("r", encoding="utf-8", newline="\n")
        for line in self._reader:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                yield event

    def close(self):
        for item in (self._reader, self._connection):
            if item is not None:
                try:
                    item.close()
                except OSError:
                    pass
        self._reader = None


def iter_events(port, host=HOST, timeout=3.0):
    """Yield events as they are pushed; returns when the agent closes the socket."""
    stream = EventStream(port, host, timeout)
    try:
        yield from stream
    finally:
        stream.close()
