import io
import json
import sys
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from long_term_memory import events, viewer
from long_term_memory.__main__ import main
from long_term_memory.model import DynamicsConfig
from long_term_memory.providers import SYSTEM, OfflineProvider
from long_term_memory.runtime import Runtime


def wait_for_listener(hub, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if hub.listener_count:
            return True
        time.sleep(0.01)
    return False


class EventHubTests(unittest.TestCase):
    def test_events_arrive_in_order(self):
        with events.EventHub() as hub:
            received = []
            done = threading.Event()

            def reader():
                for event in events.iter_events(hub.port):
                    received.append(event)
                    if len(received) == 3:
                        done.set()
                        return

            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            self.assertTrue(wait_for_listener(hub))
            for step in range(3):
                hub.publish({"type": "turn", "step": step})
            self.assertTrue(done.wait(2.0))
            thread.join(2.0)
            self.assertEqual([item["step"] for item in received], [0, 1, 2])

    def test_every_listener_receives_every_event(self):
        with events.EventHub() as hub:
            received = {index: [] for index in range(3)}
            ready = [threading.Event() for _ in range(3)]

            def reader(index):
                connection = events.connect(hub.port)
                ready[index].set()
                try:
                    with connection.makefile("r", encoding="utf-8") as stream:
                        for line in stream:
                            received[index].append(json.loads(line))
                            if len(received[index]) == 3:
                                return
                finally:
                    connection.close()

            threads = [threading.Thread(target=reader, args=(index,), daemon=True)
                       for index in range(3)]
            for thread in threads:
                thread.start()
            for event in ready:
                self.assertTrue(event.wait(2.0))
            deadline = time.monotonic() + 3.0
            while hub.listener_count < 3 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(hub.listener_count, 3)

            for step in range(3):
                hub.publish({"type": "turn", "step": step})
            for thread in threads:
                thread.join(2.0)
            for index in range(3):
                self.assertEqual([item["step"] for item in received[index]], [0, 1, 2])

    def test_port_file_is_written_and_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "agent.sqlite3"
            with events.EventHub(database) as hub:
                hub.write_port_file()
                info = events.read_event_file(database)
                self.assertEqual(info["port"], hub.port)
                self.assertEqual(info["db"], str(database))
                self.assertTrue(events.event_file(database).exists())
            self.assertFalse(events.event_file(database).exists())
            self.assertIsNone(events.read_event_file(database))

    def test_publish_without_listeners_never_blocks_or_raises(self):
        with events.EventHub() as hub:
            started = time.monotonic()
            for step in range(600):
                hub.publish({"type": "turn", "step": step})
            self.assertLess(time.monotonic() - started, 2.0)

    def test_dead_listener_does_not_break_later_listeners(self):
        with events.EventHub() as hub:
            stale = events.connect(hub.port)
            self.assertTrue(wait_for_listener(hub))
            stale.close()
            hub.publish({"type": "turn", "step": 1})

            received, connected = [], threading.Event()

            def reader():
                connection = events.connect(hub.port)
                connected.set()
                try:
                    with connection.makefile("r", encoding="utf-8") as stream:
                        for line in stream:
                            received.append(json.loads(line))
                            return
                finally:
                    connection.close()

            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            self.assertTrue(connected.wait(2.0))
            # The hub registers a connection on accept(), which can lag the
            # client's connect(); publish until the new listener sees one.
            deadline = time.monotonic() + 3.0
            while not received and time.monotonic() < deadline:
                hub.publish({"type": "turn", "step": 2})
                time.sleep(0.02)
            thread.join(2.0)
            self.assertEqual([item["step"] for item in received], [2])

    def test_closed_hub_stops_publishing(self):
        hub = events.EventHub()
        hub.close()
        self.assertFalse(hub.publish({"type": "turn", "step": 1}))


class RuntimeNotificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "agent.sqlite3"
        self.seen = []
        self.runtime = Runtime(self.database, OfflineProvider(), DynamicsConfig(),
                               notifier=self.seen.append)

    def test_commit_pushes_turn_event(self):
        result = {"answer": "ok", "used_memory_ids": [],
                  "memories": [{"content": "用户偏好简短的书面进度。"}]}
        with patch.object(self.runtime.provider, "cognize", return_value=result):
            self.runtime.turn("你好")
        event = self.seen[-1]
        self.assertEqual(event["type"], "turn")
        self.assertEqual(event["step"], 1)
        self.assertEqual(event["revision"], 1)
        self.assertEqual(event["written_ids"], [1])
        self.assertEqual(event["conversation_id"], "default")

    def test_failed_turn_pushes_error_event(self):
        with patch.object(self.runtime.provider, "cognize", side_effect=ValueError("boom")):
            with self.assertRaises(ValueError):
                self.runtime.turn("你好")
        self.assertEqual(self.seen[-1]["type"], "error")
        self.assertIn("boom", self.seen[-1]["error"])

    def test_broken_notifier_does_not_fail_the_turn(self):
        def explode(_event):
            raise RuntimeError("viewer went away")

        self.runtime.notifier = explode
        result = {"answer": "ok", "used_memory_ids": [], "memories": []}
        with patch.object(self.runtime.provider, "cognize", return_value=result):
            trace = self.runtime.turn("你好")
        self.assertEqual(trace["step"], 1)


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "agent.sqlite3"
        self.provider = OfflineProvider()
        self.runtime = Runtime(self.database, self.provider, DynamicsConfig())
        for text, content in (("first", "用户偏好简短的书面进度。"),
                              ("second", "用户下周去杭州出差。")):
            result = {"answer": "ok", "used_memory_ids": [],
                      "memories": [{"content": content}]}
            with patch.object(self.provider, "cognize", return_value=result):
                self.runtime.turn(text)

    def arguments(self, **overrides):
        base = dict(db=str(self.database), port=None, conversation="default",
                    sort="weight", rounds=6, full=False, decay=False,
                    json=False, no_clear=True, payload=False, protocol=None)
        base.update(overrides)
        return Namespace(**base)

    def test_renders_once_per_pushed_event(self):
        stream = io.StringIO()
        event = {"type": "turn", "step": 3, "input": "第三次输入",
                 "used_memory_ids": [1], "written_ids": [3]}
        renders = viewer.watch(self.database, 4321, source=[event], stream=stream,
                               clear=False, now=lambda: 0)
        text = stream.getvalue()
        self.assertEqual(renders, 2)
        self.assertEqual(text.count("● 监控中 127.0.0.1:4321"), 2)
        self.assertIn("推送 step 3 · 引用 1 · 新增 3", text)
        self.assertIn("用户偏好简短的书面进度", text)
        self.assertIn("轮次活动", text)

    def test_monitor_does_not_render_the_last_round_dialogue(self):
        stream = io.StringIO()
        viewer.watch(self.database, 1, source=[], stream=stream, clear=False)
        text = stream.getvalue()
        self.assertNotIn("最近一轮", text)
        self.assertNotIn("you>", text)
        self.assertNotIn("agent>", text)
        self.assertIn("轮次活动", text)          # 只有轮次表格，没有问答块

    def test_payload_mode_prints_the_request_actually_sent(self):
        stream = io.StringIO()
        viewer.watch(self.database, 1, source=[], stream=stream, clear=False,
                     payload_mode=True, protocol="tool")
        text = stream.getvalue()
        self.assertIn("[1] system", text)
        self.assertIn("[2] system  长期记忆数据", text)
        self.assertIn("[3] user  (seq 1 · step 1)", text)
        self.assertIn("[4] assistant  (seq 2 · step 1)", text)
        self.assertIn("[5] user  (seq 3 · step 2)  <- 本轮输入", text)
        self.assertLess(text.index("[3] user"), text.index("[4] assistant"))
        self.assertLess(text.index("[4] assistant"), text.index("[5] user"))
        self.assertIn("submit_cognition", text)          # tool 协议的提示词
        self.assertIn("used_memory_ids.enum = [1]", text)  # 只列注入的记忆 ID
        self.assertNotIn("轮次活动", text)                # 表格视图被替换掉
        self.assertIn("协议 tool", text)
        self.assertIn("step 2 第 1 个请求", text)
        self.assertIn("第 2 个请求", text)
        self.assertNotIn("下一轮的新输入", text)            # 回放已发生的请求，不再预告

    def test_payload_mode_json_carries_the_text(self):
        stream = io.StringIO()
        viewer.watch(self.database, 1, source=[], stream=stream, json_mode=True,
                     payload_mode=True, protocol="json")
        payload = json.loads(stream.getvalue().strip())
        self.assertEqual(payload["protocol"], "json")
        self.assertEqual(len(payload["messages"]), 4)
        self.assertIn("[1] system", payload["payload"])
        self.assertNotIn("submit_cognition", payload["payload"])
        self.assertIn(SYSTEM.splitlines()[0], payload["payload"])

    def test_cli_accepts_protocol_before_the_subcommand(self):
        with patch.object(sys, "argv",
                          ["demo3", "--db", str(self.database), "--conversation", "default",
                           "--protocol", "json", "context", "--payload"]), \
             redirect_stdout(io.StringIO()) as output:
            main()
        self.assertIn(SYSTEM.splitlines()[0], output.getvalue())

    def test_error_event_is_shown(self):
        stream = io.StringIO()
        viewer.watch(self.database, 1, source=[{"type": "error", "error": "boom"}],
                     stream=stream, clear=False)
        self.assertIn("上一轮失败：boom", stream.getvalue())

    def test_json_mode_prints_one_line_per_render(self):
        stream = io.StringIO()
        viewer.watch(self.database, 1, source=[{"type": "turn", "step": 3}],
                     stream=stream, json_mode=True)
        payloads = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["step"], 2)
        self.assertIsNone(payloads[0]["event"])
        self.assertEqual(payloads[1]["event"]["step"], 3)
        self.assertEqual(len(payloads[1]["memories"]), 2)

    def test_missing_endpoint_explains_how_to_start_the_agent(self):
        with self.assertRaises(ValueError) as caught:
            viewer._resolve_endpoint(self.database, None)
        self.assertIn("--events", str(caught.exception))

    def test_run_watch_follows_a_live_hub(self):
        hub = events.EventHub(self.database)
        hub.write_port_file()
        stream = io.StringIO()
        thread = threading.Thread(
            target=viewer.run_watch,
            args=(self.arguments(), stream, None, 2),
            daemon=True,
        )
        thread.start()
        deadline = time.monotonic() + 3.0
        while thread.is_alive() and time.monotonic() < deadline:
            hub.publish({"type": "turn", "step": 2, "input": "second"})
            thread.join(0.02)
        hub.close()
        thread.join(2.0)
        self.assertFalse(thread.is_alive())
        self.assertGreaterEqual(stream.getvalue().count("● 监控中"), 2)


if __name__ == "__main__":
    unittest.main()
