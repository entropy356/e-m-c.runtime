import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from long_term_memory import viewer
from long_term_memory.__main__ import _chat_history, main
from long_term_memory.model import DynamicsConfig
from long_term_memory.providers import SYSTEM, OfflineProvider
from long_term_memory.runtime import Runtime


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "agent.sqlite3"
        self.provider = OfflineProvider()
        self.runtime = Runtime(self.path, self.provider, DynamicsConfig())
        self.turn("first", "alpha", ["用户偏好简短的书面进度，每周五发送一次。"])
        self.turn("second", "alpha", ["用户下周去杭州出差，需要提前订酒店。"])
        self.turn("third", "beta", ["用户使用 Python 3.14 和 PowerShell 7。"])

    def turn(self, text, conversation, contents):
        result = {"answer": "ok", "used_memory_ids": [],
                  "memories": [{"content": content} for content in contents]}
        with patch.object(self.provider, "cognize", return_value=result):
            return self.runtime.turn(text, conversation)

    def dump(self):
        with closing(sqlite3.connect(self.path)) as db:
            return list(db.iterdump())

    def test_load_reports_runtime_and_memories(self):
        view = viewer.load(self.path)
        self.assertEqual(view["step"], 3)
        self.assertEqual(view["revision"], 3)
        self.assertEqual(view["rounds"], 3)
        self.assertEqual(len(view["memories"]), 3)
        self.assertEqual({item["source_conversation_id"] for item in view["memories"]},
                         {"alpha", "beta"})
        self.assertTrue(all(item["dimensions"] > 0 for item in view["memories"]))
        self.assertEqual(len(view["conversations"]), 2)

    def test_memory_age_uses_current_step(self):
        view = viewer.snapshot(self.path, sort="id")
        self.assertEqual([item["age"] for item in view["memories"]], [2, 1, 0])

    def test_snapshot_sorts_by_weight_descending(self):
        weights = [item["weight"] for item in viewer.snapshot(self.path, sort="weight")["memories"]]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_snapshot_filters_by_conversation(self):
        view = viewer.snapshot(self.path, conversation="beta")
        self.assertEqual(len(view["memories"]), 1)
        self.assertIn("PowerShell", view["memories"][0]["content"])
        self.assertIn("没有产生长期记忆",
                      viewer.render(viewer.snapshot(self.path, conversation="missing")))

    def test_render_shows_table_legend_and_optional_decay_column(self):
        text = viewer.render(viewer.snapshot(self.path))
        self.assertIn("WEIGHT", text)
        self.assertIn("每周五", text)
        self.assertIn("AGE = step - created_step", text)
        self.assertNotIn("NEXT", text)
        self.assertIn("NEXT", viewer.render(viewer.snapshot(self.path), decay_preview=True))

    def test_render_truncates_unless_full(self):
        self.turn("long", "alpha", ["很长的记忆内容" * 12])
        view = viewer.snapshot(self.path)
        self.assertIn("…", viewer.render(view))
        self.assertNotIn("…", viewer.render(view, full=True))

    def test_render_memory_detail_and_missing_id(self):
        view = viewer.snapshot(self.path)
        memory_id = view["memories"][0]["id"]
        detail = viewer.render_memory(view, memory_id)
        self.assertIn(f"memory #{memory_id}", detail)
        self.assertIn("weight_after_next_decay", detail)
        self.assertIn("embedding_dimensions", detail)
        with self.assertRaises(ValueError):
            viewer.render_memory(view, 999)

    def test_empty_store_message(self):
        empty = Path(self.temp.name) / "empty.sqlite3"
        Runtime(empty, OfflineProvider(), DynamicsConfig())
        self.assertIn("暂无长期记忆", viewer.render(viewer.snapshot(empty)))

    def test_missing_database_is_reported(self):
        with self.assertRaises(ValueError):
            viewer.load(Path(self.temp.name) / "missing.sqlite3")

    def test_non_demo3_database_is_reported(self):
        other = Path(self.temp.name) / "other.sqlite3"
        with closing(sqlite3.connect(other)) as db:
            db.execute("CREATE TABLE something(x)")
            db.commit()
        with self.assertRaises(ValueError):
            viewer.load(other)

    def test_viewer_never_writes(self):
        before = self.dump()
        viewer.load(self.path)
        viewer.render(viewer.snapshot(self.path), decay_preview=True)
        viewer.render_memory(viewer.snapshot(self.path), 1)
        self.assertEqual(before, self.dump())

    def test_cli_memories_command(self):
        with patch.object(sys, "argv", ["demo3", "--db", str(self.path), "memories"]), \
             redirect_stdout(io.StringIO()) as output:
            main()
        text = output.getvalue()
        self.assertIn("WEIGHT", text)
        self.assertIn("每周五", text)
        self.assertIn("revision 3", text)

    def test_cli_memory_detail_and_json(self):
        with patch.object(sys, "argv",
                          ["demo3", "--db", str(self.path), "memories", "1"]), \
             redirect_stdout(io.StringIO()) as output:
            main()
        self.assertIn("memory #1", output.getvalue())

        with patch.object(sys, "argv",
                          ["demo3", "--db", str(self.path), "memories", "--json"]), \
             redirect_stdout(io.StringIO()) as output:
            main()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["step"], 3)
        self.assertEqual(payload["sort"], "weight")
        self.assertEqual(len(payload["memories"]), 3)

    def test_context_lists_messages_and_rounds(self):
        view = viewer.load_context(self.path, "alpha")
        self.assertEqual([item["seq"] for item in view["messages"]], [1, 2, 3, 4])
        self.assertEqual([item["role"] for item in view["messages"]],
                         ["user", "assistant", "user", "assistant"])
        self.assertEqual(len(view["rounds"]), 2)
        self.assertEqual(view["rounds"][0]["written"], [1])
        text = viewer.render_context(view)
        self.assertIn("SEQ", text)
        self.assertIn("SEL", text)
        self.assertIn("first", text)
        self.assertIn("ANSWER", text)
        self.assertIn("ok", text)

    def test_rounds_table_truncates_answers_like_inputs(self):
        rounds = [{"step": 1, "input": "问题", "answer": "答" * 80,
                   "used": [], "written": [], "selected": 0, "retrieved": 0},
                  {"step": 2, "input": "问题", "answer": "",
                   "used": [], "written": [], "selected": 0, "retrieved": 0}]
        text = "\n".join(viewer._rounds_table(rounds))
        self.assertIn("ANSWER", text)
        self.assertIn("答" * 21 + "…", text)      # width 22: 21 chars + ellipsis
        self.assertIn("问题", text)
        self.assertIn(" -  ", text)               # empty answer falls back to "-"
        self.assertIn("答" * 80, "\n".join(viewer._rounds_table(rounds, full=True)))

    def test_context_unknown_conversation_is_empty(self):
        view = viewer.load_context(self.path, "missing")
        self.assertEqual(view["messages"], [])
        self.assertIn("还没有消息", viewer.render_context(view))

    def test_render_history_shows_messages_and_keeps_the_tail(self):
        view = viewer.load_context(self.path, "alpha")
        text = viewer.render_history(view)
        self.assertIn("SEQ", text)
        self.assertIn("first", text)
        self.assertIn("second", text)
        tail = viewer.render_history(view, limit=2)
        self.assertIn("second", tail)
        self.assertNotIn("first", tail)
        self.assertIn("只显示最近 2 条", tail)
        self.assertIn("（还没有消息）",
                      viewer.render_history(viewer.load_context(self.path, "missing")))

    def test_chat_history_command_reads_without_touching_the_model(self):
        self.assertIn("second", _chat_history(self.path, "alpha", "all"))
        limited = _chat_history(self.path, "alpha", "2")
        self.assertIn("只显示最近 2 条", limited)
        self.assertIn("用法：/history", _chat_history(self.path, "alpha", "abc"))
        # 默认只给最近 10 条，这里的会话只有 4 条，不出现截断提示
        self.assertNotIn("只显示", _chat_history(self.path, "alpha", ""))

    def test_context_payload_follows_protocol(self):
        view = viewer.load_context(self.path, "alpha")
        tool_payload = viewer.render_payload(view, "tool")
        self.assertIn("submit_cognition", tool_payload)
        self.assertIn("long_term_memories", tool_payload)
        self.assertIn("用户偏好简短的书面进度", tool_payload)
        self.assertIn("first", tool_payload)
        self.assertIn("[3] user", tool_payload)
        json_payload = viewer.render_payload(view, "json")
        self.assertNotIn("submit_cognition", json_payload)
        self.assertIn(SYSTEM.splitlines()[0], json_payload)

    def test_payload_contains_only_the_injected_memories(self):
        path = Path(self.temp.name) / "top1.sqlite3"
        runtime = Runtime(path, OfflineProvider(), DynamicsConfig(top_k=1))
        for text, content in (("a", "记忆甲"), ("b", "记忆乙"), ("c", "记忆丙")):
            result = {"answer": "ok", "used_memory_ids": [],
                      "memories": [{"content": content}]}
            with patch.object(runtime.provider, "cognize", return_value=result):
                runtime.turn(text, "default")
        view = viewer.load_context(path, "default")
        last = view["rounds"][-1]
        self.assertEqual(last["retrieved"], 2)       # 第 3 轮时已有 2 条候选
        self.assertEqual(len(last["injected"]), 1)   # top_k=1 只注入 1 条
        text = viewer.render_payload(view, "tool")
        injected = last["injected"][0]
        self.assertIn(injected["content"], text)
        self.assertIn(f"注入 1/{last['retrieved']}", text)
        for other in {"记忆甲", "记忆乙", "记忆丙"} - {injected["content"]}:
            self.assertNotIn(other, text)


        with patch.object(sys, "argv",
                          ["demo3", "--db", str(self.path),
                           "--conversation", "alpha", "context"]), \
             redirect_stdout(io.StringIO()) as output:
            main()
        self.assertIn("SEL", output.getvalue())

        with patch.object(sys, "argv",
                          ["demo3", "--db", str(self.path),
                           "--conversation", "alpha", "context", "--payload"]), \
             redirect_stdout(io.StringIO()) as output:
            main()
        self.assertIn("[1] system", output.getvalue())
        self.assertIn("[3] user", output.getvalue())

    def test_standalone_script_imports_package_from_its_own_directory(self):
        root = Path(__file__).resolve().parent.parent
        completed = subprocess.run(
            [sys.executable, "viewer.py", "--db", str(self.path),
             "--conversation", "alpha", "context"],
            cwd=root / "long_term_memory", capture_output=True, text=True,
            encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("SEQ", completed.stdout)
        self.assertIn("first", completed.stdout)


if __name__ == "__main__":
    unittest.main()
