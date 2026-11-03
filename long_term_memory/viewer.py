"""Read-only inspector for the long-term memory store and conversation context.

Deliberately standalone: no provider, no network, no writes. It opens the
SQLite file in read-only mode, so inspecting a live database is safe.

Runs both ways:
    python -m long_term_memory --db data/agent.sqlite3 memories
    python -m long_term_memory --db data/agent.sqlite3 context
    python -m long_term_memory --db data/agent.sqlite3 context --payload
    python -m long_term_memory --db data/agent.sqlite3 watch --payload
    python long_term_memory/viewer.py --db data/agent.sqlite3 context

``--payload`` replays the request actually sent for the last round: the system
prompt, only the memories that were injected, and that round's messages in the
exact order the model receives them.
"""

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import unicodedata
from urllib.parse import quote

try:
    from . import events
    from .model import DynamicsConfig, decay
    from .providers import SYSTEM, SYSTEM_TOOL
except ImportError:  # executed as a plain script: make the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from long_term_memory import events
    from long_term_memory.model import DynamicsConfig, decay
    from long_term_memory.providers import SYSTEM, SYSTEM_TOOL

SORTS = ("weight", "id", "age", "activations")
DEFAULT_DATABASE = Path(__file__).resolve().parent.parent / "data" / "agent.sqlite3"


def _open(path):
    database = Path(path)
    if not database.exists():
        raise ValueError(f"database not found: {database} (pass --db to an existing file)")
    uri = "file:" + quote(database.resolve().as_posix()) + "?mode=ro"
    return database, sqlite3.connect(uri, uri=True)


def _runtime_row(connection, database):
    try:
        row = connection.execute(
            "SELECT revision, embedding_profile, config, state FROM runtime WHERE id=1"
        ).fetchone()
    except sqlite3.OperationalError:
        raise ValueError(f"{database} is not a demo3 database (missing runtime table)") from None
    if row is None:
        raise ValueError(f"{database} has no runtime row")
    return row


def load(path):
    """Runtime metadata plus every long-term memory, without writing anything."""
    database, connection = _open(path)
    try:
        revision, embedding_profile, config, state = _runtime_row(connection, database)
        state = json.loads(state)
        conversations = [
            {"id": item[0], "messages": item[1], "updated_at": item[2]}
            for item in connection.execute(
                "SELECT c.id, COUNT(m.id), c.updated_at FROM conversations c "
                "LEFT JOIN messages m ON m.conversation_id=c.id "
                "GROUP BY c.id ORDER BY c.updated_at DESC"
            )
        ]
        rounds = connection.execute("SELECT COUNT(*) FROM rounds").fetchone()[0]
    finally:
        connection.close()

    step = state.get("step", 0)
    memories = []
    for memory in state.get("memories", []):
        created = memory.get("created_step")
        memories.append({
            "id": memory["id"],
            "content": memory["content"],
            "weight": memory["weight"],
            "stability": memory["stability"],
            "activated_count": memory.get("activated_count", 0),
            "last_activated_step": memory.get("last_activated_step"),
            "created_step": created,
            "age": None if created is None else step - created,
            "source_conversation_id": memory.get("source_conversation_id"),
            "dimensions": len(memory.get("vector") or []),
        })
    return {
        "path": str(database),
        "revision": revision,
        "step": step,
        "next_id": state.get("next_id"),
        "embedding_profile": embedding_profile,
        "config": json.loads(config),
        "memories": memories,
        "conversations": conversations,
        "rounds": rounds,
    }


def load_context(path, conversation="default"):
    """One conversation's ordered messages plus the memory activity of each round."""
    database, connection = _open(path)
    try:
        revision, _, _, state = _runtime_row(connection, database)
        state = json.loads(state)
        messages = [
            {"seq": item[0], "role": item[1], "step": item[2], "content": item[3]}
            for item in connection.execute(
                "SELECT seq, role, created_step, content FROM messages "
                "WHERE conversation_id=? ORDER BY seq",
                (conversation,),
            )
        ]
        rounds = []
        for step, trace in connection.execute(
            "SELECT step, trace FROM rounds WHERE conversation_id=? ORDER BY step",
            (conversation,),
        ):
            trace = json.loads(trace)
            retrieval = trace.get("retrieval") or []
            injected = [{"id": item["id"], "content": item["content"]}
                        for item in retrieval if item.get("selected")]
            rounds.append({
                "step": step,
                "input": trace.get("input", ""),
                "answer": trace.get("answer", ""),
                "protocol": trace.get("cognition_protocol"),
                "used": trace.get("cognition", {}).get("used_memory_ids", []),
                "written": trace.get("written_ids", []),
                "selected": len(injected),
                "retrieved": len(retrieval),
                "injected": injected,
            })
        conversations = [
            {"id": item[0], "messages": item[1]}
            for item in connection.execute(
                "SELECT c.id, COUNT(m.id) FROM conversations c "
                "LEFT JOIN messages m ON m.conversation_id=c.id "
                "GROUP BY c.id ORDER BY c.updated_at DESC"
            )
        ]
    finally:
        connection.close()
    return {
        "path": str(database),
        "revision": revision,
        "step": state.get("step", 0),
        "conversation": conversation,
        "messages": messages,
        "rounds": rounds,
        "conversations": conversations,
        "memories": [{"id": item["id"], "content": item["content"]}
                     for item in state.get("memories", [])],
    }


def _sorted(memories, sort):
    if sort == "weight":
        return sorted(memories, key=lambda item: (-item["weight"], item["id"]))
    if sort == "age":
        return sorted(memories, key=lambda item: (item["created_step"] or 0, item["id"]))
    if sort == "activations":
        return sorted(memories, key=lambda item: (-item["activated_count"], item["id"]))
    return sorted(memories, key=lambda item: item["id"])


def snapshot(path, sort="weight", conversation=None):
    """Filtered, sorted memory view ready for rendering or JSON output."""
    view = load(path)
    memories = view["memories"]
    if conversation:
        memories = [item for item in memories if item["source_conversation_id"] == conversation]
    view["memories"] = _sorted(memories, sort)
    view["sort"] = sort
    view["conversation"] = conversation
    return view


def _display_width(text):
    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)


def _pad(text, width, align):
    gap = max(0, width - _display_width(text))
    return (" " * gap + text) if align == "r" else (text + " " * gap)


def _table_lines(headers, rows, aligns):
    widths = [_display_width(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], _display_width(cell))
    lines = ["  ".join(_pad(cell, widths[i], aligns[i])
                       for i, cell in enumerate(headers)).rstrip()]
    for row in rows:
        lines.append("  ".join(_pad(cell, widths[i], aligns[i])
                               for i, cell in enumerate(row)).rstrip())
    return lines


def _rule(lines, minimum=56, maximum=100):
    """Separator instead of blank lines: PowerShell drops empty native output lines."""
    width = max([_display_width(line) for line in lines] + [minimum])
    return "-" * min(width, maximum)


def _shorten(text, limit):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _ids(values):
    return "-" if not values else ",".join(str(item) for item in values)


def _next_weight(memory, config):
    probe = {"weight": memory["weight"], "stability": memory["stability"]}
    return decay(probe, config)["weight_after"]


def _header(view):
    config = view["config"]
    return [
        view["path"],
        (f"revision {view['revision']} · step {view['step']} · "
         f"{len(view['memories'])} memories · {len(view['conversations'])} conversations · "
         f"{view['rounds']} rounds"),
        "embedding: " + view["embedding_profile"],
        (f"config: {config['mode']} · decay {config['decay_rate']} · "
         f"protect {config['stability_protection']} · recover {config['recovery_rate']} · "
         f"gain {config['stability_gain']} · top_k {config['top_k']}"),
    ]


def _memory_table(view, decay_preview=False, full=False, width=42):
    headers = ["ID", "WEIGHT", "STAB", "ACT", "AGE", "LAST", "CONV"]
    if decay_preview:
        headers.append("NEXT")
    headers.append("CONTENT")
    config = DynamicsConfig(**view["config"])
    rows = []
    for memory in view["memories"]:
        last = memory["last_activated_step"]
        age = memory["age"]
        row = [
            str(memory["id"]),
            f"{memory['weight']:.3f}",
            f"{memory['stability']:.3f}",
            str(memory["activated_count"]),
            "-" if age is None else str(age),
            "-" if last is None else str(last),
            memory["source_conversation_id"] or "-",
        ]
        if decay_preview:
            row.append(f"{_next_weight(memory, config):.3f}")
        row.append(memory["content"] if full else _shorten(memory["content"], width))
        rows.append(row)
    return _table_lines(headers, rows, "r" * (len(headers) - 1) + "l")


def _rounds_table(rounds, full=False, width=22):
    rows = [[str(item["step"]), _ids(item["used"]), _ids(item["written"]),
             f"{item['selected']}/{item['retrieved']}",
             item["input"] if full else _shorten(item["input"], width),
             (item.get("answer") or "-") if full
             else _shorten(item.get("answer") or "-", width)]
            for item in rounds]
    return _table_lines(["STEP", "USED", "WRITTEN", "SEL", "INPUT", "ANSWER"], rows,
                        "rlllll")


def render(view, decay_preview=False, full=False, width=42):
    header = _header(view)
    memories = view["memories"]
    if not memories:
        conversation = view["conversation"]
        message = (f"（会话 {conversation} 没有产生长期记忆）" if conversation
                   else "（暂无长期记忆）")
        return "\n".join(header + [_rule(header), message])

    table = _memory_table(view, decay_preview, full, width)
    rule = _rule(header + table)
    legend = "WEIGHT/STAB 是上一轮提交时的存储值；AGE = step - created_step；LAST = 最后使用轮次"
    if decay_preview:
        legend += "；NEXT = 再衰减一轮的预测值（不写库）"
    return "\n".join(header + [rule] + table + [rule, legend])


def render_memory(view, memory_id):
    memory = next((item for item in view["memories"] if item["id"] == memory_id), None)
    if memory is None:
        known = ", ".join(str(item["id"]) for item in view["memories"]) or "无"
        raise ValueError(f"memory {memory_id} not found (available: {known})")
    config = DynamicsConfig(**view["config"])
    rows = [
        ["content", memory["content"]],
        ["weight", f"{memory['weight']:.6f}"],
        ["stability", f"{memory['stability']:.6f}"],
        ["activated_count", str(memory["activated_count"])],
        ["last_activated_step", "-" if memory["last_activated_step"] is None else str(memory["last_activated_step"])],
        ["created_step", "-" if memory["created_step"] is None else str(memory["created_step"])],
        ["age", "-" if memory["age"] is None else str(memory["age"])],
        ["source_conversation_id", memory["source_conversation_id"] or "-"],
        ["embedding_dimensions", str(memory["dimensions"])],
        ["weight_after_next_decay", f"{_next_weight(memory, config):.6f}"],
    ]
    header = [f"memory #{memory_id}", view["path"]]
    table = _table_lines(["FIELD", "VALUE"], rows, "ll")
    rule = _rule(header + table)
    return "\n".join(header + [rule] + table)


def render_context(view, full=False, width=56):
    conversations = "  ".join(f"{item['id']}({item['messages']})" for item in view["conversations"])
    header = [
        view["path"],
        (f"conversation {view['conversation']} · {len(view['messages'])} messages · "
         f"{len(view['rounds'])} rounds · step {view['step']}"),
        "conversations: " + (conversations or "无"),
    ]
    if not view["messages"]:
        rule = _rule(header)
        return "\n".join(header + [rule, f"（会话 {view['conversation']} 还没有消息）"])

    rows = [[str(item["seq"]), item["role"], str(item["step"]),
             item["content"] if full else _shorten(item["content"], width)]
            for item in view["messages"]]
    table = _table_lines(["SEQ", "ROLE", "STEP", "CONTENT"], rows, "rlll")

    rounds = _rounds_table(view["rounds"], full)

    rule = _rule(header + table + rounds)
    legend = ("SEL = 检索命中 Top-K / 候选总数；USED 是本轮回答实际引用的记忆 ID；"
              "WRITTEN 是本轮新写入的记忆 ID；ANSWER 是本轮 assistant 回答（过长截断）")
    return "\n".join(header + [rule] + table + [rule] + rounds + [rule, legend])


def render_history(view, limit=None, full=False, width=60):
    """One conversation's messages only, oldest first; limit keeps the newest ones."""
    messages = view["messages"]
    shown = messages if not limit else messages[-limit:]
    header = [
        f"conversation {view['conversation']} · {len(messages)} messages · "
        f"{len(view['rounds'])} rounds · step {view['step']}",
    ]
    if not messages:
        return "\n".join(header + [_rule(header), "（还没有消息）"])
    if len(shown) < len(messages):
        header.append(f"只显示最近 {len(shown)} 条，共 {len(messages)} 条（/history all 看全部）")
    rows = [[str(item["seq"]), item["role"], str(item["step"]),
             item["content"] if full else _shorten(item["content"], width)]
            for item in shown]
    table = _table_lines(["SEQ", "ROLE", "STEP", "CONTENT"], rows, "rlll")
    rule = _rule(header + table)
    return "\n".join(header + [rule] + table)


def render_payload(view, protocol="tool", banner=True):
    """The first request body actually sent: only the memories that were injected."""
    system = SYSTEM_TOOL if protocol == "tool" else SYSTEM
    rounds = view.get("rounds") or []
    last = rounds[-1] if rounds else None
    step = last["step"] if last else None
    injected = (last.get("injected") or []) if last else []

    history, current = [], None
    if last is not None:
        history = [item for item in view["messages"] if item["step"] < step]
        current = next((item for item in view["messages"]
                        if item["step"] == step and item["role"] == "user"), None)
        if current is None:  # trace kept, message pruned: fall back to the trace
            current = {"seq": None, "step": step, "role": "user", "content": last["input"]}
    sent = history + ([current] if current else [])

    memory_block = json.dumps({"long_term_memories": injected},
                              ensure_ascii=False, indent=2)
    lines = []
    if banner:
        scope = (f"step {step} 第 1 个请求（记忆操作） · 注入 {len(injected)}/{last['retrieved']} 条记忆"
                 if last else "尚无已完成轮次 · 下一轮的请求骨架")
        lines += [
            f"protocol: {protocol} · conversation: {view['conversation']} · {scope} · "
            f"{len(sent)} 条消息",
            "",
        ]
    lines += [
        f"[1] system  ({len(system)} 字)",
        system,
        "",
        "[2] system  长期记忆数据（JSON，只含实际注入的记忆）",
        memory_block,
    ]
    if protocol == "tool":
        enum = [item["id"] for item in injected]
        lines.append("    tools: submit_cognition · used_memory_ids.enum = "
                     + json.dumps(enum, ensure_ascii=False))
    lines.append("")
    index = 3
    for message in sent:
        order = []
        if message.get("seq") is not None:
            order.append(f"seq {message['seq']}")
        if message.get("step") is not None:
            order.append(f"step {message['step']}")
        suffix = f"  ({' · '.join(order)})" if order else ""
        if message is current:
            suffix += "  <- 本轮输入"
        lines.append(f"[{index}] {message['role']}{suffix}")
        lines.append(message["content"])
        lines.append("")
        index += 1
    if last is None:
        lines.append(f"[{index}] user  <- 下一轮的新输入追加在这里")
        lines.append("")
        lines.append("注：还没有已提交的轮次，以上是下一轮请求的骨架；"
                     "system 提示词随认知协议变化（--protocol tool|json|text）。")
    else:
        lines.append(f"注：这是 step {step} 第 1 个请求——记忆块只有检索命中并注入的 "
                     f"{len(injected)} 条（候选 {last['retrieved']} 条），未命中的不出现；"
                     "system 提示词随认知协议变化（--protocol tool|json|text）。")
        if protocol == "tool":
            lines.append("    第 2 个请求 = 上面这份 messages + assistant 的 submit_cognition 调用 "
                         "+ tool 结果，tool_choice=none，正文即回答。")
        else:
            lines.append("    第 2 个请求 = 上面这份 messages + assistant 的 JSON 结果 "
                         "+ 一条衔接消息，正文即回答。")
    return "\n".join(lines)


def monitor_view(path, conversation="default", sort="weight", rounds_limit=6):
    """Memories plus one conversation's context, read once per pushed event."""
    view = snapshot(path, sort=sort)
    context = load_context(path, conversation)
    rounds = context["rounds"]
    return {
        "path": view["path"],
        "revision": view["revision"],
        "step": view["step"],
        "rounds": view["rounds"],
        "embedding_profile": view["embedding_profile"],
        "config": view["config"],
        "memories": view["memories"],
        "sort": sort,
        "conversation": conversation,
        "messages": context["messages"],
        "conversations": context["conversations"],
        "recent": rounds[-rounds_limit:] if rounds_limit else rounds,
    }


def _event_line(event):
    """One line describing the push that triggered this refresh."""
    if not event:
        return None
    if event.get("type") == "error":
        return "上一轮失败：" + _shorten(str(event.get("error")), 80)
    return (f"推送 step {event.get('step')} · 引用 {_ids(event.get('used_memory_ids') or [])}"
            f" · 新增 {_ids(event.get('written_ids') or [])}"
            f" · {_shorten(str(event.get('input') or ''), 34)}")


def render_monitor(view, event=None, full=False, width=38, decay_preview=False, now=None):
    stamp = time.strftime("%H:%M:%S", time.localtime(now))
    config = view["config"]
    header = [
        view["path"],
        (f"● 监控中 {view['endpoint']} · {stamp} · revision {view['revision']} · "
         f"step {view['step']} · {len(view['memories'])} memories · "
         f"{len(view['conversations'])} conversations · {view['rounds']} rounds"),
        "embedding: " + view["embedding_profile"],
        (f"config: {config['mode']} · decay {config['decay_rate']} · "
         f"protect {config['stability_protection']} · recover {config['recovery_rate']} · "
         f"gain {config['stability_gain']} · top_k {config['top_k']}"),
        f"conversation {view['conversation']} · {len(view['messages'])} messages",
    ]
    notice = _event_line(event)
    if notice:
        header.append(notice)

    table = _memory_table(view, decay_preview, full, width) if view["memories"] else []
    rule = _rule(header + table)
    lines = list(header)
    lines.append(rule)
    lines.extend(table or ["（暂无长期记忆）"])
    if view["recent"]:
        lines.append(rule)
        lines.append(f"轮次活动（最近 {len(view['recent'])} 轮，会话 {view['conversation']}）")
        lines.extend(_rounds_table(view["recent"], full))
    lines.append(rule)
    lines.append("由主进程推送事件触发刷新（不轮询）；Ctrl+C 退出")
    return "\n".join(lines)


def _enable_ansi():
    """Best-effort VT processing so the clear-screen escape works in conhost too."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _clear(stream):
    stream.write("\x1b[2J\x1b[H")


def watch(database, port, host=events.HOST, conversation="default", sort="weight",
          rounds_limit=6, full=False, decay_preview=False, json_mode=False,
          clear=True, stream=None, source=None, limit=None, now=time.time,
          payload_mode=False, protocol="tool"):
    """Render once, then once per pushed event. `source` is injectable for tests."""
    stream = stream or sys.stdout
    endpoint = f"{host}:{port}"
    renders = 0

    def draw(event=None):
        nonlocal renders
        if payload_mode:
            context = load_context(database, conversation)
            if json_mode:
                print(json.dumps({
                    "path": context["path"],
                    "endpoint": endpoint,
                    "conversation": conversation,
                    "protocol": protocol,
                    "messages": context["messages"],
                    "event": event,
                    "payload": render_payload(context, protocol, banner=False),
                }, ensure_ascii=False), file=stream)
            else:
                if clear:
                    _clear(stream)
                stamp = time.strftime("%H:%M:%S", time.localtime(now()))
                rounds = context["rounds"]
                scope = (f"step {rounds[-1]['step']} 第 1 个请求" if rounds
                         else "下一轮骨架")
                header = [
                    context["path"],
                    (f"● 监控中 {endpoint} · {stamp} · conversation {conversation} · "
                     f"{scope} · 协议 {protocol}"),
                ]
                notice = _event_line(event)
                if notice:
                    header.append(notice)
                body = render_payload(context, protocol, banner=False)
                _emit("\n".join(header + [_rule(header), body]), stream)
            stream.flush()
            renders += 1
            return

        view = monitor_view(database, conversation, sort, rounds_limit)
        view["endpoint"] = endpoint
        if json_mode:
            payload = {key: view[key] for key in
                       ("path", "revision", "step", "rounds", "conversation",
                        "memories", "recent")}
            payload["endpoint"] = endpoint
            payload["event"] = event
            print(json.dumps(payload, ensure_ascii=False), file=stream)
        else:
            if clear:
                _clear(stream)
            _emit(render_monitor(view, event, full, 38, decay_preview, now()), stream)
        stream.flush()
        renders += 1

    iterator = iter(source if source is not None else events.iter_events(port, host))
    try:
        draw()
        for event in iterator:
            draw(event)
            if limit is not None and renders >= limit:
                break
    except KeyboardInterrupt:
        if not json_mode:
            _emit("\n已退出监控。", stream)
        return renders
    finally:
        closer = getattr(source, "close", None)
        if callable(closer):
            closer()
    if not json_mode and limit is None:
        _emit("\nagent 已关闭事件通道，监控结束。", stream)
    return renders


def _resolve_endpoint(database, port):
    if port:
        return int(port), events.HOST
    info = events.read_event_file(database)
    if not info:
        raise ValueError(
            f"no event endpoint for {database}: start the agent with --events first "
            f"(expected {events.event_file(database)})"
        )
    return int(info["port"]), info.get("host", events.HOST)


def run_watch(args, stream=None, source=None, limit=None):
    stream = stream or sys.stdout
    port, host = _resolve_endpoint(args.db, args.port)
    if source is None:
        try:
            source = events.EventStream(port, host)
        except KeyboardInterrupt:
            return 0
        except OSError as error:
            raise ValueError(
                f"cannot reach the agent at {host}:{port} ({error}); "
                "it may have exited, restart it with --events"
            ) from None
    return watch(
        args.db, port, host,
        conversation=args.conversation,
        sort=args.sort,
        rounds_limit=args.rounds,
        full=args.full,
        decay_preview=args.decay,
        json_mode=args.json,
        clear=not args.no_clear and stream.isatty() and _enable_ansi(),
        stream=stream,
        source=source,
        limit=limit,
        payload_mode=args.payload,
        protocol=args.protocol or os.environ.get("AGENT_COGNITION_PROTOCOL", "tool"),
    )


def add_watch_arguments(parser):
    parser.add_argument("--port", type=int,
                        help="事件端口；默认读取 <db>.events.json")
    parser.add_argument("--sort", choices=SORTS, default="weight")
    parser.add_argument("--rounds", type=int, default=6,
                        help="显示最近多少轮活动（0 = 全部）")
    parser.add_argument("--decay", action="store_true",
                        help="多显示一列下一轮衰减预测")
    parser.add_argument("--full", action="store_true", help="不截断文本")
    parser.add_argument("--json", action="store_true", help="每个事件输出一行 JSON")
    parser.add_argument("--no-clear", action="store_true", help="不清屏，便于记录日志")
    parser.add_argument("--payload", action="store_true",
                        help="回放最近一轮第 1 个请求：system 提示词 + 注入的记忆 + 该轮消息")


def add_memory_arguments(parser):
    parser.add_argument("id", nargs="?", type=int, help="Show one memory in full")
    parser.add_argument("--sort", choices=SORTS, default="weight")
    parser.add_argument("--conversation", help="Only memories created in this conversation")
    parser.add_argument("--decay", action="store_true", help="Add the predicted weight after one more decay")
    parser.add_argument("--full", action="store_true", help="Do not truncate memory content")
    parser.add_argument("--json", action="store_true", help="Print the view as JSON")


def add_context_arguments(parser):
    parser.add_argument("--full", action="store_true", help="Do not truncate message content")
    parser.add_argument("--json", action="store_true", help="Print the view as JSON")
    parser.add_argument("--payload", action="store_true",
                        help="Replay the request actually sent for the last round "
                             "(injected memories only)")


def _emit(text, stream=None):
    # Print line by line: PowerShell drops empty lines from a single multi-line write.
    # Resolve sys.stdout at call time so redirect_stdout keeps working.
    for line in text.splitlines():
        print(line, file=stream or sys.stdout)


def run_memories(args):
    view = snapshot(args.db, sort=args.sort, conversation=args.conversation)
    if args.json:
        _emit(json.dumps(view, ensure_ascii=False, indent=2))
    elif args.id is not None:
        _emit(render_memory(view, args.id))
    else:
        _emit(render(view, decay_preview=args.decay, full=args.full))


def run_context(args):
    view = load_context(args.db, args.conversation)
    if args.json:
        _emit(json.dumps(view, ensure_ascii=False, indent=2))
    elif args.payload:
        protocol = args.protocol or os.environ.get("AGENT_COGNITION_PROTOCOL", "tool")
        _emit(render_payload(view, protocol))
    else:
        _emit(render_context(view, full=args.full))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="viewer.py", description="只读查看 Demo 3 的长期记忆与对话上下文"
    )
    parser.add_argument("--db", default=str(DEFAULT_DATABASE), help="数据库路径")
    parser.add_argument("--conversation", default="default", help="会话 ID（context 使用）")
    parser.add_argument("--protocol", choices=("tool", "json", "text"),
                        help="--payload 使用哪个认知协议的 system 提示词")
    sub = parser.add_subparsers(dest="command")
    memories = sub.add_parser("memories", help="查看长期记忆（默认）")
    add_memory_arguments(memories)
    context = sub.add_parser("context", help="查看对话上下文与每轮记忆活动")
    add_context_arguments(context)
    monitor = sub.add_parser("watch", aliases=["monitor"],
                             help="监控模式：主进程推送事件时刷新（不轮询）")
    add_watch_arguments(monitor)
    args = parser.parse_args(argv)
    try:
        if args.command == "context":
            run_context(args)
        elif args.command in ("watch", "monitor"):
            run_watch(args)
        elif args.command == "memories":
            run_memories(args)
        else:
            run_memories(argparse.Namespace(
                db=args.db, conversation=None, sort="weight", decay=False,
                full=False, json=False, id=None))
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
