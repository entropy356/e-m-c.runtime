"""Command-line interface.

Every command takes ``--config`` and opens the database only through the
engine or read-only storage queries. Non-model commands (history, inspect)
never write state and never create module state rows.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .config import load_config
from .contracts import EMCError, Event
from .engine import build_engine
from .modules import scope_key_for
from .storage import Storage

CHAT_HELP = (
    "命令：/history [条数|all] 查看本会话历史 · /trigger [原因] 发起一次内部触发 "
    "· /help 显示这段说明 · /exit 或 /quit 退出（普通输入会发给模型）"
)


def _print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _write_json(path: str, value) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _add_conversation_argument(parser: argparse.ArgumentParser, default: str = "default") -> None:
    parser.add_argument("--conversation", default=default, help="会话 id（默认 %(default)s）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emc-runtime",
        description="轻量单进程模块化 Agent 实验运行环境",
    )
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    sub = parser.add_subparsers(dest="command", required=True)

    turn = sub.add_parser("turn", help="执行一轮：发送一条用户消息")
    turn.add_argument("content", help="用户消息文本")
    _add_conversation_argument(turn)

    chat = sub.add_parser("chat", help="交互式多轮对话")
    _add_conversation_argument(chat)

    trigger = sub.add_parser("trigger", help="执行一轮：内部触发（不写用户消息）")
    _add_conversation_argument(trigger)
    trigger.add_argument("--reason", default=None, help="触发原因，仅存入元数据")

    history = sub.add_parser("history", help="查看会话原始历史（只读）")
    _add_conversation_argument(history)

    inspect = sub.add_parser("inspect", help="查看模块状态与最近轮次（只读）")
    _add_conversation_argument(inspect)

    scenario = sub.add_parser("scenario", help="按剧本依次执行事件")
    scenario.add_argument("file", help="剧本 JSON 文件")
    scenario.add_argument("--output", default=None, help="把结果写入 JSON 文件")
    return parser


# --------------------------------------------------------------------- commands


def cmd_turn(args, engine) -> int:
    event = Event(kind="user_message", conversation_id=args.conversation, content=args.content)
    result = engine.run(event)
    if args.json:
        _print_json(result.to_json_dict())
    else:
        print(result.reply)
    return 0


def cmd_trigger(args, engine) -> int:
    metadata = {"reason": args.reason} if args.reason else {}
    event = Event(
        kind="internal_trigger", conversation_id=args.conversation, metadata=metadata
    )
    result = engine.run(event)
    if args.json:
        _print_json(result.to_json_dict())
    else:
        print(result.reply)
    return 0


def cmd_history(args, config) -> int:
    storage = Storage(config.database)
    messages = storage.history(args.conversation)
    if args.json:
        _print_json(
            [
                {
                    "seq": message.seq,
                    "role": message.role,
                    "content": message.content,
                    "turn_id": message.turn_id,
                }
                for message in messages
            ]
        )
    else:
        for message in messages:
            print(f"{message.seq:4d} {message.role:9s} {message.content}")
        if not messages:
            print(f"（会话 {args.conversation} 暂无历史）")
    return 0


def cmd_inspect(args, config, engine) -> int:
    storage = Storage(config.database)
    from .modules import ModuleRegistry  # loaded lazily to keep imports honest

    registry = ModuleRegistry(config.modules)
    modules_view = []
    for module in registry.modules:
        entry = storage.module_state(
            module.module_id, scope_key_for(module.scope, args.conversation)
        )
        if entry is None:
            view = None  # read-only: never create missing state
        else:
            view = module.inspect(entry.state)
        modules_view.append(
            {
                "module_id": module.module_id,
                "scope": module.scope,
                "state_version": module.state_version,
                "state": view,
            }
        )
    turns = storage.recent_turns(args.conversation, limit=10)
    if args.json:
        _print_json(
            {
                "conversation_id": args.conversation,
                "revision": storage.revision(),
                "modules": modules_view,
                "recent_turns": [
                    {
                        "turn_id": turn["turn_id"],
                        "event_kind": turn["event_kind"],
                        "revision": turn["revision"],
                        "created_at": turn["created_at"],
                    }
                    for turn in turns
                ],
            }
        )
    else:
        print(f"conversation {args.conversation} @ revision {storage.revision()}")
        for item in modules_view:
            print(f"module {item['module_id']} ({item['scope']}): {item['state']}")
        for turn in turns:
            print(
                f"turn {turn['turn_id'][:8]} {turn['event_kind']:16s} "
                f"revision={turn['revision']}"
            )
    return 0


def cmd_chat(args, engine, config) -> int:
    storage = Storage(config.database)
    print(f"emc-runtime chat · 会话 {args.conversation} · {CHAT_HELP}")
    while True:
        try:
            line = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        if line == "/help":
            print(CHAT_HELP)
            continue
        if line.startswith("/history"):
            parts = line.split(maxsplit=1)
            messages = storage.history(args.conversation)
            if len(parts) == 2 and parts[1] != "all":
                try:
                    messages = messages[-int(parts[1]) :]
                except ValueError:
                    print("用法：/history [条数|all]")
                    continue
            for message in messages:
                print(f"{message.seq:4d} {message.role:9s} {message.content}")
            continue
        if line.startswith("/trigger"):
            parts = line.split(maxsplit=1)
            reason = parts[1] if len(parts) == 2 else None
            result = engine.run(
                Event(
                    kind="internal_trigger",
                    conversation_id=args.conversation,
                    metadata={"reason": reason} if reason else {},
                )
            )
            print(f"agent> {result.reply}")
            continue
        result = engine.run(
            Event(kind="user_message", conversation_id=args.conversation, content=line)
        )
        print(f"agent> {result.reply}")
    return 0


def cmd_scenario(args, engine) -> int:
    path = Path(args.file)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EMCError(f"cannot read scenario {path}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        raise EMCError("scenario must be an object with an 'events' list")
    default_conversation = data.get("conversation_id", "default")
    results = []
    for index, item in enumerate(data["events"]):
        if not isinstance(item, dict):
            raise EMCError(f"scenario events[{index}] must be an object")
        event = Event(
            kind=item.get("kind", "user_message"),
            conversation_id=item.get("conversation_id", default_conversation),
            content=item.get("content"),
            metadata=item.get("metadata", {}),
        )
        results.append(engine.run(event).to_json_dict())
    payload = {"scenario": data.get("name", path.stem), "results": results}
    if args.output:
        _write_json(args.output, payload)
    _print_json(payload)
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        engine = build_engine(config)
        if args.command == "turn":
            return cmd_turn(args, engine)
        if args.command == "chat":
            return cmd_chat(args, engine, config)
        if args.command == "trigger":
            return cmd_trigger(args, engine)
        if args.command == "history":
            return cmd_history(args, config)
        if args.command == "inspect":
            return cmd_inspect(args, config, engine)
        if args.command == "scenario":
            return cmd_scenario(args, engine)
        parser.error(f"unknown command {args.command!r}")
    except (EMCError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
