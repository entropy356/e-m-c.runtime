"""Command-line interface for Demo 3."""

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys

from .model import DynamicsConfig
from .providers import APIProvider, OfflineProvider
from .runtime import Runtime
from . import events, viewer


def _read_json(path):
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def _write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


CHAT_HELP = (
    "命令：/history [条数|all] 查看本会话历史 · /help 显示这段说明 · "
    "/exit 或 /quit 退出（普通输入会发给模型）"
)


def _chat_history(database, conversation, argument, default_limit=10):
    """Read the conversation back from SQLite; no model call, nothing is written."""
    argument = argument.strip().lower()
    if argument in ("all", "0"):
        limit = None
    elif argument:
        try:
            limit = max(1, int(argument))
        except ValueError:
            return "用法：/history [条数|all]"
    else:
        limit = default_limit
    return viewer.render_history(viewer.load_context(database, conversation), limit=limit)


def main():
    parser = argparse.ArgumentParser(
        description="Emotion-free long-term conversational memory agent (Python 3.10+)"
    )
    parser.add_argument("--mode", choices=("offline", "api"), default="offline")
    parser.add_argument("--protocol", choices=("tool", "json", "text"),
                        help="API 认知协议，覆盖 AGENT_COGNITION_PROTOCOL（默认 tool）")
    parser.add_argument("--db", default="data/demo3.sqlite3")
    parser.add_argument("--conversation", default="default")
    parser.add_argument("--dynamics", choices=("weight-only", "two-state"), default="two-state")
    parser.add_argument("--decay-rate", type=float, default=0.12)
    parser.add_argument("--stability-protection", type=float, default=0.85)
    parser.add_argument("--recovery-rate", type=float, default=0.65)
    parser.add_argument("--stability-gain", type=float, default=0.12)
    parser.add_argument("--top-k", type=int, default=5)
    sub = parser.add_subparsers(dest="command", required=True)

    turn = sub.add_parser("turn", help="Send one user message")
    turn.add_argument("text")

    chat = sub.add_parser("chat", help="Start an interactive conversation")
    chat.add_argument("--events", action="store_true",
                      help="开一个本地事件端口，向 watch 推送每轮提交（监控模式）")
    sub.add_parser("history", help="Show the ordered conversation history")

    retrieve = sub.add_parser("retrieve", help="Inspect long-term memory retrieval")
    retrieve.add_argument("text")

    run = sub.add_parser("run", help="Run a scripted multi-turn conversation")
    run.add_argument("scenario")
    run.add_argument("--output", help="Optional JSON result path")

    sub.add_parser("state", help="Show memories, conversations and recent traces")

    memories = sub.add_parser("memories", help="View long-term memories (read-only, no API keys needed)")
    viewer.add_memory_arguments(memories)

    context = sub.add_parser("context", help="View one conversation's context and per-round memory activity")
    viewer.add_context_arguments(context)

    watch = sub.add_parser("watch", aliases=["monitor"],
                           help="Monitor mode: refresh on pushed events (no polling)")
    viewer.add_watch_arguments(watch)

    args = parser.parse_args()
    try:
        if args.command == "memories":
            viewer.run_memories(args)
            return
        if args.command == "context":
            viewer.run_context(args)
            return
        if args.command in ("watch", "monitor"):
            viewer.run_watch(args)
            return
        if args.protocol:
            os.environ["AGENT_COGNITION_PROTOCOL"] = args.protocol
        provider = APIProvider() if args.mode == "api" else OfflineProvider()
        config = DynamicsConfig(
            mode=args.dynamics,
            decay_rate=args.decay_rate,
            stability_protection=args.stability_protection,
            recovery_rate=args.recovery_rate,
            stability_gain=args.stability_gain,
            top_k=args.top_k,
        )
        runtime = Runtime(args.db, provider, config)

        if args.command == "turn":
            result = runtime.turn(args.text, args.conversation)
        elif args.command == "chat":
            hub = None
            if args.events:
                hub = events.EventHub(args.db)
                hub.write_port_file()
                runtime.notifier = hub.publish
                print(f"[events] 监听 {hub.endpoint} · 端口文件 {hub.file}")
                print(f"[events] 另开一个窗口: python -m long_term_memory --db {args.db} "
                      f"--conversation {args.conversation} watch")
            try:
                print("Chat started. /history 看历史，/help 看命令，/exit 退出。")
                while True:
                    try:
                        text = input("you> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if not text:
                        continue
                    if text.startswith("/"):
                        command, _, argument = text.partition(" ")
                        if command in ("/exit", "/quit"):
                            break
                        if command == "/history":
                            print(_chat_history(args.db, args.conversation, argument))
                            continue
                        if command == "/help":
                            print(CHAT_HELP)
                            continue
                        print(f"未知命令 {command}；{CHAT_HELP}")
                        continue
                    result = runtime.turn(text, args.conversation)
                    print("agent> " + result["answer"])
            finally:
                if hub is not None:
                    hub.close()
            return
        elif args.command == "history":
            result = runtime.history(args.conversation)
        elif args.command == "retrieve":
            result = runtime.retrieve(args.text)
        elif args.command == "run":
            scenario = _read_json(args.scenario)
            if not isinstance(scenario, dict) or not isinstance(scenario.get("turns"), list):
                raise ValueError("Scenario must be an object containing a turns list")
            conversation_id = scenario.get("conversation_id", args.conversation)
            traces = []
            for item in scenario["turns"]:
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict):
                    text = item.get("input")
                    conversation_id = item.get("conversation_id", conversation_id)
                else:
                    raise ValueError("Each scenario turn must be text or an object with input")
                traces.append(runtime.turn(text, conversation_id))
            result = {
                "scenario": scenario.get("name", Path(args.scenario).stem),
                "mode": provider.mode,
                "provider": {
                    "chat": getattr(provider, "chat_profile", None),
                    "embedding": provider.profile,
                },
                "config": config.to_dict(),
                "traces": traces,
                "final": runtime.inspect(conversation_id),
            }
            if args.output:
                _write_json(args.output, result)
        else:
            result = runtime.inspect(args.conversation)

        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
