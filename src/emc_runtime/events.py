"""Observation events published after durable commits.

These are *output* notifications about what already happened; they are a
different type from the input events that trigger runs. A broken observer
must never damage or undo a committed turn: every subscriber call is
isolated, and ``publish`` never raises.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable

logger = logging.getLogger("emc_runtime.events")

Subscriber = Callable[[dict], None]


class EventBus:
    """Best-effort, synchronous, failure-isolated event bus."""

    def __init__(self, subscribers: list[Subscriber] | None = None):
        self._subscribers: list[Subscriber] = list(subscribers or [])

    def subscribe(self, subscriber: Subscriber) -> None:
        if not callable(subscriber):
            raise TypeError("subscriber must be callable")
        self._subscribers.append(subscriber)

    def publish(self, event: dict) -> None:
        for subscriber in list(self._subscribers):
            try:
                subscriber(event)
            except Exception as error:  # noqa: BLE001 - isolation is the contract
                logger.warning(
                    "observer %r failed after a committed turn: %s: %s",
                    getattr(subscriber, "__name__", subscriber),
                    type(error).__name__,
                    error,
                )


def logging_subscriber(stream=sys.stderr) -> Subscriber:
    """Print one JSON line per event; handy for ``watch``-style debugging."""

    def subscriber(event: dict[str, Any]) -> None:
        import json

        try:
            print("[event] " + json.dumps(event, ensure_ascii=False), file=stream)
        except Exception:  # noqa: BLE001 - never raise from observation
            pass

    return subscriber
