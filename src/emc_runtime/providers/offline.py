"""Deterministic offline chat provider.

A plumbing fixture, not a semantic model: it reports what it received so
tests can verify message order and content without any network access.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..contracts import AssistantReply


class OfflineChatProvider:
    """Echo-style deterministic replies for tests and smoke scenarios."""

    kind = "offline"

    def generate(self, messages: Sequence[dict], options: Mapping[str, Any]) -> AssistantReply:
        last = messages[-1]["content"] if messages else ""
        reply = f"[offline] messages={len(messages)} last={last}"
        return AssistantReply(content=reply)


def make_offline_provider(params: Mapping[str, Any]) -> OfflineChatProvider:
    return OfflineChatProvider()
