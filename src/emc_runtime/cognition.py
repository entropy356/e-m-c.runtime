"""Replaceable cognition strategies.

The first version is ``direct``: one chat request, ordinary assistant text.
Multi-step protocols (tools, structured output, memory extraction) belong to
future strategies, not to the core.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import (
    AssistantReply,
    ChatProvider,
    CognitionStrategy,
    ConfigError,
    ProviderError,
)


class DirectStrategy:
    """A single provider call; the reply is plain assistant content."""

    def run(
        self,
        messages: Sequence[dict],
        provider: ChatProvider,
        options: Mapping[str, Any],
    ) -> AssistantReply:
        reply = provider.generate(list(messages), dict(options))
        if not isinstance(reply, AssistantReply):
            raise ProviderError("provider must return an AssistantReply object")
        if not isinstance(reply.content, str) or not reply.content.strip():
            raise ProviderError("provider returned an empty reply; the turn was aborted")
        return AssistantReply(content=reply.content.strip())


def make_cognition(kind: str, _params: Mapping[str, Any]) -> CognitionStrategy:
    if kind == "direct":
        return DirectStrategy()
    raise ConfigError(f"unknown cognition kind {kind!r}")
