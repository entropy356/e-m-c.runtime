"""Provider construction from configuration."""

from __future__ import annotations

from typing import Mapping

from ..contracts import ChatProvider, ProviderError
from .offline import make_offline_provider
from .openai_chat import make_openai_provider


def make_provider(params: Mapping) -> ChatProvider:
    kind = params.get("kind")
    if kind == "offline":
        return make_offline_provider(params)
    if kind == "openai-compatible":
        return make_openai_provider(params)
    raise ProviderError(f"unknown provider kind {kind!r}")
