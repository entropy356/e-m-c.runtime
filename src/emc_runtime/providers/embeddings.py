"""Optional embedding capability, isolated from the chat path.

Nothing in the core or the counter example requires an embedding provider.
Future memory modules may construct these adapters themselves; a missing
embedding configuration must never break plain chat or module demos.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence

from ..contracts import ProviderError
from .openai_chat import http_post_json


def unit(vector: Sequence[float]) -> list[float]:
    """Normalize an embedding vector; rejects NaN/Inf and empty input."""
    values = list(vector)
    if not values or len(values) > 65536:
        raise ProviderError("embedding must be a nonempty vector of at most 65536 numbers")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ProviderError("embedding contains invalid numbers")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0:
        raise ProviderError("embedding has invalid norm")
    return [value / norm for value in values]


class OpenAIEmbeddingAdapter:
    """OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int | None = None,
        timeout: float = 60.0,
        proxy: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.proxy = proxy

    def embed(self, text: str) -> list[float]:
        payload: dict[str, Any] = {"model": self.model, "input": text}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        result = http_post_json(
            self.base_url + "/embeddings",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
            self.timeout,
            self.proxy,
            3,
        )
        try:
            return unit(result["data"][0]["embedding"])
        except (KeyError, IndexError, TypeError):
            raise ProviderError("invalid embedding response") from None


class GeminiEmbeddingAdapter:
    """Google Gemini ``models/{model}:embedContent`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int | None = None,
        timeout: float = 60.0,
        proxy: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.proxy = proxy

    def embed(self, text: str) -> list[float]:
        payload: dict[str, Any] = {"content": {"parts": [{"text": text}]}}
        if self.dimensions:
            payload["outputDimensionality"] = self.dimensions
        result = http_post_json(
            f"{self.base_url}/models/{self.model}:embedContent",
            payload,
            {"x-goog-api-key": self.api_key},
            self.timeout,
            self.proxy,
            3,
        )
        try:
            return unit(result["embedding"]["values"])
        except (KeyError, IndexError, TypeError):
            raise ProviderError("invalid embedding response") from None


def make_embedding_provider(params):
    """Resolve an explicitly named embedding capability only when used."""
    import os
    kind = params.get('kind')
    cls = {'openai-compatible': OpenAIEmbeddingAdapter, 'gemini': GeminiEmbeddingAdapter}.get(kind)
    if cls is None:
        raise ProviderError(f'unknown embedding provider kind {kind!r}')
    base_url, model = params.get('base_url'), params.get('model')
    if not isinstance(base_url, str) or not base_url.startswith('https://'):
        raise ProviderError('embedding base_url must use HTTPS')
    if not isinstance(model, str) or not model.strip():
        raise ProviderError('embedding model must be nonempty text')
    key_env = params.get('api_key_env', 'OPENAI_API_KEY' if kind == 'openai-compatible' else 'GEMINI_API_KEY')
    if not isinstance(key_env, str) or not key_env.strip():
        raise ProviderError('embedding api_key_env must name an environment variable')
    api_key = os.environ.get(key_env, '')
    if not api_key:
        raise ProviderError(f'set the {key_env} environment variable')
    return cls(base_url=base_url, model=model, api_key=api_key,
               dimensions=params.get('dimensions'), timeout=params.get('timeout', 60),
               proxy=params.get('proxy'))
