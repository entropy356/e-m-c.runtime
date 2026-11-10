"""OpenAI-compatible chat provider.

Only transport concerns live here: base_url/model configuration, the API key
from an environment variable, timeout, proxy, retries with backoff, error
mapping and response parsing. No memory protocol, no provider fallback, no
silent retry of a whole turn.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Sequence

from ..contracts import AssistantReply, ProviderError

DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_ATTEMPTS = 3
_BACKOFF = (2.0, 5.0)
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class ProviderHTTPError(ProviderError):
    def __init__(self, code: int, hint: str = ""):
        message = f"provider HTTP {code}; check endpoint, model, quota and credentials"
        if hint:
            message += f" ({hint})"
        super().__init__(message)
        self.code = code


def _build_opener(proxy: str | None):
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _retry_delay(error: urllib.error.HTTPError, attempt: int) -> float:
    """Honour Retry-After when the server sends one, else back off gently."""
    try:
        wait = float(error.headers.get("Retry-After") or 0)
    except (TypeError, ValueError):
        wait = 0.0
    return max(0.25, min(wait or _BACKOFF[min(attempt, 1)], 30.0))


def http_post_json(
    url: str,
    payload: dict,
    headers: Mapping[str, str],
    timeout: float,
    proxy: str | None,
    max_attempts: int,
) -> dict:
    """POST JSON with bounded retry on transient errors. Test seam for fakes."""
    body = json.dumps(payload, ensure_ascii=False).encode()
    opener = _build_opener(proxy)
    last_attempt = max_attempts - 1
    for attempt in range(max_attempts):
        request = urllib.request.Request(url, body, {**headers, "Content-Type": "application/json"})
        try:
            with opener.open(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in _RETRYABLE_STATUS or attempt >= last_attempt:
                raise ProviderHTTPError(error.code) from None
            time.sleep(_retry_delay(error, attempt))
        except OSError:
            # URLError and TimeoutError are both OSError subclasses.
            if attempt >= last_attempt:
                raise ProviderError("provider connection failed or timed out") from None
            time.sleep(_BACKOFF[min(attempt, 1)])
        except (json.JSONDecodeError, UnicodeError):
            raise ProviderError("provider returned invalid JSON") from None
    raise ProviderError("provider request failed")


class OpenAIChatProvider:
    """Talks to any OpenAI-compatible ``/chat/completions`` endpoint."""

    kind = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        options: Mapping[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        proxy: str | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        transport: Callable[..., dict] = http_post_json,
    ):
        if not base_url.startswith("https://"):
            raise ProviderError("provider base_url must use HTTPS")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.options = dict(options or {})
        self.timeout = float(timeout)
        self.proxy = proxy
        self.max_attempts = int(max_attempts)
        self._transport = transport

    def generate(self, messages: Sequence[dict], options: Mapping[str, Any]) -> AssistantReply:
        merged = {**self.options, **dict(options)}
        payload: dict[str, Any] = {"model": self.model, "messages": list(messages)}
        for key in ("temperature", "top_p", "max_tokens", "stop"):
            if key in merged and merged[key] is not None:
                payload[key] = merged[key]
        result = self._transport(
            self.base_url + "/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
            self.timeout,
            self.proxy,
            self.max_attempts,
        )
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("invalid chat completion response") from None
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("provider returned an empty reply; the turn was aborted")
        return AssistantReply(content=content.strip())


def make_openai_provider(params: Mapping[str, Any]) -> OpenAIChatProvider:
    base_url = params.get("base_url")
    model = params.get("model")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ProviderError("openai-compatible provider requires base_url")
    if not isinstance(model, str) or not model.strip():
        raise ProviderError("openai-compatible provider requires model")
    api_key_env = params.get("api_key_env", DEFAULT_API_KEY_ENV)
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise ProviderError("api_key_env must be an environment variable name")
    api_key = os.environ.get(api_key_env.strip(), "").strip()
    if not api_key:
        raise ProviderError(
            f"API key not found; set the {api_key_env.strip()} environment variable"
        )
    try:
        timeout = float(params.get("timeout", DEFAULT_TIMEOUT))
        max_attempts = int(params.get("max_attempts", DEFAULT_MAX_ATTEMPTS))
    except (TypeError, ValueError):
        raise ProviderError("provider timeout and max_attempts must be numbers") from None
    if timeout <= 0 or max_attempts < 1:
        raise ProviderError("provider timeout must be positive and max_attempts at least 1")
    proxy = params.get("proxy") or None
    options = params.get("options", {})
    if not isinstance(options, dict):
        raise ProviderError("provider options must be a JSON object")
    return OpenAIChatProvider(
        base_url=base_url.strip(),
        model=model.strip(),
        api_key=api_key,
        options=options,
        timeout=timeout,
        proxy=proxy,
        max_attempts=max_attempts,
    )
