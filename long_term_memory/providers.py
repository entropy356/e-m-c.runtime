"""Chat and embedding adapters for Demo 3."""

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=None)
def prompt(name):
    """Prompt text lives in prompts/*.md; wording changes need no code edit."""
    path = PROMPTS_DIR / name
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(f"Prompt file missing or unreadable: {path}") from error
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def unit(vector):
    if not isinstance(vector, list) or not vector or len(vector) > 65536:
        raise ValueError("Embedding must be a nonempty numeric vector")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
        raise ValueError("Embedding contains invalid numbers")
    norm = math.sqrt(sum(x * x for x in vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding has invalid norm")
    return [x / norm for x in vector]


class OfflineProvider:
    """Deterministic plumbing fixture, not a semantic model."""

    mode = "offline"
    profile = "offline:sha256-character-bigrams-256:v1"
    chat_profile = "offline:echo-cognition-v1"

    def embed(self, text, task=None):
        normalized = "".join(text.lower().split())
        tokens = [normalized[i:i + 2] for i in range(max(1, len(normalized) - 1))]
        vector = [0.0] * 256
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(digest[:2], "big") % 256] += 1 if digest[2] & 1 else -1
        return unit(vector)

    def cognize(self, messages, memories):
        current = messages[-1]["content"] if messages else ""
        if memories:
            answer = "离线夹具检索到记忆，但不会进行真实语义推理：" + memories[0]["content"]
        else:
            answer = "离线夹具已收到输入：" + current
        return {"answer": answer, "used_memory_ids": [], "memories": []}


BEHAVIOR = prompt("context.md")

# 文本协议提示词：function calling 不可用时使用。
TEXT_FORMAT = prompt("protocol-json.md")
SYSTEM = BEHAVIOR + "\n\n" + TEXT_FORMAT

# 工具协议提示词：结构交给 submit_cognition 的 JSON Schema，提示词只描述上下文与字段的含义。
SYSTEM_TOOL = BEHAVIOR + "\n\n" + prompt("protocol-tool.md")

TOOL_NAME = "submit_cognition"

# 第二阶段（json / text 协议没有 tool 结果可回填，用一条 user 消息衔接）。
ANSWER_PROMPT = prompt("answer.md")

# 认知协议必须显式配置，不做自动降级。每次认知固定两个请求：
#   1) 记忆操作：按协议拿 used_memory_ids / memories
#   2) 回答：正文即用户看到的回复
#   tool  第一阶段调用 submit_cognition（JSON Schema 约束），第二阶段 tool_choice=none（默认）
#   json  第一阶段文本 JSON + response_format: json_object
#   text  第一阶段纯文本 JSON，不发送 response_format
COGNITION_PROTOCOLS = ("tool", "json", "text")


def cognition_tool(ids):
    """Forced tool for the first request: memory operations only, the reply is content."""
    used = {
        "type": "array",
        "description": "本轮回答依赖的长期记忆 ID，取值范围是当前提供的 ID；仅仅看到记忆不算使用。",
        "items": {"type": "integer"},
    }
    if ids:
        used["items"]["enum"] = sorted({int(item) for item in ids})
    else:
        used["description"] += "当前没有提供任何长期记忆，本字段为空数组。"
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "提交本轮的记忆操作：回答依赖的长期记忆 ID，以及值得长期保存的新记忆。",
            "parameters": {
                "type": "object",
                "properties": {
                    "used_memory_ids": used,
                    "memories": {
                        "type": "array",
                        "maxItems": 3,
                        "description": "最多三条值得长期保存的新记忆；没有时为空数组。",
                        "items": {
                            "type": "object",
                            "properties": {"content": {"type": "string", "description": "一条记忆，尽量一句话说清。"}},
                            "required": ["content"],
                        },
                    },
                },
                "required": ["used_memory_ids", "memories"],
            },
        },
    }


class ProviderHTTPError(ValueError):
    def __init__(self, code, hint=""):
        message = f"Provider HTTP {code}; check endpoint, model, quota and credentials"
        if hint:
            message += f" ({hint})"
        super().__init__(message)
        self.code = code


def _env(*names, default=""):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return default


def _request(url, payload, headers, proxy=None, timeout=60, attempts=3):
    body = json.dumps(payload, ensure_ascii=False).encode()
    handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy})] if proxy else [urllib.request.ProxyHandler({})]
    opener = urllib.request.build_opener(*handlers)
    backoff = (2, 5, 10)
    for attempt in range(attempts):
        request = urllib.request.Request(url, body, {**headers, "Content-Type": "application/json"})
        try:
            with opener.open(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                try:
                    wait = float(error.headers.get("Retry-After") or 0)
                except (TypeError, ValueError):
                    wait = 0
                time.sleep(max(1.0, min(wait or backoff[attempt], 30.0)))
                continue
            raise ProviderHTTPError(error.code) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < attempts - 1:
                time.sleep(backoff[attempt])
                continue
            raise ValueError("Provider connection failed or timed out") from None
        except (json.JSONDecodeError, UnicodeError):
            raise ValueError("Provider returned invalid JSON") from None
    raise ValueError("Provider request failed")


def _parse_object(content):
    if not isinstance(content, str):
        raise ValueError("Cognition returned no text")
    text = content.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = text[len(fence):].strip()
        if text.startswith("json"):
            text = text[4:].strip()
        if text.endswith(fence):
            text = text[:-len(fence)].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("Cognition must return a JSON object") from None


def _message(result):
    try:
        message = result["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("Invalid chat completion response") from None
    if not isinstance(message, dict):
        raise ValueError("Invalid chat completion response")
    return message


def _tool_call(result):
    """The forced tool call, or None when the model answered without calling it."""
    message = _message(result)
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return None
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or function.get("name") != TOOL_NAME:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            parsed = arguments
        elif isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        else:
            continue
        if not isinstance(parsed, dict):
            continue
        return {"id": call.get("id") or "call_0", "call": call, "arguments": parsed,
                "content": message.get("content")}
    return None


def _from_text(result):
    """JSON object embedded in the message text; None when there is nothing usable."""
    content = _message(result).get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        parsed = _parse_object(content)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _metadata(parsed):
    """Memory operations only; an `answer` key, if the model still writes one, is ignored."""
    if not isinstance(parsed, dict):
        raise ValueError("Cognition must return a JSON object")
    missing = [key for key in ("used_memory_ids", "memories") if key not in parsed]
    if missing:
        raise ValueError("Cognition must report " + " and ".join(missing))
    return {"used_memory_ids": parsed["used_memory_ids"], "memories": parsed["memories"]}


class OpenAIEmbedding:
    def __init__(self, base, key, model, dimensions=None, proxy=None, timeout=60):
        self.base, self.key, self.model = base, key, model
        self.dimensions, self.proxy, self.timeout = dimensions, proxy, timeout

    def embed(self, text, task=None):
        payload = {"model": self.model, "input": text}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        result = _request(self.base + "/embeddings", payload,
                          {"Authorization": "Bearer " + self.key}, self.proxy, self.timeout)
        try:
            return unit(result["data"][0]["embedding"])
        except (KeyError, IndexError, TypeError):
            raise ValueError("Invalid embedding response") from None


class GeminiEmbedding:
    TASKS = {"query": "RETRIEVAL_QUERY", "document": "RETRIEVAL_DOCUMENT"}

    def __init__(self, base, key, model, dimensions=None, proxy=None, timeout=60):
        self.base, self.key, self.model = base.rstrip("/"), key, model
        self.dimensions, self.proxy, self.timeout = dimensions, proxy, timeout

    def embed(self, text, task=None):
        payload = {"content": {"parts": [{"text": text}]}}
        if task in self.TASKS:
            payload["taskType"] = self.TASKS[task]
        if self.dimensions:
            payload["outputDimensionality"] = self.dimensions
        result = _request(f"{self.base}/models/{self.model}:embedContent", payload,
                          {"x-goog-api-key": self.key}, self.proxy, self.timeout)
        try:
            return unit(result["embedding"]["values"])
        except (KeyError, IndexError, TypeError):
            raise ValueError("Invalid embedding response") from None


class APIProvider:
    """OpenAI-compatible chat with independently configured OpenAI/Gemini embeddings."""

    mode = "api"

    def __init__(self):
        chat_base = _env("AGENT_CHAT_BASE", "AGENT_API_BASE").rstrip("/")
        chat_key = _env("AGENT_CHAT_KEY", "AGENT_API_KEY")
        chat_model = _env("AGENT_CHAT_MODEL", "AGENT_MODEL")
        protocol = _env("AGENT_EMBED_PROTOCOL", default="openai").lower()
        embed_base = _env("AGENT_EMBED_BASE", "AGENT_CHAT_BASE", "AGENT_API_BASE").rstrip("/")
        embed_key = _env("AGENT_EMBED_KEY", "AGENT_CHAT_KEY", "AGENT_API_KEY")
        embed_model = _env("AGENT_EMBED_MODEL", "AGENT_EMBEDDING_MODEL")
        if not all((chat_base, chat_key, chat_model, embed_base, embed_key, embed_model)):
            raise ValueError("API mode requires chat and embedding base, key, and model settings")
        if not chat_base.startswith("https://") or not embed_base.startswith("https://"):
            raise ValueError("API endpoints must use HTTPS")
        dimensions = _env("AGENT_EMBED_DIMENSIONS")
        dimensions = int(dimensions) if dimensions else None
        if dimensions is not None and dimensions < 1:
            raise ValueError("AGENT_EMBED_DIMENSIONS must be positive")
        chat_timeout = float(_env("AGENT_CHAT_TIMEOUT", default="120"))
        embed_timeout = float(_env("AGENT_EMBED_TIMEOUT", default="60"))
        if not math.isfinite(chat_timeout) or chat_timeout <= 0 or not math.isfinite(embed_timeout) or embed_timeout <= 0:
            raise ValueError("Provider timeouts must be positive finite numbers")
        self.chat_base, self.chat_key, self.chat_model = chat_base, chat_key, chat_model
        self.chat_profile = f"openai-compatible:{chat_model}"
        self.chat_proxy = _env("AGENT_CHAT_PROXY") or None
        self.chat_timeout = chat_timeout
        self.cognition_protocol = _env("AGENT_COGNITION_PROTOCOL", default="tool").lower()
        if self.cognition_protocol not in COGNITION_PROTOCOLS:
            raise ValueError("AGENT_COGNITION_PROTOCOL must be one of: " + ", ".join(COGNITION_PROTOCOLS))
        embed_proxy = _env("AGENT_EMBED_PROXY") or None
        if protocol == "gemini":
            self.embedding = GeminiEmbedding(embed_base, embed_key, embed_model, dimensions, embed_proxy, embed_timeout)
        elif protocol == "openai":
            self.embedding = OpenAIEmbedding(embed_base, embed_key, embed_model, dimensions, embed_proxy, embed_timeout)
        else:
            raise ValueError("AGENT_EMBED_PROTOCOL must be openai or gemini")
        self.profile = f"api:{protocol}:{embed_model}:" + (str(dimensions) if dimensions else "native")

    def embed(self, text, task=None):
        return self.embedding.embed(text, task)

    def _chat(self, payload):
        return _request(self.chat_base + "/chat/completions", payload,
                        {"Authorization": "Bearer " + self.chat_key},
                        self.chat_proxy, self.chat_timeout)

    def _protocol_error(self, error):
        return ProviderHTTPError(
            error.code,
            f"cognition protocol is {self.cognition_protocol}; "
            "switch AGENT_COGNITION_PROTOCOL if this gateway rejects it",
        )

    def cognize(self, messages, memories):
        """Two requests per turn: memory operations first, then the reply as content."""
        base = [
            {"role": "system",
             "content": SYSTEM_TOOL if self.cognition_protocol == "tool" else SYSTEM},
            {
                "role": "system",
                "content": "当前可用的长期记忆数据（JSON）：\n" + json.dumps(
                    {"long_term_memories": [
                        {"id": item["id"], "content": item["content"]} for item in memories
                    ]}, ensure_ascii=False),
            },
            *messages,
        ]
        metadata, followup = self._cognize(base, memories)
        return {"answer": self._answer(base + followup, memories), **metadata}

    def _cognize(self, base, memories):
        """First request: memory operations, plus the messages the reply request replays."""
        if self.cognition_protocol == "tool":
            payload = {
                "model": self.chat_model,
                "temperature": 0,
                "messages": base,
                "tools": [cognition_tool([item["id"] for item in memories])],
                "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
            }
            try:
                result = self._chat(payload)
            except ProviderHTTPError as error:
                raise self._protocol_error(error) from None
            call = _tool_call(result)
            if call is None:
                raise ValueError(
                    f"Cognition must call the {TOOL_NAME} tool; "
                    "switch AGENT_COGNITION_PROTOCOL if this model or gateway cannot call tools"
                )
            followup = [
                {"role": "assistant", "content": call["content"] or None,
                 "tool_calls": [call["call"]]},
                {"role": "tool", "tool_call_id": call["id"],
                 "content": json.dumps(call["arguments"], ensure_ascii=False)},
            ]
            return _metadata(call["arguments"]), followup

        payload = {"model": self.chat_model, "temperature": 0, "messages": base}
        if self.cognition_protocol == "json":
            payload["response_format"] = {"type": "json_object"}
        try:
            result = self._chat(payload)
        except ProviderHTTPError as error:
            raise self._protocol_error(error) from None
        parsed = _from_text(result)
        if parsed is None:
            raise ValueError(
                "Cognition must return a JSON object; "
                f"current protocol is {self.cognition_protocol}"
            )
        followup = [
            {"role": "assistant", "content": _message(result).get("content") or ""},
            {"role": "user", "content": ANSWER_PROMPT},
        ]
        return _metadata(parsed), followup

    def _answer(self, messages, memories):
        """Second request: the reply is ordinary assistant content, never a tool argument."""
        payload = {"model": self.chat_model, "temperature": 0, "messages": messages}
        if self.cognition_protocol == "tool":
            payload["tools"] = [cognition_tool([item["id"] for item in memories])]
            payload["tool_choice"] = "none"
        try:
            result = self._chat(payload)
        except ProviderHTTPError as error:
            raise self._protocol_error(error) from None
        content = _message(result).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "The reply must be assistant content; "
                f"current protocol is {self.cognition_protocol}"
            )
        return content.strip()
