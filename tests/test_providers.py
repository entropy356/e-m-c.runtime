import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from long_term_memory.model import DynamicsConfig
from long_term_memory.providers import (
    ANSWER_PROMPT,
    APIProvider,
    BEHAVIOR,
    OfflineProvider,
    PROMPTS_DIR,
    ProviderHTTPError,
    SYSTEM,
    SYSTEM_TOOL,
    TOOL_NAME,
    cognition_tool,
    prompt,
    unit,
)
from long_term_memory.runtime import Runtime


def tool_response(arguments):
    return {"choices": [{"message": {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": TOOL_NAME,
                         "arguments": json.dumps(arguments, ensure_ascii=False)},
        }],
    }}]}


def text_response(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {
            "AGENT_API_BASE": "",
            "AGENT_API_KEY": "",
            "AGENT_CHAT_BASE": "https://chat.invalid/v1",
            "AGENT_CHAT_KEY": "chat-secret",
            "AGENT_CHAT_MODEL": "chat-model",
            "AGENT_EMBED_PROTOCOL": "openai",
            "AGENT_EMBED_BASE": "https://embed.invalid/v1",
            "AGENT_EMBED_KEY": "embed-secret",
            "AGENT_EMBED_MODEL": "embed-model",
            "AGENT_EMBED_DIMENSIONS": "768",
            "AGENT_COGNITION_PROTOCOL": "json",
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.provider = APIProvider()

    def test_chat_receives_ordered_history_and_separate_memory_context(self):
        metadata = {"used_memory_ids": [4], "memories": []}
        replies = [text_response(json.dumps(metadata)), text_response("context-aware answer")]
        history = [
            {"role": "user", "content": "I prefer concise answers."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "What do I prefer?"},
        ]
        memories = [{"id": 4, "content": "User prefers concise answers.",
                     "weight": 0.7, "stability": 0.2, "vector": [1.0]}]
        with patch.object(self.provider, "_chat", side_effect=replies) as chat:
            result = self.provider.cognize(history, memories)
        self.assertEqual(result["answer"], "context-aware answer")
        self.assertEqual(result["used_memory_ids"], [4])
        payload = chat.call_args_list[0].args[0]
        messages = payload["messages"]
        self.assertEqual([item["role"] for item in messages[-3:]],
                         ["user", "assistant", "user"])
        self.assertEqual(messages[-3:], history)
        memory_context = json.loads(messages[1]["content"].split("\n", 1)[1])
        self.assertEqual(memory_context, {
            "long_term_memories": [{"id": 4, "content": "User prefers concise answers."}]
        })
        self.assertNotIn("chat-secret", self.provider.chat_profile)
        self.assertNotIn("embed-secret", self.provider.profile)

    def test_embedding_is_normalized_and_uses_separate_endpoint(self):
        with patch("long_term_memory.providers._request",
                   return_value={"data": [{"embedding": [3, 4]}]}) as request:
            self.assertEqual(self.provider.embed("query", "query"), [0.6, 0.8])
        self.assertEqual(request.call_args.args[0], "https://embed.invalid/v1/embeddings")
        self.assertEqual(request.call_args.args[1]["dimensions"], 768)
        self.assertEqual(request.call_args.args[2],
                         {"Authorization": "Bearer embed-secret"})

    def test_gemini_embedding_uses_retrieval_task_type(self):
        with patch.dict("os.environ", {
            "AGENT_EMBED_PROTOCOL": "gemini",
            "AGENT_EMBED_BASE": "https://gemini.invalid/v1beta",
            "AGENT_EMBED_MODEL": "gemini-embedding-001",
        }):
            provider = APIProvider()
        with patch("long_term_memory.providers._request",
                   return_value={"embedding": {"values": [0, 5]}}) as request:
            self.assertEqual(provider.embed("memory", "document"), [0.0, 1.0])
        url, payload, headers = request.call_args.args[:3]
        self.assertEqual(url, "https://gemini.invalid/v1beta/models/gemini-embedding-001:embedContent")
        self.assertEqual(payload["taskType"], "RETRIEVAL_DOCUMENT")
        self.assertEqual(payload["outputDimensionality"], 768)
        self.assertEqual(headers, {"x-goog-api-key": "embed-secret"})

    def test_invalid_embedding_vectors_are_rejected(self):
        for vector in ([], [0, 0], [float("nan")], [float("inf")], [True], ["x"]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                unit(vector)


class CognitionProtocolTests(unittest.TestCase):
    """Each protocol is chosen explicitly: two requests per cognition, no fallback."""

    def setUp(self):
        self.env = patch.dict("os.environ", {
            "AGENT_API_BASE": "",
            "AGENT_API_KEY": "",
            "AGENT_CHAT_BASE": "https://chat.invalid/v1",
            "AGENT_CHAT_KEY": "chat-secret",
            "AGENT_CHAT_MODEL": "chat-model",
            "AGENT_EMBED_PROTOCOL": "openai",
            "AGENT_EMBED_BASE": "https://embed.invalid/v1",
            "AGENT_EMBED_KEY": "embed-secret",
            "AGENT_EMBED_MODEL": "embed-model",
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.provider = APIProvider()
        self.history = [{"role": "user", "content": "我更喜欢简短的书面进度。"}]
        self.memories = [{"id": 7, "content": "用户偏好简短的书面进度。",
                          "weight": 0.9, "stability": 0.1, "vector": [1.0]}]

    def provider_with(self, protocol):
        with patch.dict("os.environ", {"AGENT_COGNITION_PROTOCOL": protocol}):
            return APIProvider()

    def test_default_protocol_is_tool(self):
        self.assertEqual(self.provider.cognition_protocol, "tool")

    def test_prompts_are_loaded_from_files(self):
        """Wording lives in prompts/*.md, so these assertions stay structural."""
        for name in ("context.md", "protocol-tool.md", "protocol-json.md", "answer.md"):
            self.assertTrue((PROMPTS_DIR / name).is_file(), name)
        self.assertEqual(SYSTEM, BEHAVIOR + "\n\n" + prompt("protocol-json.md"))
        self.assertEqual(SYSTEM_TOOL, BEHAVIOR + "\n\n" + prompt("protocol-tool.md"))
        self.assertTrue(SYSTEM.startswith(BEHAVIOR))
        self.assertTrue(SYSTEM_TOOL.startswith(BEHAVIOR))
        self.assertIn(TOOL_NAME, SYSTEM_TOOL)   # 模型必须知道调哪个工具
        self.assertNotIn(TOOL_NAME, SYSTEM)     # json/text 协议没有工具通道
        self.assertTrue(ANSWER_PROMPT.strip())

    def test_missing_prompt_file_reports_its_path(self):
        prompt.cache_clear()
        with patch("long_term_memory.providers.PROMPTS_DIR", Path("/nonexistent-prompts")):
            with self.assertRaises(ValueError) as caught:
                prompt("context.md")
        prompt.cache_clear()
        self.assertIn("nonexistent-prompts", str(caught.exception))

    def test_tool_schema_cites_only_provided_memory_ids(self):
        parameters = cognition_tool([4, 2, 4])["function"]["parameters"]
        self.assertEqual(cognition_tool([4])["function"]["name"], TOOL_NAME)
        self.assertEqual(parameters["properties"]["used_memory_ids"]["items"]["enum"], [2, 4])
        self.assertEqual(sorted(parameters["required"]), ["memories", "used_memory_ids"])
        self.assertNotIn("answer", parameters["properties"])
        self.assertNotIn("enum", cognition_tool([])["function"]["parameters"]
                         ["properties"]["used_memory_ids"]["items"])

    def test_tool_protocol_asks_for_memory_then_for_the_reply(self):
        metadata = {"used_memory_ids": [7], "memories": [{"content": "偏好简短进度。"}]}
        replies = [tool_response(metadata), text_response("记住了。")]
        with patch.object(self.provider, "_chat", side_effect=replies) as chat:
            result = self.provider.cognize(self.history, self.memories)

        self.assertEqual(result, {"answer": "记住了。", **metadata})
        self.assertEqual(chat.call_count, 2)

        first, second = (call.args[0] for call in chat.call_args_list)
        self.assertEqual(first["tool_choice"],
                         {"type": "function", "function": {"name": TOOL_NAME}})
        self.assertEqual(first["tools"][0]["function"]["parameters"]["properties"]
                         ["used_memory_ids"]["items"]["enum"], [7])
        self.assertEqual(first["messages"][0], {"role": "system", "content": SYSTEM_TOOL})
        self.assertEqual(first["messages"][-1], self.history[-1])
        self.assertNotIn("response_format", first)

        self.assertEqual(second["tool_choice"], "none")
        self.assertEqual(second["messages"][:len(first["messages"])], first["messages"])
        self.assertEqual(second["messages"][-2]["role"], "assistant")
        self.assertEqual(second["messages"][-2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(second["messages"][-1], {
            "role": "tool", "tool_call_id": "call_1",
            "content": json.dumps(metadata, ensure_ascii=False),
        })

    def test_reply_is_assistant_content_not_tool_arguments(self):
        stale = {"answer": "工具里的旧答案", "used_memory_ids": [], "memories": []}
        replies = [tool_response(stale), text_response("正文里的回答")]
        with patch.object(self.provider, "_chat", side_effect=replies):
            result = self.provider.cognize(self.history, [])
        self.assertEqual(result["answer"], "正文里的回答")
        self.assertEqual(result["used_memory_ids"], [])

    def test_missing_reply_content_is_rejected(self):
        replies = [tool_response({"used_memory_ids": [], "memories": []}),
                   {"choices": [{"message": {"role": "assistant", "content": "   "}}]}]
        with patch.object(self.provider, "_chat", side_effect=replies) as chat:
            with self.assertRaises(ValueError) as caught:
                self.provider.cognize(self.history, [])
        self.assertEqual(chat.call_count, 2)
        self.assertIn("assistant content", str(caught.exception))

    def test_tool_protocol_requires_a_tool_call(self):
        with patch.object(self.provider, "_chat",
                          return_value=text_response("我先说明一下……")) as chat:
            with self.assertRaises(ValueError) as caught:
                self.provider.cognize(self.history, self.memories)
        self.assertEqual(chat.call_count, 1)
        self.assertIn(TOOL_NAME, str(caught.exception))

    def test_tool_protocol_does_not_fall_back_when_gateway_rejects_tools(self):
        with patch.object(self.provider, "_chat", side_effect=ProviderHTTPError(400)) as chat:
            with self.assertRaises(ProviderHTTPError) as caught:
                self.provider.cognize(self.history, self.memories)
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("AGENT_COGNITION_PROTOCOL", str(caught.exception))

    def test_json_protocol_sends_response_format_without_tools(self):
        provider = self.provider_with("json")
        metadata = {"used_memory_ids": [], "memories": []}
        replies = [text_response(json.dumps(metadata)), text_response("好的。")]
        with patch.object(provider, "_chat", side_effect=replies) as chat:
            result = provider.cognize(self.history, self.memories)

        self.assertEqual(result, {"answer": "好的。", **metadata})
        self.assertEqual(chat.call_count, 2)
        first, second = (call.args[0] for call in chat.call_args_list)
        self.assertEqual(first["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", first)
        self.assertEqual(first["messages"][0], {"role": "system", "content": SYSTEM})
        self.assertNotIn("response_format", second)
        self.assertEqual(second["messages"][-1],
                         {"role": "user", "content": ANSWER_PROMPT})

    def test_text_protocol_sends_plain_request(self):
        provider = self.provider_with("text")
        metadata = {"used_memory_ids": [], "memories": []}
        replies = [text_response(json.dumps(metadata)), text_response("好的。")]
        with patch.object(provider, "_chat", side_effect=replies) as chat:
            result = provider.cognize(self.history, self.memories)

        self.assertEqual(result, {"answer": "好的。", **metadata})
        self.assertEqual(chat.call_count, 2)
        payload = chat.call_args_list[0].args[0]
        self.assertNotIn("tools", payload)
        self.assertNotIn("response_format", payload)

    def test_text_protocol_accepts_fenced_json(self):
        provider = self.provider_with("text")
        metadata = {"used_memory_ids": [], "memories": []}
        fenced = "```json\n" + json.dumps(metadata, ensure_ascii=False) + "\n```"
        replies = [text_response(fenced), text_response("好的。")]
        with patch.object(provider, "_chat", side_effect=replies):
            self.assertEqual(provider.cognize(self.history, self.memories),
                             {"answer": "好的。", **metadata})

    def test_json_protocol_rejects_non_json_answer_without_retrying(self):
        provider = self.provider_with("json")
        with patch.object(provider, "_chat",
                          return_value=text_response("只有一句话，没有 JSON。")) as chat:
            with self.assertRaises(ValueError):
                provider.cognize(self.history, self.memories)
        self.assertEqual(chat.call_count, 1)

    def test_invalid_protocol_is_rejected(self):
        with patch.dict("os.environ", {"AGENT_COGNITION_PROTOCOL": "auto"}):
            with self.assertRaises(ValueError) as caught:
                APIProvider()
        self.assertIn("AGENT_COGNITION_PROTOCOL", str(caught.exception))

    def test_unauthorized_error_carries_protocol_hint(self):
        with patch.object(self.provider, "_chat", side_effect=ProviderHTTPError(401)):
            with self.assertRaises(ProviderHTTPError) as caught:
                self.provider.cognize(self.history, self.memories)
        self.assertEqual(caught.exception.code, 401)
        self.assertIn("AGENT_COGNITION_PROTOCOL", str(caught.exception))

    def test_runtime_records_cognition_protocol_and_writes_memory(self):
        offline = OfflineProvider()
        metadata = {"used_memory_ids": [], "memories": [{"content": "偏好简短进度。"}]}
        replies = [tool_response(metadata), text_response("记住了。")]
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(Path(directory) / "agent.sqlite3", self.provider, DynamicsConfig())
            with patch.object(self.provider, "embed", side_effect=offline.embed), \
                 patch.object(self.provider, "_chat", side_effect=replies):
                trace = runtime.turn("我更喜欢简短的书面进度。")
                state = runtime.inspect()

        self.assertEqual(trace["cognition_protocol"], "tool")
        self.assertEqual(trace["cognition"]["answer"], "记住了。")
        self.assertEqual(trace["written_ids"], [1])
        self.assertEqual(state["cognition_protocol"], "tool")


if __name__ == "__main__":
    unittest.main()
