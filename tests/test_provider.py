"""Provider tests: offline determinism and OpenAI-compatible transport."""

from __future__ import annotations

import os
import urllib.error
import unittest
from unittest import mock

from emc_runtime.contracts import AssistantReply, ProviderError
from emc_runtime.providers import make_provider
from emc_runtime.providers import openai_chat
from emc_runtime.providers.offline import OfflineChatProvider


class OfflineProviderTest(unittest.TestCase):
    def test_deterministic(self):
        provider = OfflineChatProvider()
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]
        first = provider.generate(messages, {})
        second = provider.generate(messages, {})
        self.assertEqual(first, second)
        self.assertIsInstance(first, AssistantReply)
        self.assertIn("messages=2", first.content)


class OpenAIProviderTest(unittest.TestCase):
    PARAMS = {
        "kind": "openai-compatible",
        "base_url": "https://api.example.com/v1",
        "model": "test-model",
        "api_key_env": "EMC_TEST_KEY",
    }

    def _make(self, params=None):
        with mock.patch.dict(os.environ, {"EMC_TEST_KEY": "sk-test"}):
            return make_provider(params or self.PARAMS)

    def _make_with_transport(self, transport, params=None):
        provider = self._make(params)
        provider._transport = transport
        return provider

    def test_request_shape_and_success(self):
        captured = {}

        def fake_post(url, payload, headers, timeout, proxy, max_attempts):
            captured.update(
                url=url, payload=payload, headers=dict(headers),
                timeout=timeout, proxy=proxy, max_attempts=max_attempts,
            )
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": " 你好 "}}
                ]
            }

        provider = self._make_with_transport(fake_post)
        reply = provider.generate(
            [{"role": "user", "content": "hi"}], {"temperature": 0.2}
        )
        self.assertEqual(reply.content, "你好")
        self.assertEqual(captured["url"], "https://api.example.com/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "test-model")
        self.assertEqual(captured["payload"]["temperature"], 0.2)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer sk-test")
        self.assertIsNone(captured["proxy"])

    def test_empty_content_rejected(self):
        provider = self._make_with_transport(
            lambda *args, **kwargs: {"choices": [{"message": {"content": "   "}}]}
        )
        with self.assertRaises(ProviderError):
            provider.generate([{"role": "user", "content": "hi"}], {})

    def test_malformed_response_rejected(self):
        provider = self._make_with_transport(
            lambda *args, **kwargs: {"unexpected": True}
        )
        with self.assertRaises(ProviderError):
            provider.generate([{"role": "user", "content": "hi"}], {})

    def test_http_error_propagates_without_key(self):
        from emc_runtime.providers.openai_chat import ProviderHTTPError

        def fake_post(url, payload, headers, timeout, proxy, max_attempts):
            raise ProviderHTTPError(401, "invalid api key")

        provider = self._make_with_transport(fake_post)
        with self.assertRaises(ProviderHTTPError) as ctx:
            provider.generate([{"role": "user", "content": "hi"}], {})
        self.assertNotIn("sk-test", str(ctx.exception))

    def test_missing_api_key(self):
        env = {k: v for k, v in os.environ.items() if k != "EMC_TEST_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ProviderError) as ctx:
                make_provider(self.PARAMS)
        self.assertIn("EMC_TEST_KEY", str(ctx.exception))

    def test_https_enforced(self):
        params = dict(self.PARAMS, base_url="http://api.example.com/v1")
        with self.assertRaises(ProviderError):
            self._make(params)

    def test_transport_error_wrapped(self):
        def fake_post(url, payload, headers, timeout, proxy, max_attempts):
            raise urllib.error.URLError("connection refused")

        provider = self._make()
        with mock.patch.object(openai_chat, "http_post_json", side_effect=fake_post):
            with self.assertRaises(ProviderError):
                provider.generate([{"role": "user", "content": "hi"}], {})

    def test_retry_config_validated(self):
        params = dict(self.PARAMS, max_attempts=0)
        with self.assertRaises(ProviderError):
            self._make(params)


if __name__ == "__main__":
    unittest.main()
