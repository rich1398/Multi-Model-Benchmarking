import unittest
from unittest.mock import AsyncMock, patch

import httpx

from llm_client import LLMClient
from models import AppConfig, ProviderConfig


class GeminiResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = LLMClient(
            AppConfig(
                models=(
                    ProviderConfig(
                        name="gemini-2.5-flash",
                        base_url="https://generativelanguage.googleapis.com",
                        api_key="gemini-test-key",
                        provider="gemini",
                    ),
                ),
                default_model="gemini-2.5-flash",
                default_provider="gemini",
                max_concurrent=3,
            )
        )

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_gemini_retries_transient_503_then_succeeds(self):
        req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent")
        responses = [
            httpx.Response(503, headers={"Retry-After": "0"}, request=req),
            httpx.Response(
                200,
                request=req,
                json={
                    "candidates": [{"content": {"parts": [{"text": "hello from gemini"}]}}],
                    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4},
                },
            ),
        ]
        self.client._http.post = AsyncMock(side_effect=responses)

        with patch("llm_client.asyncio.sleep", new=AsyncMock()) as mocked_sleep:
            result = await self.client.generate("hello", model="gemini-2.5-flash")

        self.assertTrue(result.ok)
        self.assertEqual(result.text, "hello from gemini")
        self.assertEqual(self.client._http.post.await_count, 2)
        self.assertGreaterEqual(mocked_sleep.await_count, 1)


if __name__ == "__main__":
    unittest.main()
