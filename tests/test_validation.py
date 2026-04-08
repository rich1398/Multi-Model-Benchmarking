import unittest
from unittest.mock import AsyncMock

from llm_client import LLMClient
from models import AppConfig, JudgeConfig, ProviderConfig


class ValidationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            models=(
                ProviderConfig(
                    name="llama3.2",
                    base_url="http://localhost:11434",
                    provider="ollama",
                ),
                ProviderConfig(
                    name="gpt-4o",
                    base_url="https://api.openai.com",
                    api_key="sk-test",
                    provider="openai",
                ),
                ProviderConfig(
                    name="claude-sonnet-4-20250514",
                    base_url="https://api.anthropic.com",
                    api_key="anthropic-test",
                    provider="anthropic",
                ),
            ),
            judge=JudgeConfig(backend="ollama", model="llama3.2"),
        )
        self.client = LLMClient(self.config)

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_preflight_rejects_embedding_generation_model(self):
        self.client.list_ollama_model_details = AsyncMock(return_value=[
            {
                "name": "llama3.2",
                "chat_capable": True,
                "reason": "",
            },
            {
                "name": "nomic-embed-text:latest",
                "chat_capable": False,
                "reason": "Embedding models cannot be used for chat benchmarks.",
            },
        ])

        result = await self.client.preflight_check(
            generation_model="nomic-embed-text:latest",
            role_models={},
            judge_model="llama3.2",
            judge_provider="ollama",
        )

        self.assertFalse(result["ok"])
        self.assertTrue(
            any("Embedding models cannot be used" in error for error in result["errors"])
        )

    async def test_preflight_rejects_missing_role_model(self):
        self.client.list_ollama_model_details = AsyncMock(return_value=[
            {"name": "llama3.2", "chat_capable": True, "reason": ""},
        ])

        result = await self.client.preflight_check(
            generation_model="llama3.2",
            role_models={"critic": "missing-model"},
            judge_model="llama3.2",
            judge_provider="ollama",
        )

        self.assertFalse(result["ok"])
        self.assertTrue(
            any('Model "missing-model" is not installed in Ollama.' in error for error in result["errors"])
        )

    async def test_ollama_alias_without_latest_tag_is_accepted(self):
        self.client.list_ollama_model_details = AsyncMock(return_value=[
            {"name": "llama3.2:latest", "chat_capable": True, "reason": ""},
        ])

        ok, reason = await self.client.validate_model(
            "llama3.2",
            provider="ollama",
            purpose="generation",
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "")

    async def test_openai_provider_rejects_ollama_style_model_name(self):
        ok, reason = await self.client.validate_model(
            "llama3.2",
            provider="openai",
            purpose="judge",
        )

        self.assertFalse(ok)
        self.assertIn("local/Ollama model", reason)

    async def test_anthropic_provider_rejects_non_claude_model_name(self):
        ok, reason = await self.client.validate_model(
            "gpt-4o",
            provider="anthropic",
            purpose="judge",
        )

        self.assertFalse(ok)
        self.assertIn("does not look like an Anthropic Claude model", reason)
