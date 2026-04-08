import unittest

from models import AppConfig, LLMResponse
from pipelines.debate import RedTeamBlueTeamPipeline
from pipelines.deep import ChainOfVerificationPipeline


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)

    async def generate(self, *args, **kwargs):
        return self._responses.pop(0)


class PipelineFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_chain_of_verification_returns_initial_draft_if_verification_step_fails(self):
        pipeline = ChainOfVerificationPipeline()
        client = FakeClient([
            LLMResponse(text="Draft answer", ok=True, model="claude", provider="anthropic", tokens_used=10),
            LLMResponse(ok=False, error="Gemini error: 503", model="gemini-2.5-flash", provider="gemini"),
        ])

        result = await pipeline.execute(
            "What is the capital of Australia?",
            client,
            AppConfig(role_models={"generator": "claude", "reviewer": "gemini-2.5-flash", "synthesizer": "claude"}),
            model="claude",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.final_text, "Draft answer")

    async def test_red_team_blue_team_returns_blue_answer_if_revision_fails(self):
        pipeline = RedTeamBlueTeamPipeline()
        client = FakeClient([
            LLMResponse(text="Blue answer", ok=True, model="claude", provider="anthropic", tokens_used=10),
            LLMResponse(text="Here are the flaws", ok=True, model="gemini-2.5-flash", provider="gemini", tokens_used=10),
            LLMResponse(ok=False, error="Gemini error: 503", model="gemini-2.5-flash", provider="gemini"),
        ])

        result = await pipeline.execute(
            "Explain stacks vs queues",
            client,
            AppConfig(role_models={"generator": "claude", "critic": "gemini-2.5-flash", "reviewer": "gemini-2.5-flash"}),
            model="claude",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.final_text, "Blue answer")


if __name__ == "__main__":
    unittest.main()
