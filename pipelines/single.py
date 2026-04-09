"""Baseline single-model pipeline."""

from __future__ import annotations

from models import AppConfig, PipelineResult, PipelineSpec, StepTrace
from llm_client import LLMClient
from pipelines.base import BasePipeline


class SinglePipeline(BasePipeline):
    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id="single",
            name="Single Model",
            description="Single model, single call. The control group baseline.",
            tier=1,
            estimated_calls=1,
            tags=("baseline",),
        )

    async def execute(
        self,
        prompt: str,
        client: LLMClient,
        config: AppConfig,
        *,
        model: str | None = None,
        progress_callback=None,
    ) -> PipelineResult:
        model = self._role_model(config, model, "generator")
        await self._notify(progress_callback, "Generating response...")

        resp = await client.generate(self.wrap_task(prompt, cot=self._cot(config)), model=model)

        step = StepTrace(
            phase="generate",
            model=resp.model,
            latency_ms=resp.latency_ms,
            tokens=resp.tokens_used,
            text_preview=resp.text[:200] if resp.text else resp.error,
        )

        if not resp.ok:
            return PipelineResult(
                pipeline_id="single",
                ok=False,
                error=resp.error,
                steps=(step,),
                total_latency_ms=resp.latency_ms,
                llm_calls=1,
            )

        return PipelineResult(
            pipeline_id="single",
            final_text=resp.text,
            ok=True,
            steps=(step,),
            total_latency_ms=resp.latency_ms,
            total_tokens=resp.tokens_used,
            llm_calls=1,
        )
