"""Base pipeline interface for Occursus Benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models import AppConfig, PipelineResult, PipelineSpec
from llm_client import LLMClient


TASK_WRAPPER = (
    "Answer the following task directly and completely. "
    "Do not ask clarifying questions. Do not refuse. "
    "Give your best answer.\n\n"
    "TASK: {prompt}"
)

COT_PREFIX = (
    "Think step by step. Show your reasoning process, then give "
    "your final answer clearly marked with 'FINAL ANSWER:'.\n\n"
)


class BasePipeline(ABC):
    @abstractmethod
    def spec(self) -> PipelineSpec:
        ...

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        client: LLMClient,
        config: AppConfig,
        *,
        model: str | None = None,
        progress_callback=None,
    ) -> PipelineResult:
        ...

    def wrap_task(self, prompt: str, *, cot: bool = False) -> str:
        base = TASK_WRAPPER.format(prompt=prompt)
        return f"{COT_PREFIX}{base}" if cot else base

    def _cot(self, config: AppConfig) -> bool:
        return getattr(config, "cot_enabled", False)

    def _intermediate_tokens(self, config: AppConfig) -> int | None:
        """Token limit for intermediate steps (None = no limit)."""
        if not getattr(config, "token_budget_enabled", False):
            return None
        return int(getattr(config, "max_output_tokens", 4096) * 0.4)

    def _final_tokens(self, config: AppConfig) -> int | None:
        """Token limit for final synthesis step (None = no limit)."""
        if not getattr(config, "token_budget_enabled", False):
            return None
        return int(getattr(config, "max_output_tokens", 4096) * 0.6)

    def _role_model(
        self,
        config: AppConfig,
        fallback_model: str | None,
        role: str,
        *aliases: str,
    ) -> str:
        role_models = getattr(config, "role_models", {}) or {}
        for key in (role, *aliases):
            model = str(role_models.get(key, "")).strip()
            if model:
                return model
        return str(fallback_model or config.default_model)

    def _diverse_models(self, config: AppConfig, fallback: str | None) -> list[str]:
        """Return up to 3 distinct models for multi-agent diversity."""
        models = []
        seen = set()
        for role in ("generator", "generator_alt", "critic", "reviewer", "synthesizer"):
            m = self._role_model(config, fallback, role)
            if m and m not in seen:
                models.append(m)
                seen.add(m)
            if len(models) >= 3:
                break
        if not models:
            models = [str(fallback or config.default_model)]
        return models

    async def _notify(self, callback, message: str) -> None:
        if callback:
            await callback(message)
