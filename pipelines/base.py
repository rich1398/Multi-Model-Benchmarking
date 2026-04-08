"""Base pipeline interface for Occursus-Claude."""

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

    def wrap_task(self, prompt: str) -> str:
        return TASK_WRAPPER.format(prompt=prompt)

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

    async def _notify(self, callback, message: str) -> None:
        if callback:
            await callback(message)
