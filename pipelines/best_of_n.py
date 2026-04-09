"""Best-of-N selection pipelines for Occursus Benchmark."""

from __future__ import annotations

import asyncio
import re

from models import AppConfig, LLMResponse, PipelineResult, PipelineSpec, StepTrace
from llm_client import LLMClient
from pipelines.base import BasePipeline


def _trace_from(phase: str, resp: LLMResponse) -> StepTrace:
    return StepTrace(
        phase=phase,
        model=resp.model,
        latency_ms=resp.latency_ms,
        tokens=resp.tokens_used,
        text_preview=resp.text[:200] if resp.text else resp.error,
    )


def _aggregate(steps: tuple[StepTrace, ...]) -> tuple[float, int, int]:
    total_latency = sum(s.latency_ms for s in steps)
    total_tokens = sum(s.tokens for s in steps)
    llm_calls = len(steps)
    return total_latency, total_tokens, llm_calls


def _parse_choice(text: str, upper_bound: int) -> int | None:
    match = re.search(r"\b([1-9])\b", text.strip())
    if match:
        num = int(match.group(1))
        if 1 <= num <= upper_bound:
            return num
    return None


def _fail_result(
    pipeline_id: str, error: str, steps: tuple[StepTrace, ...],
) -> PipelineResult:
    total_latency, total_tokens, llm_calls = _aggregate(steps)
    return PipelineResult(
        pipeline_id=pipeline_id,
        ok=False,
        error=error,
        steps=steps,
        total_latency_ms=total_latency,
        total_tokens=total_tokens,
        llm_calls=llm_calls,
    )


def _success_result(
    pipeline_id: str, text: str, steps: tuple[StepTrace, ...],
) -> PipelineResult:
    total_latency, total_tokens, llm_calls = _aggregate(steps)
    return PipelineResult(
        pipeline_id=pipeline_id,
        final_text=text,
        ok=True,
        steps=steps,
        total_latency_ms=total_latency,
        total_tokens=total_tokens,
        llm_calls=llm_calls,
    )


def _build_judge_prompt(responses: list[str], n: int) -> str:
    sections = []
    for i, text in enumerate(responses, 1):
        sections.append(f"--- Response {i} ---\n{text}")
    joined = "\n\n".join(sections)
    return (
        f"Below are {n} responses to the same task. "
        f"Which response is best? Reply with ONLY: "
        f"{', '.join(str(i) for i in range(1, n + 1))}\n\n"
        f"{joined}"
    )


class BestOf3Pipeline(BasePipeline):
    """Generate 3 responses at different temperatures, judge picks the best."""

    _TEMPS = (0.3, 0.5, 0.7)
    _PIPELINE_ID = "best_of_3"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._PIPELINE_ID,
            name="Best of 3",
            description=(
                "Generate 3 responses at temperatures 0.3, 0.5, 0.7 "
                "then a judge LLM picks the best one."
            ),
            tier=1,
            estimated_calls=4,
            tags=("selection", "best-of-n"),
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
        gen_model = self._role_model(config, model, "generator")
        judge_model = self._role_model(config, model, "judge", "reviewer", "generator")
        wrapped = self.wrap_task(prompt, cot=self._cot(config))

        # --- Phase 1: Generate 3 responses in parallel ---
        await self._notify(progress_callback, "Generating 3 candidate responses...")

        gen_tasks = tuple(
            client.generate(wrapped, model=gen_model, temperature=temp)
            for temp in self._TEMPS
        )
        gen_results: tuple[LLMResponse, ...] = await asyncio.gather(*gen_tasks)

        steps: list[StepTrace] = []
        responses: list[str] = []
        for i, resp in enumerate(gen_results):
            steps.append(_trace_from(f"generate_t{self._TEMPS[i]}", resp))
            if resp.ok:
                responses.append(resp.text)
            else:
                responses.append("")

        ok_responses = [r for r in responses if r]
        if not ok_responses:
            return _fail_result(
                self._PIPELINE_ID,
                "All 3 generation calls failed",
                tuple(steps),
            )

        if len(ok_responses) == 1:
            return _success_result(
                self._PIPELINE_ID, ok_responses[0], tuple(steps),
            )

        # --- Phase 2: Judge picks the best ---
        await self._notify(progress_callback, "Judging best response...")

        valid_texts = [r for r in responses if r]
        judge_prompt = _build_judge_prompt(valid_texts, len(valid_texts))
        judge_resp = await client.generate(
            judge_prompt, model=judge_model, temperature=0.1,
        )
        steps.append(_trace_from("judge", judge_resp))

        if judge_resp.ok:
            choice = _parse_choice(judge_resp.text, len(valid_texts))
            if choice is not None:
                chosen_text = valid_texts[choice - 1]
                return _success_result(
                    self._PIPELINE_ID, chosen_text, tuple(steps),
                )

        # Fallback: longest response
        await self._notify(
            progress_callback, "Judge failed, selecting longest response...",
        )
        longest = max(valid_texts, key=len)
        return _success_result(self._PIPELINE_ID, longest, tuple(steps))


class SampleAndVotePipeline(BasePipeline):
    """Generate 5 responses with varied temperatures, then vote for the best."""

    _TEMPS = (0.2, 0.4, 0.6, 0.8, 1.0)
    _PIPELINE_ID = "sample_and_vote"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._PIPELINE_ID,
            name="Sample & Vote",
            description=(
                "Generate 5 responses at temperatures 0.2-1.0 "
                "then vote for the most consistent and complete."
            ),
            tier=1,
            estimated_calls=6,
            tags=("selection", "voting"),
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
        gen_model = self._role_model(config, model, "generator")
        judge_model = self._role_model(config, model, "judge", "reviewer", "generator")
        wrapped = self.wrap_task(prompt, cot=self._cot(config))

        # --- Phase 1: Generate 5 responses in parallel ---
        await self._notify(progress_callback, "Generating 5 candidate responses...")

        gen_tasks = tuple(
            client.generate(wrapped, model=gen_model, temperature=temp)
            for temp in self._TEMPS
        )
        gen_results: tuple[LLMResponse, ...] = await asyncio.gather(*gen_tasks)

        steps: list[StepTrace] = []
        responses: list[str] = []
        for i, resp in enumerate(gen_results):
            steps.append(_trace_from(f"generate_t{self._TEMPS[i]}", resp))
            if resp.ok:
                responses.append(resp.text)
            else:
                responses.append("")

        valid_texts = [r for r in responses if r]
        if not valid_texts:
            return _fail_result(
                self._PIPELINE_ID,
                "All 5 generation calls failed",
                tuple(steps),
            )

        if len(valid_texts) == 1:
            return _success_result(
                self._PIPELINE_ID, valid_texts[0], tuple(steps),
            )

        # --- Phase 2: Vote for the best ---
        await self._notify(progress_callback, "Voting on best response...")

        vote_prompt = _build_vote_prompt(valid_texts)
        vote_resp = await client.generate(
            vote_prompt, model=judge_model, temperature=0.1,
        )
        steps.append(_trace_from("vote", vote_resp))

        if vote_resp.ok:
            choice = _parse_choice(vote_resp.text, len(valid_texts))
            if choice is not None:
                chosen_text = valid_texts[choice - 1]
                return _success_result(
                    self._PIPELINE_ID, chosen_text, tuple(steps),
                )

        # Fallback: ask LLM to pick the most consistent
        await self._notify(
            progress_callback,
            "Vote parse failed, asking for consistency pick...",
        )
        fallback_prompt = _build_consistency_fallback(valid_texts)
        fallback_resp = await client.generate(
            fallback_prompt, model=judge_model, temperature=0.1,
        )
        steps.append(_trace_from("vote_fallback", fallback_resp))

        if fallback_resp.ok:
            choice = _parse_choice(fallback_resp.text, len(valid_texts))
            if choice is not None:
                return _success_result(
                    self._PIPELINE_ID,
                    valid_texts[choice - 1],
                    tuple(steps),
                )

        # Ultimate fallback: longest
        longest = max(valid_texts, key=len)
        return _success_result(self._PIPELINE_ID, longest, tuple(steps))


def _build_vote_prompt(responses: list[str]) -> str:
    n = len(responses)
    sections = []
    for i, text in enumerate(responses, 1):
        sections.append(f"--- Response {i} ---\n{text}")
    joined = "\n\n".join(sections)
    choices = ", ".join(str(i) for i in range(1, n + 1))
    return (
        f"Below are {n} responses to the same task. "
        f"Which response is most consistent and complete? "
        f"Reply with ONLY: {choices}\n\n"
        f"{joined}"
    )


def _build_consistency_fallback(responses: list[str]) -> str:
    n = len(responses)
    sections = []
    for i, text in enumerate(responses, 1):
        sections.append(f"--- Response {i} ---\n{text}")
    joined = "\n\n".join(sections)
    choices = ", ".join(str(i) for i in range(1, n + 1))
    return (
        f"Below are {n} responses. Compare them carefully. "
        f"Which one is most factually consistent with the others "
        f"and gives the most complete answer? "
        f"Reply with ONLY the number: {choices}\n\n"
        f"{joined}"
    )
