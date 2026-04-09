"""Deep reasoning pipelines: CoV, Iterative Refinement, Mixture of Agents."""

from __future__ import annotations

import asyncio

from models import AppConfig, PipelineResult, PipelineSpec, StepTrace
from llm_client import LLMClient
from pipelines.base import BasePipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trace(phase: str, resp) -> StepTrace:
    return StepTrace(
        phase=phase,
        model=resp.model,
        latency_ms=resp.latency_ms,
        tokens=resp.tokens_used,
        text_preview=resp.text[:200] if resp.text else resp.error,
    )


def _fail(pipeline_id: str, steps: tuple[StepTrace, ...], error: str) -> PipelineResult:
    return PipelineResult(
        pipeline_id=pipeline_id,
        ok=False,
        error=error,
        steps=steps,
        total_latency_ms=sum(s.latency_ms for s in steps),
        total_tokens=sum(s.tokens for s in steps),
        llm_calls=len(steps),
    )


def _ok(
    pipeline_id: str,
    final_text: str,
    steps: tuple[StepTrace, ...],
) -> PipelineResult:
    return PipelineResult(
        pipeline_id=pipeline_id,
        final_text=final_text,
        ok=True,
        steps=steps,
        total_latency_ms=sum(s.latency_ms for s in steps),
        total_tokens=sum(s.tokens for s in steps),
        llm_calls=len(steps),
    )


# ---------------------------------------------------------------------------
# 1. Chain of Verification
# ---------------------------------------------------------------------------

class ChainOfVerificationPipeline(BasePipeline):
    """Draft -> generate verification questions -> answer independently -> revise."""

    _ID = "chain_of_verification"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID,
            name="Chain of Verification",
            description=(
                "Generates an answer, creates verification questions, "
                "answers them independently, then revises."
            ),
            tier=4,
            estimated_calls=4,
            tags=("deep", "verification"),
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
        review_model = self._role_model(config, model, "reviewer", "critic", "generator")
        revise_model = self._role_model(config, model, "synthesizer", "reviewer", "generator")
        steps: list[StepTrace] = []

        # Step 1 -- initial draft
        await self._notify(progress_callback, "Generating initial answer...")
        draft = await client.generate(self.wrap_task(prompt, cot=self._cot(config)), model=gen_model)
        steps.append(_trace("draft", draft))
        if not draft.ok:
            return _fail(self._ID, tuple(steps), draft.error)

        # Step 2 -- generate verification questions
        await self._notify(progress_callback, "Generating verification questions...")
        gen_q_prompt = (
            f"Given your answer above, generate 3-5 specific verification "
            f"questions that would help check if your answer is correct and "
            f"complete. Questions should be independently answerable.\n\n"
            f"Answer:\n{draft.text}"
        )
        gen_q = await client.generate(gen_q_prompt, model=review_model)
        steps.append(_trace("gen_questions", gen_q))
        if not gen_q.ok:
            return _ok(self._ID, draft.text, tuple(steps))

        # Step 3 -- answer questions independently (fresh context)
        await self._notify(progress_callback, "Answering verification questions independently...")
        answer_q_prompt = (
            f"Answer these verification questions about the following task. "
            f"Answer each one independently.\n\n"
            f"Task: {prompt}\n\n"
            f"Questions:\n{gen_q.text}"
        )
        answer_q = await client.generate(answer_q_prompt, model=review_model)
        steps.append(_trace("answer_questions", answer_q))
        if not answer_q.ok:
            return _ok(self._ID, draft.text, tuple(steps))

        # Step 4 -- revise based on verification
        await self._notify(progress_callback, "Revising answer based on verification...")
        revise_prompt = (
            f"Here is your original answer, and the results of independent "
            f"verification. Revise your answer to fix any inconsistencies "
            f"found.\n\n"
            f"Original answer:\n{draft.text}\n\n"
            f"Verification Q&A:\n{answer_q.text}\n\n"
            f"Revised answer:"
        )
        revised = await client.generate(revise_prompt, model=revise_model)
        steps.append(_trace("revise", revised))
        if not revised.ok:
            return _ok(self._ID, draft.text, tuple(steps))

        return _ok(self._ID, revised.text, tuple(steps))


# ---------------------------------------------------------------------------
# 2. Iterative Refinement
# ---------------------------------------------------------------------------

class IterativeRefinementPipeline(BasePipeline):
    """Generate -> (critique + revise) x 2 cycles with early stopping."""

    _ID = "iterative_refinement"
    _MAX_CYCLES = 2
    _NO_ISSUES_MARKERS = (
        "no significant issues",
        "no major issues",
        "no issues found",
        "looks good",
        "well-written",
        "no changes needed",
    )

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID,
            name="Iterative Refinement",
            description=(
                "Generates an answer then refines it through up to 2 "
                "critique-and-revise cycles with early stopping."
            ),
            tier=4,
            estimated_calls=5,
            tags=("deep", "refinement"),
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
        critique_model = self._role_model(config, model, "critic", "reviewer", "generator")
        revise_model = self._role_model(config, model, "reviewer", "synthesizer", "generator")
        steps: list[StepTrace] = []

        # Step 1 -- initial generation
        await self._notify(progress_callback, "Generating initial answer...")
        resp = await client.generate(self.wrap_task(prompt, cot=self._cot(config)), model=gen_model)
        steps.append(_trace("generate", resp))
        if not resp.ok:
            return _fail(self._ID, tuple(steps), resp.error)

        current_answer = resp.text

        # Critique-revise cycles
        for cycle in range(1, self._MAX_CYCLES + 1):
            # Critique
            await self._notify(
                progress_callback, f"Cycle {cycle}: critiquing answer..."
            )
            critique_prompt = (
                f"Identify specific weaknesses, errors, or missing elements "
                f"in this response:\n\n{current_answer}"
            )
            critique = await client.generate(critique_prompt, model=critique_model)
            steps.append(_trace(f"critique_{cycle}", critique))
            if not critique.ok:
                break

            # Early stop if critique finds no significant issues
            critique_lower = critique.text.lower()
            if any(marker in critique_lower for marker in self._NO_ISSUES_MARKERS):
                await self._notify(
                    progress_callback,
                    f"Cycle {cycle}: no significant issues found, stopping early.",
                )
                break

            # Revise
            await self._notify(
                progress_callback, f"Cycle {cycle}: revising answer..."
            )
            revise_prompt = (
                f"Revise this response based on the following critique:\n\n"
                f"Original:\n{current_answer}\n\n"
                f"Critique:\n{critique.text}"
            )
            revised = await client.generate(revise_prompt, model=revise_model)
            steps.append(_trace(f"revise_{cycle}", revised))
            if not revised.ok:
                break

            current_answer = revised.text

        return _ok(self._ID, current_answer, tuple(steps))


# ---------------------------------------------------------------------------
# 3. Mixture of Agents
# ---------------------------------------------------------------------------

class MixtureOfAgentsPipeline(BasePipeline):
    """3-layer pipeline: 3 parallel -> 3 parallel refiners -> 1 synthesizer."""

    _ID = "mixture_of_agents"
    _L1_TEMPS = (0.3, 0.5, 0.7)

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID,
            name="Mixture of Agents",
            description=(
                "Three layers of LLM calls: 3 independent generators, "
                "3 refiners that see all outputs, and 1 final synthesizer."
            ),
            tier=4,
            estimated_calls=7,
            tags=("deep", "mixture"),
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
        diverse = self._diverse_models(config, model)
        synth_model = self._role_model(config, model, "synthesizer", "generator")
        steps: list[StepTrace] = []
        wrapped = self.wrap_task(prompt, cot=self._cot(config))

        # Layer 1 -- 3 independent generators in parallel (one per model)
        await self._notify(progress_callback, "Layer 1: generating 3 responses (multi-model)...")
        l1_coros = tuple(
            client.generate(wrapped, model=diverse[i % len(diverse)], temperature=temp)
            for i, temp in enumerate(self._L1_TEMPS)
        )
        l1_results = await asyncio.gather(*l1_coros)

        l1_texts: list[str] = []
        for idx, resp in enumerate(l1_results, start=1):
            steps.append(_trace(f"layer1_gen_{idx}", resp))
            if not resp.ok:
                return _fail(self._ID, tuple(steps), resp.error)
            l1_texts.append(resp.text)

        # Layer 2 -- 3 refiners see all L1 outputs, run in parallel
        await self._notify(progress_callback, "Layer 2: refining with all Layer 1 outputs...")
        combined_l1 = _format_numbered_responses(l1_texts)
        l2_prompt = (
            f"Here are 3 different responses to the same task. Generate an "
            f"improved response that builds on the best elements of all "
            f"three.\n\n{combined_l1}"
        )
        l2_coros = tuple(
            client.generate(l2_prompt, model=diverse[i % len(diverse)], temperature=temp)
            for i, temp in enumerate(self._L1_TEMPS)
        )
        l2_results = await asyncio.gather(*l2_coros)

        l2_texts: list[str] = []
        for idx, resp in enumerate(l2_results, start=1):
            steps.append(_trace(f"layer2_refine_{idx}", resp))
            if resp.ok:
                l2_texts.append(resp.text)
        if not l2_texts:
            l2_texts = list(l1_texts)

        # Layer 3 -- final synthesizer
        await self._notify(progress_callback, "Layer 3: synthesizing final answer...")
        combined_l2 = _format_numbered_responses(l2_texts)
        l3_prompt = (
            f"Here are 3 refined responses to the same task. Produce the "
            f"definitive, final answer that represents the best synthesis "
            f"of all three.\n\n{combined_l2}"
        )
        final = await client.generate(l3_prompt, model=synth_model)
        steps.append(_trace("layer3_synthesize", final))
        if not final.ok:
            fallback = max(l2_texts or l1_texts, key=len)
            return _ok(self._ID, fallback, tuple(steps))

        return _ok(self._ID, final.text, tuple(steps))


def _format_numbered_responses(texts: list[str]) -> str:
    parts = [f"Response {i}:\n{text}" for i, text in enumerate(texts, start=1)]
    return "\n\n".join(parts)
