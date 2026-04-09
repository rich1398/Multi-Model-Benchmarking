"""Debate-family pipelines for Occursus Benchmark."""

from __future__ import annotations

import asyncio

from models import AppConfig, LLMResponse, PipelineResult, PipelineSpec, StepTrace
from llm_client import LLMClient
from pipelines.base import BasePipeline


def _trace(phase: str, resp: LLMResponse) -> StepTrace:
    return StepTrace(
        phase=phase,
        model=resp.model,
        latency_ms=resp.latency_ms,
        tokens=resp.tokens_used,
        text_preview=resp.text[:200] if resp.text else resp.error,
    )


def _fail(
    pipeline_id: str,
    error: str,
    steps: tuple[StepTrace, ...],
) -> PipelineResult:
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
# 2-Way Debate
# ---------------------------------------------------------------------------


class Debate2WayPipeline(BasePipeline):
    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id="debate_2way",
            name="2-Way Debate",
            description=(
                "Two models take opposing positions, rebut each other, "
                "then an arbiter synthesises the strongest answer."
            ),
            tier=3,
            estimated_calls=5,
            tags=("debate", "adversarial"),
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
        pid = "debate_2way"
        gen_model = self._role_model(config, model, "generator")
        alt_model = self._role_model(config, model, "generator_alt", "generator")
        critic_model = self._role_model(config, model, "critic", "reviewer", "generator")
        arbiter_model = self._role_model(config, model, "arbiter", "synthesizer", "judge", "generator")
        task = self.wrap_task(prompt, cot=self._cot(config))
        steps: list[StepTrace] = []

        # --- Phase 1: parallel positions ----------------------------------
        await self._notify(progress_callback, "Generating two opposing positions...")

        prompt_a = task
        prompt_b = (
            "Provide an alternative perspective/answer to this task. "
            "If the task has a definitive answer, still try to explore "
            "different angles.\n\n"
            f"TASK: {prompt}"
        )

        resp_a, resp_b = await asyncio.gather(
            client.generate(prompt_a, model=gen_model),
            client.generate(prompt_b, model=alt_model),
        )

        steps.append(_trace("position_a", resp_a))
        steps.append(_trace("position_b", resp_b))

        if not resp_a.ok:
            return _fail(pid, f"Position A failed: {resp_a.error}", tuple(steps))
        if not resp_b.ok:
            return _fail(pid, f"Position B failed: {resp_b.error}", tuple(steps))

        # --- Phase 2: parallel rebuttals ----------------------------------
        await self._notify(progress_callback, "Models rebutting each other...")

        rebuttal_a_prompt = (
            "Here is another response to the same task:\n\n"
            f"{resp_b.text}\n\n"
            "Defend what's correct in your answer, concede any valid points "
            "from theirs, and strengthen your position.\n\n"
            f"Your original answer:\n{resp_a.text}"
        )
        rebuttal_b_prompt = (
            "Here is another response to the same task:\n\n"
            f"{resp_a.text}\n\n"
            "Defend what's correct in your answer, concede any valid points "
            "from theirs, and strengthen your position.\n\n"
            f"Your original answer:\n{resp_b.text}"
        )

        reb_a, reb_b = await asyncio.gather(
            client.generate(rebuttal_a_prompt, model=critic_model),
            client.generate(rebuttal_b_prompt, model=critic_model),
        )

        steps.append(_trace("rebuttal_a", reb_a))
        steps.append(_trace("rebuttal_b", reb_b))

        if not reb_a.ok:
            return _fail(pid, f"Rebuttal A failed: {reb_a.error}", tuple(steps))
        if not reb_b.ok:
            return _fail(pid, f"Rebuttal B failed: {reb_b.error}", tuple(steps))

        # --- Phase 3: arbiter synthesis -----------------------------------
        await self._notify(progress_callback, "Arbiter synthesising final answer...")

        arbiter_prompt = (
            "Two experts debated this task. Here are their final positions "
            "after rebuttal:\n\n"
            f"Position A:\n{reb_a.text}\n\n"
            f"Position B:\n{reb_b.text}\n\n"
            "Synthesize the strongest elements into one definitive answer."
        )

        arbiter = await client.generate(arbiter_prompt, model=arbiter_model)
        steps.append(_trace("arbiter", arbiter))

        if not arbiter.ok:
            return _fail(pid, f"Arbiter failed: {arbiter.error}", tuple(steps))

        return _ok(pid, arbiter.text, tuple(steps))


# ---------------------------------------------------------------------------
# Dissent Then Merge
# ---------------------------------------------------------------------------


class DissentThenMergePipeline(BasePipeline):
    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id="dissent_then_merge",
            name="Dissent Then Merge",
            description=(
                "Generate two responses, harshly critique each, "
                "then merge the best elements while filtering errors."
            ),
            tier=3,
            estimated_calls=5,
            tags=("debate", "critique"),
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
        pid = "dissent_then_merge"
        gen_model = self._role_model(config, model, "generator")
        alt_model = self._role_model(config, model, "generator_alt", "generator")
        critic_model = self._role_model(config, model, "critic", "reviewer", "generator")
        synth_model = self._role_model(config, model, "synthesizer", "generator")
        task = self.wrap_task(prompt, cot=self._cot(config))
        steps: list[StepTrace] = []

        # --- Phase 1: parallel generation ---------------------------------
        await self._notify(progress_callback, "Generating two initial responses...")

        resp_1, resp_2 = await asyncio.gather(
            client.generate(task, model=gen_model),
            client.generate(task, model=alt_model, temperature=0.9),
        )

        steps.append(_trace("generate_1", resp_1))
        steps.append(_trace("generate_2", resp_2))

        if not resp_1.ok:
            return _fail(pid, f"Generate 1 failed: {resp_1.error}", tuple(steps))
        if not resp_2.ok:
            return _fail(pid, f"Generate 2 failed: {resp_2.error}", tuple(steps))

        # --- Phase 2: parallel dissent ------------------------------------
        await self._notify(progress_callback, "Dissent models critiquing responses...")

        dissent_1_prompt = (
            "Find everything WRONG with this response. "
            "Be harsh and specific:\n\n"
            f"{resp_1.text}"
        )
        dissent_2_prompt = (
            "Find everything WRONG with this response. "
            "Be harsh and specific:\n\n"
            f"{resp_2.text}"
        )

        dis_1, dis_2 = await asyncio.gather(
            client.generate(dissent_1_prompt, model=critic_model),
            client.generate(dissent_2_prompt, model=critic_model),
        )

        steps.append(_trace("dissent_1", dis_1))
        steps.append(_trace("dissent_2", dis_2))

        if not dis_1.ok:
            return _fail(pid, f"Dissent 1 failed: {dis_1.error}", tuple(steps))
        if not dis_2.ok:
            return _fail(pid, f"Dissent 2 failed: {dis_2.error}", tuple(steps))

        # --- Phase 3: merge -----------------------------------------------
        await self._notify(progress_callback, "Merging best elements...")

        merge_prompt = (
            "Here are two responses and critiques of each. "
            "Use the dissent to filter out errors and merge the best "
            "elements.\n\n"
            f"Response 1:\n{resp_1.text}\n\n"
            f"Dissent 1:\n{dis_1.text}\n\n"
            f"Response 2:\n{resp_2.text}\n\n"
            f"Dissent 2:\n{dis_2.text}"
        )

        merged = await client.generate(merge_prompt, model=synth_model)
        steps.append(_trace("merge", merged))

        if not merged.ok:
            return _fail(pid, f"Merge failed: {merged.error}", tuple(steps))

        return _ok(pid, merged.text, tuple(steps))


# ---------------------------------------------------------------------------
# Red Team / Blue Team
# ---------------------------------------------------------------------------


class RedTeamBlueTeamPipeline(BasePipeline):
    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id="red_team_blue_team",
            name="Red Team / Blue Team",
            description=(
                "Blue team answers, red team attacks, "
                "blue team revises to address all valid criticisms."
            ),
            tier=3,
            estimated_calls=3,
            tags=("debate", "adversarial", "red-team"),
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
        pid = "red_team_blue_team"
        gen_model = self._role_model(config, model, "generator")
        critic_model = self._role_model(config, model, "critic", "reviewer", "generator")
        revise_model = self._role_model(config, model, "reviewer", "synthesizer", "generator")
        task = self.wrap_task(prompt, cot=self._cot(config))
        steps: list[StepTrace] = []

        # --- Phase 1: blue team generates ---------------------------------
        await self._notify(progress_callback, "Blue team generating answer...")

        blue = await client.generate(task, model=gen_model)
        steps.append(_trace("blue_generate", blue))

        if not blue.ok:
            return _fail(pid, f"Blue team failed: {blue.error}", tuple(steps))

        # --- Phase 2: red team attacks ------------------------------------
        await self._notify(progress_callback, "Red team attacking response...")

        red_prompt = (
            "You are a red team adversary. Find every flaw, error, "
            "weakness, and gap in this response. Be thorough and "
            "adversarial:\n\n"
            f"{blue.text}"
        )

        red = await client.generate(red_prompt, model=critic_model)
        steps.append(_trace("red_attack", red))

        if not red.ok:
            return _ok(pid, blue.text, tuple(steps))

        # --- Phase 3: blue team revises -----------------------------------
        await self._notify(progress_callback, "Blue team revising answer...")

        revise_prompt = (
            "Your original response was attacked by a red team. "
            "Here are the flaws they found:\n\n"
            f"{red.text}\n\n"
            "Revise your answer to address ALL valid criticisms "
            "while keeping what was correct.\n\n"
            f"Your original answer:\n{blue.text}"
        )

        revised = await client.generate(revise_prompt, model=revise_model)
        steps.append(_trace("blue_revise", revised))

        if not revised.ok:
            return _ok(pid, blue.text, tuple(steps))

        return _ok(pid, revised.text, tuple(steps))
