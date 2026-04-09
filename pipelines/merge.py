"""Multi-expert synthesis merge pipelines for Occursus Benchmark."""

from __future__ import annotations

import asyncio
import json
import re

from models import (
    AppConfig, LLMResponse, Persona, PipelineResult, PipelineSpec, StepTrace,
)
from llm_client import LLMClient
from pipelines.base import BasePipeline

_DEFAULT_PERSONAS: tuple[Persona, ...] = (
    Persona(
        name="Analytical", temperature=0.3,
        system_prompt=(
            "You are an analytical expert. Focus on logical structure, "
            "evidence-based reasoning, accuracy, and systematic analysis."
        ),
    ),
    Persona(
        name="Creative", temperature=0.8,
        system_prompt=(
            "You are a creative expert. Focus on novel perspectives, "
            "unexpected connections, vivid explanations, and engaging "
            "communication. Think outside the box."
        ),
    ),
    Persona(
        name="Critical", temperature=0.4,
        system_prompt=(
            "You are a critical evaluator. Focus on identifying flaws, "
            "edge cases, counter-arguments, and potential issues. "
            "Be thorough in your critique and suggest improvements."
        ),
    ),
)


def _get_personas(config: AppConfig) -> tuple[Persona, ...]:
    if config.personas and len(config.personas) >= 3:
        return config.personas[:3]
    return _DEFAULT_PERSONAS


def _trace(phase: str, resp: LLMResponse) -> StepTrace:
    return StepTrace(
        phase=phase, model=resp.model, latency_ms=resp.latency_ms,
        tokens=resp.tokens_used,
        text_preview=resp.text[:200] if resp.text else resp.error,
    )


def _build_result(
    pid: str, text: str, ok: bool, steps: tuple[StepTrace, ...],
    error: str = "",
) -> PipelineResult:
    return PipelineResult(
        pipeline_id=pid, final_text=text, ok=ok, error=error,
        steps=steps,
        total_latency_ms=sum(s.latency_ms for s in steps),
        total_tokens=sum(s.tokens for s in steps),
        llm_calls=len(steps),
    )


async def _gen_experts(
    wrapped: str, personas: tuple[Persona, ...],
    client: LLMClient, model: str,
) -> tuple[tuple[LLMResponse, ...], list[StepTrace]]:
    tasks = tuple(
        client.generate(
            wrapped, model=model,
            temperature=p.temperature, system_prompt=p.system_prompt,
        )
        for p in personas
    )
    results: tuple[LLMResponse, ...] = await asyncio.gather(*tasks)
    steps = [
        _trace(f"expert_{p.name.lower()}", r)
        for p, r in zip(personas, results)
    ]
    return results, steps


def _extract_valid(
    personas: tuple[Persona, ...], expert_results: tuple[LLMResponse, ...],
) -> tuple[list[str], tuple[Persona, ...], list[int]]:
    """Return (all_texts, valid_personas, valid_indices)."""
    texts = [r.text if r.ok else "" for r in expert_results]
    indices = [i for i, t in enumerate(texts) if t]
    valid_p = tuple(personas[i] for i in indices)
    return texts, valid_p, indices


def _synth_prompt(personas: tuple[Persona, ...], texts: list[str]) -> str:
    sections = "\n\n".join(
        f"Expert ({p.name}):\n{t}" for p, t in zip(personas, texts)
    )
    return (
        "Here are 3 expert perspectives on the task. "
        "Synthesize them into one comprehensive, accurate answer "
        f"that combines the best insights from each.\n\n{sections}"
    )


class MergeFullPipeline(BasePipeline):
    """Three expert personas generate, then a synthesizer merges."""

    _ID = "merge_full"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID, name="Full Synthesis Merge",
            description="3 expert personas generate responses, then a synthesizer merges them.",
            tier=2, estimated_calls=4,
            tags=("synthesis", "merge", "multi-persona"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        gen_model = self._role_model(config, model, "generator")
        synth_model = self._role_model(config, model, "synthesizer", "generator")
        wrapped = self.wrap_task(prompt, cot=self._cot(config))
        personas = _get_personas(config)

        await self._notify(progress_callback, "Generating 3 expert perspectives...")
        expert_results, steps = await _gen_experts(wrapped, personas, client, gen_model)
        all_texts, valid_personas, valid_idx = _extract_valid(personas, expert_results)
        valid_texts = [all_texts[i] for i in valid_idx]

        if not valid_texts:
            return _build_result(self._ID, "", False, tuple(steps), "All 3 expert calls failed")

        await self._notify(progress_callback, "Synthesizing expert responses...")
        synth_resp = await client.generate(
            _synth_prompt(valid_personas, valid_texts), model=synth_model, temperature=0.3,
        )
        steps.append(_trace("synthesize", synth_resp))

        chosen = synth_resp.text if synth_resp.ok else max(valid_texts, key=len)
        return _build_result(self._ID, chosen, True, tuple(steps))


class CritiqueThenMergePipeline(BasePipeline):
    """Experts generate, each gets critiqued, then all merge with critiques."""

    _ID = "critique_then_merge"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID, name="Critique Then Merge",
            description="3 expert responses, each critiqued, then synthesized with critique context.",
            tier=2, estimated_calls=7,
            tags=("synthesis", "merge", "critique"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        gen_model = self._role_model(config, model, "generator")
        critique_model = self._role_model(config, model, "critic", "reviewer", "generator")
        synth_model = self._role_model(config, model, "synthesizer", "generator")
        wrapped = self.wrap_task(prompt, cot=self._cot(config))
        personas = _get_personas(config)

        # Phase 1: Expert generation
        await self._notify(progress_callback, "Generating 3 expert perspectives...")
        expert_results, steps = await _gen_experts(wrapped, personas, client, gen_model)
        all_texts, _, valid_idx = _extract_valid(personas, expert_results)

        if not valid_idx:
            return _build_result(self._ID, "", False, tuple(steps), "All 3 expert calls failed")

        # Phase 2: Critique each in parallel
        await self._notify(progress_callback, "Generating critiques for each expert...")
        critique_tasks = tuple(
            client.generate(
                f"Evaluate this response for strengths and weaknesses:\n\n{all_texts[i]}",
                model=critique_model, temperature=0.2,
            )
            for i in valid_idx
        )
        critique_results: tuple[LLMResponse, ...] = await asyncio.gather(*critique_tasks)

        critiques: dict[int, str] = {}
        for idx, resp in zip(valid_idx, critique_results):
            steps.append(_trace(f"critique_{personas[idx].name.lower()}", resp))
            critiques[idx] = resp.text if resp.ok else "(critique unavailable)"

        # Phase 3: Synthesis with critiques
        await self._notify(progress_callback, "Synthesizing with critique context...")
        synth_sections = []
        for i in valid_idx:
            name = personas[i].name
            synth_sections.append(
                f"Expert ({name}):\n{all_texts[i]}\n\n"
                f"Critique of {name}:\n{critiques.get(i, '(unavailable)')}"
            )
        synth_prompt = (
            "Here are expert perspectives on a task, each with a critique. "
            "Synthesize them into one comprehensive, accurate answer. "
            "Lean on the strengths identified in critiques and address "
            f"the weaknesses.\n\n" + "\n\n---\n\n".join(synth_sections)
        )
        synth_resp = await client.generate(synth_prompt, model=synth_model, temperature=0.3)
        steps.append(_trace("synthesize", synth_resp))

        valid_texts = [all_texts[i] for i in valid_idx]
        chosen = synth_resp.text if synth_resp.ok else max(valid_texts, key=len)
        return _build_result(self._ID, chosen, True, tuple(steps))


class RankedMergePipeline(BasePipeline):
    """Experts generate, a judge ranks them, synthesizer prioritizes top."""

    _ID = "ranked_merge"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID, name="Ranked Merge",
            description="3 expert responses ranked by a judge, then synthesized with top priority.",
            tier=2, estimated_calls=5,
            tags=("synthesis", "merge", "ranking"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        gen_model = self._role_model(config, model, "generator")
        judge_model = self._role_model(config, model, "judge", "reviewer", "generator")
        synth_model = self._role_model(config, model, "synthesizer", "generator")
        wrapped = self.wrap_task(prompt, cot=self._cot(config))
        personas = _get_personas(config)

        # Phase 1: Expert generation
        await self._notify(progress_callback, "Generating 3 expert perspectives...")
        expert_results, steps = await _gen_experts(wrapped, personas, client, gen_model)
        all_texts, valid_personas, valid_idx = _extract_valid(personas, expert_results)
        valid_texts = [all_texts[i] for i in valid_idx]

        if not valid_texts:
            return _build_result(self._ID, "", False, tuple(steps), "All 3 expert calls failed")

        n = len(valid_texts)

        # Phase 2: Rank responses
        await self._notify(progress_callback, "Ranking expert responses...")
        rank_sections = "\n\n".join(
            f"--- Response {i} ({p.name}) ---\n{t}"
            for i, (p, t) in enumerate(zip(valid_personas, valid_texts), 1)
        )
        rank_prompt = (
            f"Rank these {n} responses from best to worst. "
            f"Reply with JSON: "
            f'{{"ranking": [best_num, mid_num, worst_num], "reasoning": "..."}}\n\n'
            f"Response numbers are: {', '.join(str(i) for i in range(1, n + 1))}\n\n"
            f"{rank_sections}"
        )
        rank_resp = await client.generate(rank_prompt, model=judge_model, temperature=0.1)
        steps.append(_trace("rank", rank_resp))
        ranking = _parse_ranking(rank_resp, n)

        # Phase 3: Synthesize with ranking context
        await self._notify(progress_callback, "Synthesizing with ranking priority...")
        synth_prompt = _ranked_synth_prompt(valid_personas, valid_texts, ranking)
        synth_resp = await client.generate(synth_prompt, model=synth_model, temperature=0.3)
        steps.append(_trace("synthesize", synth_resp))

        if synth_resp.ok:
            return _build_result(self._ID, synth_resp.text, True, tuple(steps))

        # Fallback: use top-ranked or longest
        fallback = (
            valid_texts[ranking[0] - 1] if ranking
            else max(valid_texts, key=len)
        )
        return _build_result(self._ID, fallback, True, tuple(steps))


def _parse_ranking(resp: LLMResponse, n: int) -> list[int] | None:
    if not resp.ok:
        return None
    text = resp.text.strip()

    json_match = re.search(r"\{[^}]*\"ranking\"\s*:\s*\[[^\]]*\][^}]*\}", text)
    if json_match:
        try:
            ranking = json.loads(json_match.group()).get("ranking", [])
            if (
                isinstance(ranking, list) and len(ranking) == n
                and all(isinstance(x, int) and 1 <= x <= n for x in ranking)
                and len(set(ranking)) == n
            ):
                return ranking
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    numbers = [int(m) for m in re.findall(r"\b(\d+)\b", text)]
    valid = [x for x in numbers if 1 <= x <= n]
    if len(valid) >= n and len(set(valid[:n])) == n:
        return valid[:n]
    return None


def _ranked_synth_prompt(
    personas: tuple[Persona, ...], texts: list[str],
    ranking: list[int] | None,
) -> str:
    if ranking:
        best_name = personas[ranking[0] - 1].name
        sections = "\n\n---\n\n".join(
            f"Rank #{pos}{' (BEST)' if pos == 1 else ''} - "
            f"{personas[num - 1].name}:\n{texts[num - 1]}"
            for pos, num in enumerate(ranking, 1)
        )
        return (
            f"Here are expert responses ranked from best to worst. "
            f"The top-ranked is from the {best_name} expert. "
            f"Synthesize into one comprehensive answer. "
            f"Draw primarily from the top-ranked while incorporating "
            f"unique valid points from lower-ranked ones.\n\n{sections}"
        )

    sections = "\n\n---\n\n".join(
        f"Expert ({p.name}):\n{t}" for p, t in zip(personas, texts)
    )
    return (
        "Here are expert perspectives on the task. "
        "Synthesize them into one comprehensive, accurate answer "
        f"that combines the best insights from each.\n\n{sections}"
    )
