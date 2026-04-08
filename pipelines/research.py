"""Research-backed pipeline architectures for Occursus-Claude.

Self-MoA          — Princeton 2025 (arxiv.org/abs/2502.00674)
Adaptive Debate   — A-HMAD 2025 (doi.org/10.1007/s44443-025-00353-3)
Reflexion Loop    — Reflexion (arxiv.org/abs/2303.11366)
Graph-Mesh Collab — MultiAgentBench ACL 2025 (aclanthology.org/2025.acl-long.421)
"""

from __future__ import annotations

import asyncio
import json
import re

from models import AppConfig, LLMResponse, PipelineResult, PipelineSpec, StepTrace
from llm_client import LLMClient
from pipelines.base import BasePipeline


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _trace(phase: str, resp: LLMResponse) -> StepTrace:
    return StepTrace(
        phase=phase,
        model=resp.model,
        latency_ms=resp.latency_ms,
        tokens=resp.tokens_used,
        text_preview=resp.text[:200] if resp.text else resp.error,
    )


def _fail(pid: str, error: str, steps: tuple[StepTrace, ...]) -> PipelineResult:
    return PipelineResult(
        pipeline_id=pid, ok=False, error=error, steps=steps,
        total_latency_ms=sum(s.latency_ms for s in steps),
        total_tokens=sum(s.tokens for s in steps),
        llm_calls=len(steps),
    )


def _ok(pid: str, text: str, steps: tuple[StepTrace, ...]) -> PipelineResult:
    return PipelineResult(
        pipeline_id=pid, final_text=text, ok=True, steps=steps,
        total_latency_ms=sum(s.latency_ms for s in steps),
        total_tokens=sum(s.tokens for s in steps),
        llm_calls=len(steps),
    )


def _numbered(texts: list[str]) -> str:
    return "\n\n".join(f"Response {i}:\n{t}" for i, t in enumerate(texts, 1))


# ---------------------------------------------------------------------------
# 1. Self-MoA  (Princeton 2025)
#
# Key insight: intra-model diversity (same model, varied temperatures)
# outperforms inter-model diversity by 6.6% on AlpacaEval 2.0.
# Uses the SINGLE best model for all 3 layers instead of rotating.
# ---------------------------------------------------------------------------

class SelfMoAPipeline(BasePipeline):
    """Same-model MoA: 3 temperature samples → 3 self-refiners → 1 synthesis.
    All calls use the primary (strongest) model only — no model rotation."""

    _ID = "self_moa"
    _TEMPS_L1 = (0.2, 0.5, 0.8)
    _TEMPS_L2 = (0.15, 0.35, 0.55)

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID,
            name="Self-MoA",
            description=(
                "Same-model Mixture-of-Agents (Princeton 2025). "
                "Uses only the strongest model at varied temperatures "
                "across 3 layers, testing intra-model vs inter-model diversity."
            ),
            tier=4,
            estimated_calls=7,
            tags=("research", "self-moa", "single-model"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        # Force all layers to use the primary generator — no rotation
        best_model = self._role_model(config, model, "generator")
        steps: list[StepTrace] = []
        wrapped = self.wrap_task(prompt)

        # Layer 1: 3 diverse samples from the SAME model
        await self._notify(progress_callback, "Self-MoA Layer 1: 3 temperature-diverse samples...")
        l1_coros = tuple(
            client.generate(wrapped, model=best_model, temperature=t)
            for t in self._TEMPS_L1
        )
        l1_results = await asyncio.gather(*l1_coros)

        l1_texts: list[str] = []
        for idx, resp in enumerate(l1_results, 1):
            steps.append(_trace(f"self_l1_{idx}", resp))
            if resp.ok:
                l1_texts.append(resp.text)
        if not l1_texts:
            return _fail(self._ID, "All Layer 1 samples failed", tuple(steps))

        # Layer 2: 3 self-refiners see all L1 outputs (same model)
        await self._notify(progress_callback, "Self-MoA Layer 2: self-refining with all outputs...")
        combined = _numbered(l1_texts)
        l2_prompt = (
            f"Here are 3 responses to the same task from the same model at "
            f"different temperatures. Write an improved response that combines "
            f"the best reasoning and content from all three.\n\n{combined}"
        )
        l2_coros = tuple(
            client.generate(l2_prompt, model=best_model, temperature=t)
            for t in self._TEMPS_L2
        )
        l2_results = await asyncio.gather(*l2_coros)

        l2_texts: list[str] = []
        for idx, resp in enumerate(l2_results, 1):
            steps.append(_trace(f"self_l2_{idx}", resp))
            if resp.ok:
                l2_texts.append(resp.text)
        if not l2_texts:
            l2_texts = list(l1_texts)

        # Layer 3: final synthesis (same model, low temperature)
        await self._notify(progress_callback, "Self-MoA Layer 3: final synthesis...")
        l3_prompt = (
            f"Here are 3 refined responses. Produce the definitive final answer "
            f"that represents the strongest synthesis.\n\n{_numbered(l2_texts)}"
        )
        final = await client.generate(l3_prompt, model=best_model, temperature=0.15)
        steps.append(_trace("self_l3_synth", final))

        text = final.text if final.ok else max(l2_texts, key=len)
        return _ok(self._ID, text, tuple(steps))


# ---------------------------------------------------------------------------
# 2. Adaptive Heterogeneous Debate  (A-HMAD 2025)
#
# Key insight: specialist debaters + dynamic coordination outperform
# generic debate by +13.2% on GSM8K.
# ---------------------------------------------------------------------------

_SPECIALIST_PROMPTS = {
    "logical_reasoner": (
        "You are a logical reasoning specialist. Evaluate the answer's "
        "logical structure, identify any fallacies, verify deductive and "
        "inductive chains, and check mathematical/computational steps."
    ),
    "factual_verifier": (
        "You are a factual verification specialist. Check every factual "
        "claim, identify unsupported assertions, flag potential hallucinations, "
        "and verify that cited information is plausible and consistent."
    ),
    "strategic_planner": (
        "You are a strategic planning specialist. Evaluate whether the answer "
        "addresses the full scope of the task, check for missing requirements, "
        "assess the completeness of the solution, and identify blind spots."
    ),
}


class AdaptiveDebatePipeline(BasePipeline):
    """A-HMAD: specialist debaters with dynamic coordination."""

    _ID = "adaptive_debate"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID,
            name="Adaptive Debate",
            description=(
                "A-HMAD (2025): specialist debaters (logic, facts, strategy) "
                "challenge a draft through 2 rounds, with a consensus synthesizer."
            ),
            tier=4,
            estimated_calls=8,
            tags=("research", "debate", "specialist"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        gen_model = self._role_model(config, model, "generator")
        critic_model = self._role_model(config, model, "critic", "generator_alt", "generator")
        synth_model = self._role_model(config, model, "synthesizer", "generator")
        steps: list[StepTrace] = []

        # Step 1: Initial draft
        await self._notify(progress_callback, "A-HMAD: generating initial draft...")
        draft = await client.generate(self.wrap_task(prompt), model=gen_model)
        steps.append(_trace("draft", draft))
        if not draft.ok:
            return _fail(self._ID, draft.error, tuple(steps))

        current_answer = draft.text

        # Steps 2-3: Two specialist debate rounds
        for round_num in range(1, 3):
            await self._notify(
                progress_callback,
                f"A-HMAD round {round_num}: specialist critiques...",
            )

            # All specialists critique in parallel
            critique_coros = []
            specialist_names = list(_SPECIALIST_PROMPTS.keys())
            for name in specialist_names:
                system = _SPECIALIST_PROMPTS[name]
                critique_prompt = (
                    f"TASK: {prompt}\n\n"
                    f"CURRENT ANSWER:\n{current_answer}\n\n"
                    f"Provide your specialist critique. Identify specific issues "
                    f"within your domain of expertise. Be precise and actionable."
                )
                critique_coros.append(
                    client.generate(
                        critique_prompt, model=critic_model,
                        system_prompt=system, temperature=0.2,
                    )
                )
            critique_results = await asyncio.gather(*critique_coros)

            critiques: list[str] = []
            for idx, resp in enumerate(critique_results):
                steps.append(_trace(f"r{round_num}_{specialist_names[idx]}", resp))
                if resp.ok:
                    critiques.append(
                        f"[{specialist_names[idx].upper()}]: {resp.text}"
                    )

            if not critiques:
                break

            # Revise based on all specialist feedback
            await self._notify(
                progress_callback,
                f"A-HMAD round {round_num}: revising with specialist feedback...",
            )
            revise_prompt = (
                f"TASK: {prompt}\n\n"
                f"CURRENT ANSWER:\n{current_answer}\n\n"
                f"SPECIALIST CRITIQUES:\n" + "\n\n".join(critiques) + "\n\n"
                f"Revise the answer to address all valid specialist concerns. "
                f"Fix factual errors, strengthen logic, and fill gaps."
            )
            revised = await client.generate(
                revise_prompt, model=gen_model, temperature=0.25,
            )
            steps.append(_trace(f"r{round_num}_revise", revised))
            if revised.ok:
                current_answer = revised.text

        return _ok(self._ID, current_answer, tuple(steps))


# ---------------------------------------------------------------------------
# 3. Reflexion Loop  (Reflexion 2023, successors 2025)
#
# Key insight: verbal self-reflection with accumulated memory produces
# >18% accuracy gains vs simple critique-revise.
# ---------------------------------------------------------------------------

class ReflexionPipeline(BasePipeline):
    """Reflexion: attempt → evaluate → reflect on WHY it failed → retry."""

    _ID = "reflexion"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID,
            name="Reflexion Loop",
            description=(
                "Reflexion (2023+): generates, evaluates against rubric, "
                "reflects on WHY failures occurred (not just WHAT), then "
                "retries with accumulated reflection memory."
            ),
            tier=4,
            estimated_calls=6,
            tags=("research", "reflexion", "self-correction"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        gen_model = self._role_model(config, model, "generator")
        eval_model = self._role_model(config, model, "critic", "generator_alt", "generator")
        steps: list[StepTrace] = []
        reflection_memory: list[str] = []

        for attempt in range(1, 3):  # up to 2 attempts
            # Generate (with reflection context if available)
            await self._notify(
                progress_callback,
                f"Reflexion attempt {attempt}: generating answer...",
            )
            if reflection_memory:
                gen_prompt = (
                    f"{self.wrap_task(prompt)}\n\n"
                    f"REFLECTION FROM PREVIOUS ATTEMPT:\n"
                    + "\n".join(reflection_memory)
                    + "\n\nUse these reflections to produce a better answer."
                )
            else:
                gen_prompt = self.wrap_task(prompt)

            resp = await client.generate(gen_prompt, model=gen_model)
            steps.append(_trace(f"attempt_{attempt}", resp))
            if not resp.ok:
                return _fail(self._ID, resp.error, tuple(steps))

            current_answer = resp.text

            # Evaluate against the task rubric
            await self._notify(
                progress_callback,
                f"Reflexion attempt {attempt}: evaluating against rubric...",
            )
            eval_prompt = (
                f"TASK: {prompt}\n\n"
                f"ANSWER:\n{current_answer}\n\n"
                f"Evaluate this answer. Score it 0-100 and identify specific "
                f"weaknesses. Reply with JSON:\n"
                f'{{"score": N, "weaknesses": ["...", "..."], "verdict": "pass|fail"}}'
            )
            eval_resp = await client.generate(
                eval_prompt, model=eval_model, temperature=0.1,
            )
            steps.append(_trace(f"evaluate_{attempt}", eval_resp))
            if not eval_resp.ok:
                break

            evaluation = _parse_evaluation(eval_resp.text)
            if evaluation.get("verdict") == "pass" or evaluation.get("score", 0) >= 85:
                break  # Good enough, stop iterating

            # Reflect: articulate WHY it failed (the key Reflexion insight)
            await self._notify(
                progress_callback,
                f"Reflexion attempt {attempt}: reflecting on failures...",
            )
            weaknesses = evaluation.get("weaknesses", [])
            weakness_text = "\n".join(f"- {w}" for w in weaknesses) if weaknesses else "General quality issues"

            reflect_prompt = (
                f"TASK: {prompt}\n\n"
                f"YOUR ANSWER:\n{current_answer}\n\n"
                f"IDENTIFIED WEAKNESSES:\n{weakness_text}\n\n"
                f"Reflect deeply on WHY your answer had these weaknesses. "
                f"What assumptions did you make incorrectly? What knowledge "
                f"did you lack or misapply? What strategy should you use "
                f"differently next time? Be specific and self-critical."
            )
            reflect_resp = await client.generate(
                reflect_prompt, model=gen_model, temperature=0.3,
            )
            steps.append(_trace(f"reflect_{attempt}", reflect_resp))
            if reflect_resp.ok:
                reflection_memory.append(
                    f"[Attempt {attempt} reflection]: {reflect_resp.text}"
                )

        return _ok(self._ID, current_answer, tuple(steps))


def _parse_evaluation(text: str) -> dict:
    json_match = re.search(r"\{[^}]*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except (json.JSONDecodeError, TypeError):
            pass
    lower = text.lower()
    if any(w in lower for w in ("excellent", "strong", "well-written", "comprehensive")):
        return {"score": 85, "verdict": "pass", "weaknesses": []}
    return {"score": 50, "verdict": "fail", "weaknesses": ["Could not parse evaluation"]}


# ---------------------------------------------------------------------------
# 4. Graph-Mesh Collaboration  (MultiAgentBench ACL 2025)
#
# Key insight: all-to-all communication topology outperforms star/chain/tree
# because every agent benefits from every other agent's latest work.
# ---------------------------------------------------------------------------

class GraphMeshPipeline(BasePipeline):
    """Graph-mesh: 3 agents share work in rounds, all see all outputs."""

    _ID = "graph_mesh"
    _AGENTS = 3
    _ROUNDS = 3

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID,
            name="Graph-Mesh Collab",
            description=(
                "MultiAgentBench (ACL 2025): 3 agents collaborate over 3 rounds. "
                "Each round, every agent sees ALL other agents' latest output "
                "and produces an improved version. Final synthesis merges round 3."
            ),
            tier=5,
            estimated_calls=10,
            tags=("research", "graph-mesh", "topology"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        # Use 3 different models for diversity (or same model if fewer available)
        models = [
            self._role_model(config, model, "generator"),
            self._role_model(config, model, "generator_alt", "critic", "generator"),
            self._role_model(config, model, "reviewer", "synthesizer", "generator"),
        ]
        steps: list[StepTrace] = []
        wrapped = self.wrap_task(prompt)

        # Round 0: independent initial generation
        await self._notify(progress_callback, "Graph-mesh round 0: independent generation...")
        init_coros = tuple(
            client.generate(wrapped, model=models[i], temperature=0.3 + i * 0.2)
            for i in range(self._AGENTS)
        )
        init_results = await asyncio.gather(*init_coros)

        current_texts: list[str] = []
        for idx, resp in enumerate(init_results):
            steps.append(_trace(f"r0_agent_{idx}", resp))
            current_texts.append(resp.text if resp.ok else "")

        if not any(current_texts):
            return _fail(self._ID, "All initial generations failed", tuple(steps))

        # Rounds 1-3: each agent sees ALL others and improves
        for round_num in range(1, self._ROUNDS + 1):
            await self._notify(
                progress_callback,
                f"Graph-mesh round {round_num}: all-to-all collaboration...",
            )

            round_coros = []
            for agent_idx in range(self._AGENTS):
                # Build context: all OTHER agents' current outputs
                others = [
                    f"[Agent {j}]:\n{current_texts[j]}"
                    for j in range(self._AGENTS)
                    if j != agent_idx and current_texts[j]
                ]
                other_text = "\n\n".join(others)

                mesh_prompt = (
                    f"TASK: {prompt}\n\n"
                    f"YOUR CURRENT ANSWER:\n{current_texts[agent_idx]}\n\n"
                    f"OTHER AGENTS' LATEST ANSWERS:\n{other_text}\n\n"
                    f"Write an improved answer. Incorporate insights from other "
                    f"agents that strengthen your response. Fix any errors they "
                    f"caught. Add perspectives you missed. Do not simply merge — "
                    f"produce your own best answer informed by all inputs."
                )
                round_coros.append(
                    client.generate(
                        mesh_prompt, model=models[agent_idx],
                        temperature=max(0.15, 0.4 - round_num * 0.1),
                    )
                )

            round_results = await asyncio.gather(*round_coros)
            for idx, resp in enumerate(round_results):
                steps.append(_trace(f"r{round_num}_agent_{idx}", resp))
                if resp.ok:
                    current_texts[idx] = resp.text

        # Final synthesis
        await self._notify(progress_callback, "Graph-mesh: final synthesis...")
        synth_model = self._role_model(config, model, "synthesizer", "generator")
        synth_prompt = (
            f"TASK: {prompt}\n\n"
            f"Three agents collaborated over {self._ROUNDS} rounds. "
            f"Here are their final answers:\n\n"
            + "\n\n".join(
                f"[Agent {i} final]:\n{t}" for i, t in enumerate(current_texts) if t
            )
            + "\n\nProduce the definitive final answer."
        )
        final = await client.generate(synth_prompt, model=synth_model, temperature=0.2)
        steps.append(_trace("synthesis", final))

        valid = [t for t in current_texts if t]
        text = final.text if final.ok else max(valid, key=len) if valid else ""
        return _ok(self._ID, text, tuple(steps))
