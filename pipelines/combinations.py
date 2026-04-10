"""Combination pipelines — merging the top-performing architectures.

Variant A: Graph-Mesh → Verification (collaboration + validation)
Variant B: Graph-Mesh → Ranked Selection (collaboration + selection)
Variant C: Ranked Selection → Verification / GSV (selection + validation)
Variant D: Graph-Mesh → Ranked → Verification (all three)
Variant E: Verify-first adaptive cascade (cheap fast path, heavy fallback)
"""

from __future__ import annotations

import asyncio
import json
import re

from models import AppConfig, LLMResponse, PipelineResult, PipelineSpec, StepTrace
from llm_client import LLMClient
from pipelines.base import BasePipeline


def _trace(phase: str, resp: LLMResponse) -> StepTrace:
    return StepTrace(
        phase=phase, model=resp.model, latency_ms=resp.latency_ms,
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


# ── Shared sub-routines ──────────────────────────────────────────────

async def _graph_mesh_rounds(
    prompt: str, client: LLMClient, models: list[str],
    steps: list[StepTrace], progress_callback, *, cot_prompt: str,
    agents: int = 3, rounds: int = 3,
) -> list[str]:
    """Run graph-mesh collaboration, return each agent's final text."""
    # Round 0: independent generation
    await _notify(progress_callback, "Graph-mesh: independent generation...")
    init_coros = tuple(
        client.generate(cot_prompt, model=models[i % len(models)], temperature=0.3 + i * 0.15)
        for i in range(agents)
    )
    init_results = await asyncio.gather(*init_coros)
    texts: list[str] = []
    for idx, resp in enumerate(init_results):
        steps.append(_trace(f"mesh_r0_{idx}", resp))
        texts.append(resp.text if resp.ok else "")

    # Rounds 1-N: all-to-all
    for rnd in range(1, rounds + 1):
        await _notify(progress_callback, f"Graph-mesh round {rnd}...")
        coros = []
        for i in range(agents):
            others = "\n\n".join(
                f"[Agent {j}]:\n{texts[j]}" for j in range(agents) if j != i and texts[j]
            )
            mesh_prompt = (
                f"TASK: {prompt}\n\nYOUR CURRENT ANSWER:\n{texts[i]}\n\n"
                f"OTHER AGENTS' ANSWERS:\n{others}\n\n"
                f"Write an improved answer incorporating the best insights."
            )
            coros.append(client.generate(
                mesh_prompt, model=models[i % len(models)],
                temperature=max(0.15, 0.35 - rnd * 0.08),
            ))
        results = await asyncio.gather(*coros)
        for idx, resp in enumerate(results):
            steps.append(_trace(f"mesh_r{rnd}_{idx}", resp))
            if resp.ok:
                texts[idx] = resp.text
    return texts


async def _ranked_select(
    prompt: str, candidates: list[str], client: LLMClient,
    judge_model: str, steps: list[StepTrace], progress_callback,
) -> tuple[str, int]:
    """Pairwise tournament to pick the best candidate. Returns (best_text, best_idx)."""
    await _notify(progress_callback, "Ranked selection: pairwise comparison...")
    remaining = list(range(len(candidates)))

    while len(remaining) > 1:
        next_round = []
        for i in range(0, len(remaining), 2):
            if i + 1 >= len(remaining):
                next_round.append(remaining[i])
                continue
            a_idx, b_idx = remaining[i], remaining[i + 1]
            compare_prompt = (
                f"TASK: {prompt}\n\n"
                f"ANSWER A:\n{candidates[a_idx]}\n\n"
                f"ANSWER B:\n{candidates[b_idx]}\n\n"
                f"Which answer is better for the task? Reply with ONLY 'A' or 'B'."
            )
            resp = await client.generate(
                compare_prompt, model=judge_model, temperature=0.1,
            )
            steps.append(_trace(f"rank_compare", resp))
            choice = (resp.text or "").strip().upper()
            next_round.append(b_idx if "B" in choice else a_idx)
        remaining = next_round

    best_idx = remaining[0]
    return candidates[best_idx], best_idx


async def _verify_and_fix(
    prompt: str, text: str, client: LLMClient,
    checker_model: str, fixer_model: str,
    steps: list[StepTrace], progress_callback,
    max_cycles: int = 2,
) -> str:
    """Structured verification + targeted fix loop."""
    current = text
    for cycle in range(max_cycles):
        await _notify(progress_callback, f"Verification cycle {cycle + 1}...")
        check_prompt = (
            f"You are a strict constraint validator.\n\n"
            f"ORIGINAL TASK: {prompt}\n\n"
            f"RESPONSE TO CHECK:\n{current}\n\n"
            f"List EVERY issue: factual errors, missing requirements, "
            f"constraint violations. Be precise.\n"
            f"If everything passes, say 'ALL CHECKS PASSED'.\n"
            f'Reply JSON: {{"issues": ["..."], "all_passed": true|false}}'
        )
        check_resp = await client.generate(check_prompt, model=checker_model, temperature=0.1)
        steps.append(_trace(f"verify_{cycle + 1}", check_resp))
        if not check_resp.ok:
            break

        check_text = check_resp.text or ""
        if "ALL CHECKS PASSED" in check_text.upper() or '"all_passed": true' in check_text.lower():
            break

        issues = _extract_issues(check_text)
        if not issues:
            break

        await _notify(progress_callback, f"Fixing {len(issues)} issues...")
        fix_prompt = (
            f"TASK: {prompt}\n\n"
            f"CURRENT RESPONSE:\n{current}\n\n"
            f"ISSUES FOUND:\n" + "\n".join(f"- {iss}" for iss in issues) + "\n\n"
            f"Fix ONLY the identified issues. Do not change what already works."
        )
        fix_resp = await client.generate(fix_prompt, model=fixer_model, temperature=0.2)
        steps.append(_trace(f"fix_{cycle + 1}", fix_resp))
        if fix_resp.ok and fix_resp.text:
            current = fix_resp.text
    return current


def _extract_issues(text: str) -> list[str]:
    if "ALL CHECKS PASSED" in text.upper():
        return []
    json_match = re.search(r"\{.*\"issues\"\s*:\s*\[.*?\].*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if parsed.get("all_passed"):
                return []
            return [str(i) for i in parsed.get("issues", []) if i]
        except (json.JSONDecodeError, TypeError):
            pass
    lines = [l.strip().lstrip("-•*").strip() for l in text.splitlines() if l.strip().startswith(("-", "•", "*"))]
    return [l for l in lines if len(l) > 10][:5]


async def _notify(cb, msg: str) -> None:
    if cb:
        await cb(msg)


# ── Variant A: Graph-Mesh → Verification ─────────────────────────────

class MeshVerifyPipeline(BasePipeline):
    """Graph-mesh collaboration → structured verification → targeted fix."""

    _ID = "mesh_verify"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID, name="Mesh + Verify",
            description="Graph-mesh 3-agent collaboration, then structured verification with targeted fixes.",
            tier=5, estimated_calls=14,
            tags=("combination", "graph-mesh", "verification"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        models = self._diverse_models(config, model)
        synth_model = self._role_model(config, model, "synthesizer", "generator")
        checker_model = self._role_model(config, model, "critic", "generator")
        steps: list[StepTrace] = []
        wrapped = self.wrap_task(prompt, cot=self._cot(config))

        texts = await _graph_mesh_rounds(
            prompt, client, models, steps, progress_callback,
            cot_prompt=wrapped,
        )

        valid = [t for t in texts if t]
        if not valid:
            return _fail(self._ID, "Graph-mesh produced no valid outputs", tuple(steps))

        # Synthesise mesh outputs
        await _notify(progress_callback, "Synthesising mesh outputs...")
        synth_prompt = (
            f"TASK: {prompt}\n\n"
            + "\n\n".join(f"[Agent {i}]:\n{t}" for i, t in enumerate(texts) if t)
            + "\n\nProduce the definitive final answer."
        )
        synth = await client.generate(synth_prompt, model=synth_model, temperature=0.2)
        steps.append(_trace("synthesise", synth))
        current = synth.text if synth.ok else max(valid, key=len)

        # Verification + fix
        current = await _verify_and_fix(
            prompt, current, client, checker_model, synth_model, steps, progress_callback,
        )
        return _ok(self._ID, current, tuple(steps))


# ── Variant B: Graph-Mesh → Ranked Selection ─────────────────────────

class MeshRankedPipeline(BasePipeline):
    """Graph-mesh collaboration → pairwise tournament selection of best agent."""

    _ID = "mesh_ranked"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID, name="Mesh + Ranked",
            description="Graph-mesh 3-agent collaboration, then pairwise tournament picks the best agent's output.",
            tier=5, estimated_calls=13,
            tags=("combination", "graph-mesh", "ranking"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        models = self._diverse_models(config, model)
        judge_model = self._role_model(config, model, "critic", "generator")
        steps: list[StepTrace] = []
        wrapped = self.wrap_task(prompt, cot=self._cot(config))

        texts = await _graph_mesh_rounds(
            prompt, client, models, steps, progress_callback,
            cot_prompt=wrapped,
        )

        valid = [t for t in texts if t]
        if not valid:
            return _fail(self._ID, "Graph-mesh produced no valid outputs", tuple(steps))

        best_text, _ = await _ranked_select(
            prompt, valid, client, judge_model, steps, progress_callback,
        )
        return _ok(self._ID, best_text, tuple(steps))


# ── Variant C: Generate-Select-Verify (GSV) ──────────────────────────

class GSVPipeline(BasePipeline):
    """Generate diverse candidates → pairwise select best → verify → fix."""

    _ID = "gsv"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID, name="Generate-Select-Verify",
            description="5 diverse candidates across models/temps, pairwise tournament, structured verification, targeted fix.",
            tier=5, estimated_calls=12,
            tags=("combination", "selection", "verification"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        models = self._diverse_models(config, model)
        checker_model = self._role_model(config, model, "critic", "generator")
        fixer_model = self._role_model(config, model, "synthesizer", "generator")
        steps: list[StepTrace] = []
        wrapped = self.wrap_task(prompt, cot=self._cot(config))

        # Generate 5 candidates: cycle through models and temperatures
        temps = [0.2, 0.4, 0.6, 0.3, 0.5]
        await _notify(progress_callback, "GSV: generating 5 diverse candidates...")
        gen_coros = tuple(
            client.generate(wrapped, model=models[i % len(models)], temperature=temps[i])
            for i in range(5)
        )
        gen_results = await asyncio.gather(*gen_coros)

        candidates: list[str] = []
        for idx, resp in enumerate(gen_results):
            steps.append(_trace(f"gen_{idx}", resp))
            if resp.ok and resp.text:
                candidates.append(resp.text)

        if not candidates:
            return _fail(self._ID, "All generation calls failed", tuple(steps))

        # Pairwise tournament selection
        best_text, _ = await _ranked_select(
            prompt, candidates, client, checker_model, steps, progress_callback,
        )

        # Structured verification + fix
        best_text = await _verify_and_fix(
            prompt, best_text, client, checker_model, fixer_model, steps, progress_callback,
        )
        return _ok(self._ID, best_text, tuple(steps))


# ── Variant D: Graph-Mesh → Ranked → Verification (full combo) ──────

class MeshRankedVerifyPipeline(BasePipeline):
    """The full combination: graph-mesh → ranked selection → verification → fix."""

    _ID = "mesh_ranked_verify"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID, name="Mesh+Ranked+Verify",
            description="Full combination: 3-agent graph-mesh collaboration, pairwise tournament, structured verification, targeted fix.",
            tier=5, estimated_calls=17,
            tags=("combination", "graph-mesh", "ranking", "verification"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        models = self._diverse_models(config, model)
        checker_model = self._role_model(config, model, "critic", "generator")
        fixer_model = self._role_model(config, model, "synthesizer", "generator")
        steps: list[StepTrace] = []
        wrapped = self.wrap_task(prompt, cot=self._cot(config))

        # Phase 1: Graph-mesh collaboration
        texts = await _graph_mesh_rounds(
            prompt, client, models, steps, progress_callback,
            cot_prompt=wrapped,
        )
        valid = [t for t in texts if t]
        if not valid:
            return _fail(self._ID, "Graph-mesh produced no valid outputs", tuple(steps))

        # Phase 2: Ranked selection
        best_text, _ = await _ranked_select(
            prompt, valid, client, checker_model, steps, progress_callback,
        )

        # Phase 3: Verification + fix
        best_text = await _verify_and_fix(
            prompt, best_text, client, checker_model, fixer_model, steps, progress_callback,
        )
        return _ok(self._ID, best_text, tuple(steps))


# ── Variant E: Verify-First Adaptive Cascade ─────────────────────────

class AdaptiveCascadePipeline(BasePipeline):
    """Cheap path first: generate → verify. If passes, done (2 calls).
    If fails, escalate to full graph-mesh → verify (15 calls)."""

    _ID = "adaptive_cascade"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID, name="Adaptive Cascade",
            description="Verify-first: single model + verification (2 calls if passes). Escalates to graph-mesh only on failure.",
            tier=4, estimated_calls=8,
            tags=("combination", "adaptive", "cascade", "verification"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        gen_model = self._role_model(config, model, "generator")
        checker_model = self._role_model(config, model, "critic", "generator")
        models = self._diverse_models(config, model)
        steps: list[StepTrace] = []
        wrapped = self.wrap_task(prompt, cot=self._cot(config))

        # Fast path: single model generation
        await _notify(progress_callback, "Cascade: fast path (single model)...")
        resp = await client.generate(wrapped, model=gen_model)
        steps.append(_trace("fast_gen", resp))
        if not resp.ok:
            return _fail(self._ID, resp.error, tuple(steps))

        # Quick verification
        await _notify(progress_callback, "Cascade: quick verification...")
        check_prompt = (
            f"TASK: {prompt}\n\nRESPONSE:\n{resp.text}\n\n"
            f"Does this response fully and correctly answer the task? "
            f"Reply JSON: {{\"pass\": true|false, \"issues\": [\"...\"]}}"
        )
        check = await client.generate(check_prompt, model=checker_model, temperature=0.1)
        steps.append(_trace("fast_verify", check))

        # Check if fast path passes
        check_text = (check.text or "").lower()
        passed = '"pass": true' in check_text or '"pass":true' in check_text
        issues = _extract_issues(check.text or "")

        if passed and not issues:
            await _notify(progress_callback, "Cascade: fast path PASSED (2 calls)")
            return _ok(self._ID, resp.text, tuple(steps))

        # Slow path: escalate to graph-mesh
        await _notify(progress_callback, "Cascade: escalating to graph-mesh (verification failed)...")
        texts = await _graph_mesh_rounds(
            prompt, client, models, steps, progress_callback,
            cot_prompt=wrapped, rounds=2,  # fewer rounds for the cascade
        )
        valid = [t for t in texts if t]
        if not valid:
            return _ok(self._ID, resp.text, tuple(steps))  # fall back to fast path

        # Synthesise
        synth_model = self._role_model(config, model, "synthesizer", "generator")
        synth_prompt = (
            f"TASK: {prompt}\n\n"
            f"INITIAL ATTEMPT (had issues):\n{resp.text}\n\n"
            f"ISSUES FOUND: {'; '.join(issues) if issues else 'quality concerns'}\n\n"
            f"IMPROVED VERSIONS:\n"
            + "\n\n".join(f"[Version {i}]:\n{t}" for i, t in enumerate(valid))
            + "\n\nProduce the best final answer, fixing the original issues."
        )
        synth = await client.generate(synth_prompt, model=synth_model, temperature=0.2)
        steps.append(_trace("cascade_synth", synth))

        final = synth.text if synth.ok else max(valid, key=len)

        # Final verification pass
        final = await _verify_and_fix(
            prompt, final, client, checker_model, synth_model, steps, progress_callback,
            max_cycles=1,
        )
        return _ok(self._ID, final, tuple(steps))
