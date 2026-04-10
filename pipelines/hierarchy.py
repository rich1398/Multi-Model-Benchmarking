"""Corporate hierarchy topology pipelines for Occursus Benchmark.

Based on the "Organisational Hierarchies as a Topology" design document.
Models an AI system as a corporate organisation with tiered escalation.

Managed Team  — T2: lead decomposes → specialists execute → critic reviews → verifier checks
Corp Hierarchy — T0-T3 adaptive: router classifies task complexity, dispatches to cheapest viable tier
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


async def _notify(cb, msg: str) -> None:
    if cb:
        await cb(msg)


# ── Shared prompts ───────────────────────────────────────────────────

CLASSIFY_PROMPT = (
    "You are an executive triage router. Classify this task on 5 dimensions "
    "(each 1-5):\n"
    "- complexity: how many steps or sub-problems?\n"
    "- ambiguity: how open-ended or underspecified?\n"
    "- stakes: how bad is a wrong answer?\n"
    "- tool_dependence: does it need code execution, calculation, or retrieval?\n"
    "- domain_mix: how many distinct knowledge domains?\n\n"
    "TASK: {prompt}\n\n"
    "Reply with ONLY valid JSON:\n"
    '{{"complexity": N, "ambiguity": N, "stakes": N, '
    '"tool_dependence": N, "domain_mix": N, "tier": "T0|T1|T2|T3", '
    '"reasoning": "one sentence"}}'
)

LEAD_DECOMPOSE_PROMPT = (
    "You are a team lead. Decompose this task into 2-4 specialist work items. "
    "For each, specify the specialist type and what they must deliver.\n\n"
    "TASK: {prompt}\n\n"
    "Reply with ONLY valid JSON:\n"
    '{{"work_items": [{{"specialist": "analyst|coder|writer|researcher|planner", '
    '"deliverable": "what to produce"}}]}}'
)

SPECIALIST_PROMPTS = {
    "analyst": "You are a senior analyst. Focus on data, calculations, logic, and evidence-based reasoning.",
    "coder": "You are a senior software engineer. Focus on correct, production-quality code with proper error handling.",
    "writer": "You are a senior writer. Focus on clarity, structure, and persuasive communication.",
    "researcher": "You are a senior researcher. Focus on depth, accuracy, nuance, and citing evidence.",
    "planner": "You are a senior planner. Focus on feasibility, timelines, risks, and actionable steps.",
}

CRITIC_PROMPT = (
    "You are an independent reviewer. You did NOT produce this work. "
    "Examine it critically against the original task requirements.\n\n"
    "ORIGINAL TASK: {prompt}\n\n"
    "WORK TO REVIEW:\n{work}\n\n"
    "List specific issues: factual errors, missing requirements, logical gaps, "
    "quality problems. If the work is acceptable, say 'APPROVED'.\n"
    'Reply JSON: {{"approved": true|false, "issues": ["..."]}}'
)

VERIFIER_PROMPT = (
    "You are a compliance verifier. Check this final answer against the "
    "original task requirements using a structured checklist.\n\n"
    "ORIGINAL TASK: {prompt}\n\n"
    "FINAL ANSWER:\n{answer}\n\n"
    "For each requirement in the task, state PASS or FAIL with a reason.\n"
    'Reply JSON: {{"all_pass": true|false, "checks": [{{"requirement": "...", "status": "PASS|FAIL", "reason": "..."}}]}}'
)


def _parse_json_safe(text: str) -> dict:
    """Extract first JSON object from text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _parse_work_items(text: str) -> list[dict]:
    parsed = _parse_json_safe(text)
    items = parsed.get("work_items", [])
    if isinstance(items, list):
        return [
            {"specialist": it.get("specialist", "analyst"), "deliverable": it.get("deliverable", "")}
            for it in items if isinstance(it, dict) and it.get("deliverable")
        ]
    return []


def _parse_tier(text: str) -> tuple[str, int]:
    """Parse classifier output, return (tier, total_score)."""
    parsed = _parse_json_safe(text)
    tier = parsed.get("tier", "T1")
    dims = ["complexity", "ambiguity", "stakes", "tool_dependence", "domain_mix"]
    total = sum(int(parsed.get(d, 2)) for d in dims)
    if tier not in ("T0", "T1", "T2", "T3"):
        if total <= 8:
            tier = "T0"
        elif total <= 12:
            tier = "T1"
        elif total <= 18:
            tier = "T2"
        else:
            tier = "T3"
    return tier, total


# ── Managed Team Pipeline (T2) ───────────────────────────────────────

class ManagedTeamPipeline(BasePipeline):
    """T2 corporate structure: lead decomposes → specialists execute →
    independent critic reviews → verifier checks."""

    _ID = "managed_team"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID, name="Managed Team",
            description=(
                "Corporate T2: lead decomposes task, specialists execute in parallel "
                "(each a different model), independent critic reviews, verifier checks."
            ),
            tier=4, estimated_calls=8,
            tags=("hierarchy", "managed-team", "specialist", "critique"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        diverse = self._diverse_models(config, model)
        lead_model = self._role_model(config, model, "generator")
        critic_model = self._role_model(config, model, "critic", "generator_alt", "generator")
        verifier_model = self._role_model(config, model, "reviewer", "generator")
        steps: list[StepTrace] = []

        # Phase 1: Lead decomposes the task
        await _notify(progress_callback, "Lead: decomposing task...")
        decomp_resp = await client.generate(
            LEAD_DECOMPOSE_PROMPT.format(prompt=prompt),
            model=lead_model, temperature=0.15,
        )
        steps.append(_trace("lead_decompose", decomp_resp))
        if not decomp_resp.ok:
            return _fail(self._ID, decomp_resp.error, tuple(steps))

        work_items = _parse_work_items(decomp_resp.text)
        if not work_items:
            work_items = [{"specialist": "analyst", "deliverable": prompt}]

        # Phase 2: Specialists execute in parallel (different models)
        await _notify(progress_callback, f"Specialists: {len(work_items)} work items...")
        spec_coros = []
        for i, item in enumerate(work_items[:4]):
            spec_type = item["specialist"]
            system = SPECIALIST_PROMPTS.get(spec_type, SPECIALIST_PROMPTS["analyst"])
            spec_prompt = (
                f"ORIGINAL TASK: {prompt}\n\n"
                f"YOUR ASSIGNMENT: {item['deliverable']}\n\n"
                f"Produce your deliverable completely and accurately."
            )
            spec_coros.append(client.generate(
                spec_prompt, model=diverse[i % len(diverse)],
                system_prompt=system, temperature=0.3,
            ))

        spec_results = await asyncio.gather(*spec_coros)
        deliverables: list[str] = []
        for i, resp in enumerate(spec_results):
            steps.append(_trace(f"specialist_{work_items[i]['specialist']}", resp))
            if resp.ok and resp.text:
                deliverables.append(f"[{work_items[i]['specialist'].upper()}]: {resp.text}")

        if not deliverables:
            return _fail(self._ID, "All specialists failed", tuple(steps))

        # Phase 3: Lead assembles deliverables
        await _notify(progress_callback, "Lead: assembling deliverables...")
        assemble_prompt = (
            f"TASK: {prompt}\n\n"
            f"SPECIALIST DELIVERABLES:\n" + "\n\n".join(deliverables) + "\n\n"
            f"Assemble these into one complete, coherent answer. "
            f"Resolve any contradictions. Do not mention the specialists."
        )
        assembled = await client.generate(assemble_prompt, model=lead_model, temperature=0.2)
        steps.append(_trace("lead_assemble", assembled))
        current = assembled.text if assembled.ok else max(deliverables, key=len)

        # Phase 4: Independent critic review
        await _notify(progress_callback, "Critic: independent review...")
        critic_resp = await client.generate(
            CRITIC_PROMPT.format(prompt=prompt, work=current),
            model=critic_model, temperature=0.1,
        )
        steps.append(_trace("critic_review", critic_resp))

        critique = _parse_json_safe(critic_resp.text) if critic_resp.ok else {}
        issues = critique.get("issues", [])

        if issues and not critique.get("approved", True):
            await _notify(progress_callback, "Lead: addressing critique...")
            fix_prompt = (
                f"TASK: {prompt}\n\nCURRENT ANSWER:\n{current}\n\n"
                f"REVIEWER ISSUES:\n" + "\n".join(f"- {i}" for i in issues) + "\n\n"
                f"Address all valid issues. Do not change what already works."
            )
            fixed = await client.generate(fix_prompt, model=lead_model, temperature=0.2)
            steps.append(_trace("lead_fix", fixed))
            if fixed.ok and fixed.text:
                current = fixed.text

        # Phase 5: Verifier check
        await _notify(progress_callback, "Verifier: compliance check...")
        verify_resp = await client.generate(
            VERIFIER_PROMPT.format(prompt=prompt, answer=current),
            model=verifier_model, temperature=0.1,
        )
        steps.append(_trace("verifier", verify_resp))

        verification = _parse_json_safe(verify_resp.text) if verify_resp.ok else {}
        if not verification.get("all_pass", True):
            failed_checks = [
                c.get("requirement", "")
                for c in verification.get("checks", [])
                if c.get("status") == "FAIL"
            ]
            if failed_checks:
                await _notify(progress_callback, "Lead: fixing verification failures...")
                vfix_prompt = (
                    f"TASK: {prompt}\n\nCURRENT ANSWER:\n{current}\n\n"
                    f"FAILED CHECKS:\n" + "\n".join(f"- {f}" for f in failed_checks) + "\n\n"
                    f"Fix ONLY the failed checks."
                )
                vfixed = await client.generate(vfix_prompt, model=lead_model, temperature=0.2)
                steps.append(_trace("verifier_fix", vfixed))
                if vfixed.ok and vfixed.text:
                    current = vfixed.text

        return _ok(self._ID, current, tuple(steps))


# ── Corp Hierarchy Pipeline (T0-T3 adaptive) ─────────────────────────

class CorpHierarchyPipeline(BasePipeline):
    """Full 4-tier corporate hierarchy with adaptive routing.
    T0: fast local model. T1: strong single flagship. T2: managed team.
    T3: multi-clique deliberation with rotating model roles."""

    _ID = "corp_hierarchy"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID, name="Corp Hierarchy",
            description=(
                "4-tier corporate hierarchy: router classifies task complexity, "
                "dispatches to cheapest viable tier. T0=local, T1=flagship, "
                "T2=managed team, T3=multi-clique deliberation."
            ),
            tier=5, estimated_calls=10,
            tags=("hierarchy", "adaptive", "routing", "corporate"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        diverse = self._diverse_models(config, model)
        primary = self._role_model(config, model, "generator")
        critic_model = self._role_model(config, model, "critic", "generator_alt", "generator")
        verifier_model = self._role_model(config, model, "reviewer", "generator")
        steps: list[StepTrace] = []

        # Phase 1: Executive triage — classify the task
        await _notify(progress_callback, "Executive triage: classifying task...")
        classify_resp = await client.generate(
            CLASSIFY_PROMPT.format(prompt=prompt),
            model=primary, temperature=0.1,
        )
        steps.append(_trace("triage", classify_resp))

        tier, score = _parse_tier(classify_resp.text) if classify_resp.ok else ("T2", 15)
        await _notify(progress_callback, f"Tier assigned: {tier} (score {score})")

        # ── T0: Fast path (single cheap model, no review) ──
        if tier == "T0":
            await _notify(progress_callback, "T0: fast single model...")
            resp = await client.generate(
                self.wrap_task(prompt, cot=self._cot(config)),
                model=diverse[-1] if len(diverse) > 1 else primary,  # cheapest model
                temperature=0.2,
            )
            steps.append(_trace("t0_generate", resp))
            if resp.ok:
                return _ok(self._ID, resp.text, tuple(steps))
            # Fall through to T1 on failure

        # ── T1: Strong single flagship ──
        if tier in ("T0", "T1"):
            await _notify(progress_callback, "T1: strong flagship model...")
            resp = await client.generate(
                self.wrap_task(prompt, cot=self._cot(config)),
                model=primary, temperature=0.25,
            )
            steps.append(_trace("t1_generate", resp))

            # Quick verification gate
            await _notify(progress_callback, "T1: verification gate...")
            gate_resp = await client.generate(
                f"TASK: {prompt}\n\nANSWER: {resp.text}\n\n"
                f"Does this fully answer the task? Reply ONLY 'PASS' or 'FAIL: reason'.",
                model=critic_model, temperature=0.1,
            )
            steps.append(_trace("t1_gate", gate_resp))

            gate_text = (gate_resp.text or "").strip().upper()
            if resp.ok and ("PASS" in gate_text and "FAIL" not in gate_text):
                return _ok(self._ID, resp.text, tuple(steps))
            # Fall through to T2

        # ── T2: Managed team (decompose → specialists → critique → verify) ──
        if tier in ("T0", "T1", "T2"):
            await _notify(progress_callback, "T2: managed team activation...")

            # Lead decomposition
            decomp_resp = await client.generate(
                LEAD_DECOMPOSE_PROMPT.format(prompt=prompt),
                model=primary, temperature=0.15,
            )
            steps.append(_trace("t2_decompose", decomp_resp))

            work_items = _parse_work_items(decomp_resp.text) if decomp_resp.ok else []
            if not work_items:
                work_items = [{"specialist": "analyst", "deliverable": prompt}]

            # Parallel specialist execution
            await _notify(progress_callback, f"T2: {len(work_items)} specialists working...")
            spec_coros = []
            for i, item in enumerate(work_items[:4]):
                system = SPECIALIST_PROMPTS.get(item["specialist"], SPECIALIST_PROMPTS["analyst"])
                spec_coros.append(client.generate(
                    f"TASK: {prompt}\n\nYOUR ASSIGNMENT: {item['deliverable']}",
                    model=diverse[i % len(diverse)], system_prompt=system, temperature=0.3,
                ))
            spec_results = await asyncio.gather(*spec_coros)
            deliverables = []
            for i, resp in enumerate(spec_results):
                steps.append(_trace(f"t2_spec_{work_items[i]['specialist']}", resp))
                if resp.ok:
                    deliverables.append(f"[{work_items[i]['specialist'].upper()}]: {resp.text}")

            # Assemble + critique
            assemble_prompt = (
                f"TASK: {prompt}\n\nSPECIALIST WORK:\n" + "\n\n".join(deliverables or ["No specialist output"])
                + "\n\nAssemble into one complete answer."
            )
            assembled = await client.generate(assemble_prompt, model=primary, temperature=0.2)
            steps.append(_trace("t2_assemble", assembled))
            current = assembled.text if assembled.ok else (deliverables[0] if deliverables else "")

            # Critic review
            critic_resp = await client.generate(
                CRITIC_PROMPT.format(prompt=prompt, work=current),
                model=critic_model, temperature=0.1,
            )
            steps.append(_trace("t2_critic", critic_resp))
            critique = _parse_json_safe(critic_resp.text) if critic_resp.ok else {}

            if critique.get("issues") and not critique.get("approved", True):
                fix_prompt = (
                    f"TASK: {prompt}\n\nANSWER:\n{current}\n\n"
                    f"ISSUES:\n" + "\n".join(f"- {i}" for i in critique["issues"])
                    + "\n\nFix these issues."
                )
                fixed = await client.generate(fix_prompt, model=primary, temperature=0.2)
                steps.append(_trace("t2_fix", fixed))
                if fixed.ok:
                    current = fixed.text

            # Verification
            verify_resp = await client.generate(
                VERIFIER_PROMPT.format(prompt=prompt, answer=current),
                model=verifier_model, temperature=0.1,
            )
            steps.append(_trace("t2_verify", verify_resp))
            verification = _parse_json_safe(verify_resp.text) if verify_resp.ok else {}

            if verification.get("all_pass", True) or tier == "T2":
                return _ok(self._ID, current, tuple(steps))
            # T2 failed verification → escalate to T3

        # ── T3: Multi-clique deliberation ──
        await _notify(progress_callback, "T3: multi-clique deliberation...")

        # Three independent cliques generate answers with different approaches
        clique_prompts = [
            f"TASK: {prompt}\n\nApproach this analytically. Show your reasoning step by step.",
            f"TASK: {prompt}\n\nApproach this practically. Focus on completeness and accuracy.",
            f"TASK: {prompt}\n\nApproach this critically. Challenge assumptions and verify claims.",
        ]
        clique_coros = [
            client.generate(cp, model=diverse[i % len(diverse)], temperature=0.3)
            for i, cp in enumerate(clique_prompts)
        ]
        clique_results = await asyncio.gather(*clique_coros)
        clique_texts = []
        for i, resp in enumerate(clique_results):
            steps.append(_trace(f"t3_clique_{i}", resp))
            if resp.ok:
                clique_texts.append(resp.text)

        if not clique_texts:
            return _fail(self._ID, "All T3 cliques failed", tuple(steps))

        # Cross-clique critique (each clique reviews the others)
        await _notify(progress_callback, "T3: cross-clique review...")
        critique_coros = []
        for i, text in enumerate(clique_texts):
            others = "\n\n".join(
                f"[Clique {j}]: {t}" for j, t in enumerate(clique_texts) if j != i
            )
            critique_coros.append(client.generate(
                f"TASK: {prompt}\n\nYOUR ANSWER:\n{text}\n\n"
                f"OTHER CLIQUES' ANSWERS:\n{others}\n\n"
                f"Improve your answer based on the strongest points from other cliques. "
                f"Fix any errors they caught.",
                model=diverse[i % len(diverse)], temperature=0.2,
            ))
        critique_results = await asyncio.gather(*critique_coros)
        revised_texts = []
        for i, resp in enumerate(critique_results):
            steps.append(_trace(f"t3_revise_{i}", resp))
            revised_texts.append(resp.text if resp.ok else clique_texts[i])

        # Final synthesis by strongest model
        await _notify(progress_callback, "T3: executive synthesis...")
        synth_prompt = (
            f"TASK: {prompt}\n\n"
            + "\n\n".join(f"[Perspective {i}]:\n{t}" for i, t in enumerate(revised_texts))
            + "\n\nAs the executive synthesiser, produce the definitive final answer. "
            f"Take the strongest elements from each perspective."
        )
        final = await client.generate(synth_prompt, model=primary, temperature=0.2)
        steps.append(_trace("t3_synthesise", final))

        # Final verification
        verify_resp = await client.generate(
            VERIFIER_PROMPT.format(prompt=prompt, answer=final.text if final.ok else revised_texts[0]),
            model=verifier_model, temperature=0.1,
        )
        steps.append(_trace("t3_verify", verify_resp))

        result_text = final.text if final.ok else max(revised_texts, key=len)
        return _ok(self._ID, result_text, tuple(steps))
