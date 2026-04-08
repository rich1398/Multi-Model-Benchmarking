"""Expert Routing + Constraint Checker pipelines for Occursus-Claude."""

from __future__ import annotations

import asyncio
import json
import re

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


TRIAGE_PROMPT = (
    "Analyze this task and decompose it into 2-4 specialist sub-tasks. "
    "For each sub-task, specify which type of expert should handle it.\n"
    "Expert types available: code (programming/debugging), analysis (reasoning/math/logic), "
    "writing (creative/persuasive text), domain (science/law/medicine/finance).\n\n"
    "Reply with ONLY valid JSON:\n"
    '  {{"subtasks": [{{"expert": "code|analysis|writing|domain", "task": "what to do"}}]}}\n\n'
    "TASK: {prompt}"
)

ASSEMBLY_PROMPT = (
    "You are assembling a final answer from specialist contributions. "
    "Merge these expert outputs into one coherent, complete response. "
    "Resolve any contradictions by favoring the domain expert. "
    "Do not mention that multiple experts contributed.\n\n"
    "ORIGINAL TASK: {prompt}\n\n"
    "{expert_outputs}\n\n"
    "Write the complete, unified answer:"
)

EXPERT_PROMPTS = {
    "code": "You are a senior software engineer. Focus on correct, production-quality code. {task}",
    "analysis": "You are a quantitative analyst. Focus on precise reasoning, calculations, and logic. {task}",
    "writing": "You are a skilled writer. Focus on clarity, persuasion, and engaging prose. {task}",
    "domain": "You are a domain expert. Focus on accuracy, depth, and nuance. {task}",
}


class ExpertRoutingPipeline(BasePipeline):
    """Triage task into specialist sub-tasks, route to experts, assemble."""

    _ID = "expert_routing"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID,
            name="Expert Routing",
            description="Decomposes task into specialist sub-tasks, routes each to an expert model, assembles the results.",
            tier=3,
            estimated_calls=5,
            tags=("routing", "specialist", "decomposition"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        router_model = self._role_model(config, model, "generator")
        expert_model = self._role_model(config, model, "generator_alt", "generator")
        assembler_model = self._role_model(config, model, "synthesizer", "generator")
        steps: list[StepTrace] = []

        await self._notify(progress_callback, "Triaging task into sub-tasks...")
        triage_resp = await client.generate(
            TRIAGE_PROMPT.format(prompt=prompt),
            model=router_model, temperature=0.1,
        )
        steps.append(_trace("triage", triage_resp))
        if not triage_resp.ok:
            return _fail(self._ID, f"Triage failed: {triage_resp.error}", tuple(steps))

        subtasks = _parse_subtasks(triage_resp.text)
        if not subtasks:
            subtasks = [{"expert": "analysis", "task": prompt}]

        await self._notify(progress_callback, f"Routing to {len(subtasks)} specialist experts...")
        expert_coros = []
        for st in subtasks[:4]:
            expert_type = st.get("expert", "analysis")
            task_text = st.get("task", prompt)
            expert_prompt = EXPERT_PROMPTS.get(
                expert_type, EXPERT_PROMPTS["analysis"]
            ).format(task=task_text)
            expert_coros.append(client.generate(
                expert_prompt, model=expert_model, temperature=0.3,
            ))

        expert_results: list[LLMResponse] = list(await asyncio.gather(*expert_coros))
        for i, resp in enumerate(expert_results):
            steps.append(_trace(f"expert_{subtasks[i].get('expert', 'unknown')}", resp))

        expert_outputs = "\n\n".join(
            f"[{subtasks[i].get('expert', 'expert').upper()} SPECIALIST]:\n{r.text}"
            for i, r in enumerate(expert_results)
            if r.ok and r.text
        )
        if not expert_outputs:
            return _fail(self._ID, "All expert calls failed", tuple(steps))

        await self._notify(progress_callback, "Assembling final answer...")
        assembly_resp = await client.generate(
            ASSEMBLY_PROMPT.format(prompt=prompt, expert_outputs=expert_outputs),
            model=assembler_model, temperature=0.25,
        )
        steps.append(_trace("assemble", assembly_resp))

        final = assembly_resp.text if assembly_resp.ok else expert_outputs
        return _ok(self._ID, final, tuple(steps))


CONSTRAINT_CHECK_PROMPT = (
    "You are a strict constraint validator. Check this response against the task requirements.\n\n"
    "ORIGINAL TASK: {prompt}\n\n"
    "RESPONSE TO CHECK:\n{response}\n\n"
    "List EVERY constraint violation you find. Be extremely literal and precise.\n"
    "For each violation, state: the constraint, what was expected, and what was actually found.\n"
    "If the response passes all constraints, say 'ALL CONSTRAINTS SATISFIED'.\n\n"
    "Reply with ONLY valid JSON:\n"
    '  {{"violations": [{{"constraint": "...", "expected": "...", "found": "..."}}], "all_passed": true|false}}'
)

CONSTRAINT_FIX_PROMPT = (
    "Revise this response to fix ALL identified constraint violations. "
    "Do NOT change content that already satisfies constraints. "
    "Be surgical — fix only what's broken.\n\n"
    "ORIGINAL TASK: {prompt}\n\n"
    "CURRENT RESPONSE:\n{response}\n\n"
    "CONSTRAINT VIOLATIONS FOUND:\n{violations}\n\n"
    "Write the corrected response:"
)


class ConstraintCheckerPipeline(BasePipeline):
    """Generate, then validate constraints, then fix violations iteratively."""

    _ID = "constraint_checker"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID,
            name="Constraint Checker",
            description="Generates response, validates against constraints, fixes violations in a loop (up to 2 cycles).",
            tier=3,
            estimated_calls=5,
            tags=("constraints", "validation", "iterative"),
        )

    async def execute(
        self, prompt: str, client: LLMClient, config: AppConfig,
        *, model: str | None = None, progress_callback=None,
    ) -> PipelineResult:
        gen_model = self._role_model(config, model, "generator")
        checker_model = self._role_model(config, model, "critic", "generator")
        steps: list[StepTrace] = []

        await self._notify(progress_callback, "Generating initial response...")
        gen_resp = await client.generate(self.wrap_task(prompt), model=gen_model)
        steps.append(_trace("generate", gen_resp))
        if not gen_resp.ok:
            return _fail(self._ID, f"Generation failed: {gen_resp.error}", tuple(steps))

        current_text = gen_resp.text
        max_cycles = 2

        for cycle in range(max_cycles):
            await self._notify(progress_callback, f"Checking constraints (cycle {cycle + 1})...")
            check_resp = await client.generate(
                CONSTRAINT_CHECK_PROMPT.format(prompt=prompt, response=current_text),
                model=checker_model, temperature=0.1,
            )
            steps.append(_trace(f"check_{cycle + 1}", check_resp))
            if not check_resp.ok:
                break

            violations = _parse_violations(check_resp.text)
            if not violations:
                break

            await self._notify(progress_callback, f"Fixing {len(violations)} violations (cycle {cycle + 1})...")
            fix_resp = await client.generate(
                CONSTRAINT_FIX_PROMPT.format(
                    prompt=prompt,
                    response=current_text,
                    violations="\n".join(
                        f"- {v.get('constraint', '?')}: expected {v.get('expected', '?')}, found {v.get('found', '?')}"
                        for v in violations
                    ),
                ),
                model=gen_model, temperature=0.2,
            )
            steps.append(_trace(f"fix_{cycle + 1}", fix_resp))
            if fix_resp.ok and fix_resp.text:
                current_text = fix_resp.text

        return _ok(self._ID, current_text, tuple(steps))


def _parse_subtasks(text: str) -> list[dict[str, str]]:
    json_match = re.search(r"\{[^}]*\"subtasks\"\s*:\s*\[.*?\]\s*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            subtasks = parsed.get("subtasks", [])
            if isinstance(subtasks, list):
                return [
                    {"expert": st.get("expert", "analysis"), "task": st.get("task", "")}
                    for st in subtasks
                    if isinstance(st, dict) and st.get("task")
                ]
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _parse_violations(text: str) -> list[dict[str, str]]:
    if "ALL CONSTRAINTS SATISFIED" in text.upper():
        return []
    json_match = re.search(r"\{.*\"violations\"\s*:\s*\[.*?\].*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if parsed.get("all_passed", False):
                return []
            violations = parsed.get("violations", [])
            if isinstance(violations, list):
                return [v for v in violations if isinstance(v, dict)]
        except (json.JSONDecodeError, TypeError):
            pass
    return []
