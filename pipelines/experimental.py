"""Experimental tier-5 pipelines for Occursus-Claude."""

from __future__ import annotations

import asyncio
import json
import re

from models import AppConfig, LLMResponse, PipelineResult, PipelineSpec, StepTrace
from llm_client import LLMClient
from pipelines.base import BasePipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _parse_sub_questions(text: str) -> list[str]:
    """Extract sub-questions from JSON or fall back to numbered-line parsing."""
    json_match = re.search(
        r"\{[^}]*\"sub_questions\"\s*:\s*\[[^\]]*\][^}]*\}", text, re.DOTALL,
    )
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            questions = parsed.get("sub_questions", [])
            if isinstance(questions, list) and all(
                isinstance(q, str) for q in questions
            ):
                return [q.strip() for q in questions if q.strip()]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    lines = text.strip().splitlines()
    questions: list[str] = []
    for line in lines:
        stripped = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        if stripped and len(stripped) > 10:
            questions.append(stripped)
    return questions


def _parse_choice(text: str) -> int | None:
    """Extract a 1 or 2 from judge response."""
    match = re.search(r"\b([12])\b", text.strip())
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Persona definitions for the Council
# ---------------------------------------------------------------------------

_COUNCIL_PERSONAS: tuple[tuple[str, str, float], ...] = (
    (
        "Scientist",
        "You approach every question with rigorous empirical analysis. "
        "Cite evidence, demand data, and reason from observable facts.",
        0.3,
    ),
    (
        "Philosopher",
        "You reason from first principles with conceptual clarity. "
        "Examine underlying assumptions and logical structure.",
        0.5,
    ),
    (
        "Child",
        "You explain things simply and ask naive but insightful questions. "
        "Cut through complexity to the essential truth.",
        0.7,
    ),
    (
        "Devil's Advocate",
        "You challenge everything and find counterarguments. "
        "Assume the opposite position and stress-test every claim.",
        0.6,
    ),
    (
        "Poet",
        "You find beauty in ideas and use metaphor and narrative. "
        "Express insights through vivid, memorable language.",
        0.9,
    ),
    (
        "Domain Expert",
        "You provide deep technical precision. "
        "Focus on accuracy, terminology, and domain-specific nuance.",
        0.2,
    ),
    (
        "Sceptic",
        "You demand evidence and question assumptions. "
        "Nothing is taken at face value without proof.",
        0.4,
    ),
)


# ---------------------------------------------------------------------------
# 1. Persona Council
# ---------------------------------------------------------------------------

class PersonaCouncilPipeline(BasePipeline):
    """7 radically different personas respond, then a chair synthesizes."""

    _ID = "persona_council"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID,
            name="Persona Council",
            description=(
                "7 personas with radically different perspectives answer "
                "in parallel, then a council chair synthesizes one definitive answer."
            ),
            tier=5,
            estimated_calls=8,
            tags=("experimental", "persona", "council"),
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
        synth_model = self._role_model(config, model, "synthesizer", "generator")
        task = self.wrap_task(prompt)
        steps: list[StepTrace] = []

        # --- Phase 1: 7 persona responses in parallel ---------------------
        await self._notify(
            progress_callback, "Convening council of 7 personas...",
        )

        persona_coros = tuple(
            client.generate(
                task,
                model=gen_model,
                temperature=temperature,
                system_prompt=system_prompt,
            )
            for _, system_prompt, temperature in _COUNCIL_PERSONAS
        )
        persona_results: tuple[LLMResponse, ...] = await asyncio.gather(
            *persona_coros,
        )

        persona_texts: list[tuple[str, str]] = []
        for (name, _, _), resp in zip(_COUNCIL_PERSONAS, persona_results):
            steps.append(_trace(f"persona_{name.lower()}", resp))
            if resp.ok:
                persona_texts.append((name, resp.text))

        if not persona_texts:
            return _fail(self._ID, "All 7 persona calls failed", tuple(steps))

        # --- Phase 2: Council Chair synthesis -----------------------------
        await self._notify(
            progress_callback, "Council Chair synthesizing insights...",
        )

        council_sections = "\n\n".join(
            f"[{name}]:\n{text}" for name, text in persona_texts
        )
        chair_prompt = (
            "A council of 7 experts with radically different perspectives "
            "has answered this task. Synthesize their insights into one "
            "definitive answer that captures the strongest elements from "
            f"each perspective.\n\n{council_sections}"
        )

        chair_resp = await client.generate(chair_prompt, model=synth_model)
        steps.append(_trace("council_chair", chair_resp))

        if not chair_resp.ok:
            longest = max(persona_texts, key=lambda pt: len(pt[1]))
            return _ok(self._ID, longest[1], tuple(steps))

        return _ok(self._ID, chair_resp.text, tuple(steps))


# ---------------------------------------------------------------------------
# 2. Adversarial Decomposition
# ---------------------------------------------------------------------------

class AdversarialDecompositionPipeline(BasePipeline):
    """Decompose into sub-questions, answer each, then synthesize."""

    _ID = "adversarial_decomposition"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID,
            name="Adversarial Decomposition",
            description=(
                "Decomposes the task into independent sub-questions, "
                "answers each in parallel, then synthesizes a unified answer."
            ),
            tier=5,
            estimated_calls=5,
            tags=("experimental", "decomposition", "adversarial"),
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
        decompose_model = self._role_model(config, model, "critic", "reviewer", "generator")
        answer_model = self._role_model(config, model, "generator")
        synth_model = self._role_model(config, model, "synthesizer", "generator")
        steps: list[StepTrace] = []

        # --- Phase 1: Decompose into sub-questions ------------------------
        await self._notify(progress_callback, "Decomposing task into sub-questions...")

        decompose_prompt = (
            "Break this task into 3-5 independent sub-questions that "
            "together fully address the original task. Reply with JSON: "
            '{"sub_questions": ["q1", "q2", ...]}\n\n'
            f"TASK: {prompt}"
        )
        decompose_resp = await client.generate(
            decompose_prompt, model=decompose_model, temperature=0.3,
        )
        steps.append(_trace("decompose", decompose_resp))

        if not decompose_resp.ok:
            return _fail(
                self._ID,
                f"Decomposition failed: {decompose_resp.error}",
                tuple(steps),
            )

        sub_questions = _parse_sub_questions(decompose_resp.text)
        if not sub_questions:
            return _fail(
                self._ID,
                "Failed to parse sub-questions from decomposition",
                tuple(steps),
            )

        # --- Phase 2: Answer each sub-question in parallel ----------------
        await self._notify(
            progress_callback,
            f"Answering {len(sub_questions)} sub-questions in parallel...",
        )

        sub_coros = tuple(
            client.generate(self.wrap_task(sq), model=answer_model)
            for sq in sub_questions
        )
        sub_results: tuple[LLMResponse, ...] = await asyncio.gather(*sub_coros)

        sub_answers: list[tuple[str, str]] = []
        for idx, (sq, resp) in enumerate(zip(sub_questions, sub_results)):
            steps.append(_trace(f"sub_answer_{idx + 1}", resp))
            if resp.ok:
                sub_answers.append((sq, resp.text))

        if not sub_answers:
            return _fail(
                self._ID,
                "All sub-question answers failed",
                tuple(steps),
            )

        # --- Phase 3: Synthesize into a unified answer --------------------
        await self._notify(progress_callback, "Synthesizing sub-answers...")

        qa_sections = "\n\n".join(
            f"Sub-question: {sq}\nAnswer: {ans}"
            for sq, ans in sub_answers
        )
        synth_prompt = (
            "Here are answers to sub-parts of the original task. "
            "Synthesize them into one complete, coherent answer.\n\n"
            f"Original task: {prompt}\n\n{qa_sections}"
        )
        synth_resp = await client.generate(synth_prompt, model=synth_model)
        steps.append(_trace("synthesize", synth_resp))

        if not synth_resp.ok:
            combined = "\n\n".join(ans for _, ans in sub_answers)
            return _ok(self._ID, combined, tuple(steps))

        return _ok(self._ID, synth_resp.text, tuple(steps))


# ---------------------------------------------------------------------------
# 3. Reverse Engineer
# ---------------------------------------------------------------------------

class ReverseEngineerPipeline(BasePipeline):
    """Answer, reverse-engineer the question, compare for gaps, then revise."""

    _ID = "reverse_engineer"

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID,
            name="Reverse Engineer",
            description=(
                "Generates an answer, reverse-engineers what question it answers, "
                "identifies gaps, then revises to address them."
            ),
            tier=5,
            estimated_calls=4,
            tags=("experimental", "reverse-engineer", "gap-analysis"),
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
        task = self.wrap_task(prompt)
        steps: list[StepTrace] = []

        # --- Phase 1: Generate initial answer -----------------------------
        await self._notify(progress_callback, "Generating initial answer...")

        initial = await client.generate(task, model=gen_model)
        steps.append(_trace("initial_answer", initial))

        if not initial.ok:
            return _fail(
                self._ID,
                f"Initial answer failed: {initial.error}",
                tuple(steps),
            )

        # --- Phase 2: Reverse engineer the question -----------------------
        await self._notify(
            progress_callback,
            "Reverse-engineering what question this answer addresses...",
        )

        reverse_prompt = (
            "Given ONLY this answer (not the original question), what question "
            "or task would produce this answer? Infer the most likely original "
            f"question:\n\n{initial.text}"
        )
        reverse_resp = await client.generate(reverse_prompt, model=review_model)
        steps.append(_trace("reverse_engineer", reverse_resp))

        if not reverse_resp.ok:
            return _ok(self._ID, initial.text, tuple(steps))

        # --- Phase 3: Compare and identify gaps ---------------------------
        await self._notify(progress_callback, "Comparing for gaps...")

        compare_prompt = (
            f"Original task: {prompt}\n\n"
            f"Inferred task from the answer: {reverse_resp.text}\n\n"
            "Identify any gaps -- what aspects of the original task are "
            "NOT addressed by the answer?"
        )
        compare_resp = await client.generate(compare_prompt, model=review_model)
        steps.append(_trace("gap_analysis", compare_resp))

        if not compare_resp.ok:
            return _ok(self._ID, initial.text, tuple(steps))

        # --- Phase 4: Revise to fill gaps ---------------------------------
        await self._notify(progress_callback, "Revising to address all gaps...")

        revise_prompt = (
            "Revise this answer to address ALL identified gaps.\n\n"
            f"Original task: {prompt}\n\n"
            f"Current answer: {initial.text}\n\n"
            f"Gaps identified: {compare_resp.text}"
        )
        revised = await client.generate(revise_prompt, model=review_model)
        steps.append(_trace("revise", revised))

        if not revised.ok:
            return _ok(self._ID, initial.text, tuple(steps))

        return _ok(self._ID, revised.text, tuple(steps))


# ---------------------------------------------------------------------------
# 4. Tournament
# ---------------------------------------------------------------------------

class TournamentPipeline(BasePipeline):
    """8 responses compete in a single-elimination tournament, winner is polished."""

    _ID = "tournament"
    _TEMPS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            id=self._ID,
            name="Tournament",
            description=(
                "8 responses at varied temperatures compete in a "
                "single-elimination tournament (QF/SF/F), winner is polished."
            ),
            tier=5,
            estimated_calls=12,
            tags=("experimental", "tournament", "elimination"),
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
        polish_model = self._role_model(config, model, "synthesizer", "reviewer", "generator")
        task = self.wrap_task(prompt)
        steps: list[StepTrace] = []

        # --- Phase 1: Generate 8 responses in parallel --------------------
        await self._notify(
            progress_callback, "Generating 8 tournament contenders...",
        )

        gen_coros = tuple(
            client.generate(task, model=gen_model, temperature=temp)
            for temp in self._TEMPS
        )
        gen_results: tuple[LLMResponse, ...] = await asyncio.gather(*gen_coros)

        contenders: list[str] = []
        for idx, resp in enumerate(gen_results):
            steps.append(_trace(f"generate_t{self._TEMPS[idx]}", resp))
            contenders.append(resp.text if resp.ok else "")

        valid_contenders = [c for c in contenders if c]
        if not valid_contenders:
            return _fail(self._ID, "All 8 generation calls failed", tuple(steps))

        if len(valid_contenders) == 1:
            return _ok(self._ID, valid_contenders[0], tuple(steps))

        # Pad to even count if needed (duplicate last valid entry)
        while len(valid_contenders) % 2 != 0:
            valid_contenders.append(valid_contenders[-1])

        # --- Phase 2: Quarter-finals --------------------------------------
        await self._notify(progress_callback, "Quarter-finals: 4 matchups...")

        qf_winners, qf_steps = await self._run_round(
            valid_contenders, "qf", client, judge_model,
        )
        steps.extend(qf_steps)

        if len(qf_winners) < 2:
            return _ok(self._ID, qf_winners[0], tuple(steps))

        # Pad to even if needed
        while len(qf_winners) % 2 != 0:
            qf_winners.append(qf_winners[-1])

        # --- Phase 3: Semi-finals -----------------------------------------
        await self._notify(progress_callback, "Semi-finals: 2 matchups...")

        sf_winners, sf_steps = await self._run_round(
            qf_winners, "sf", client, judge_model,
        )
        steps.extend(sf_steps)

        if len(sf_winners) < 2:
            return _ok(self._ID, sf_winners[0], tuple(steps))

        # --- Phase 4: Final -----------------------------------------------
        await self._notify(progress_callback, "Final matchup...")

        final_winners, final_steps = await self._run_round(
            sf_winners[:2], "final", client, judge_model,
        )
        steps.extend(final_steps)

        champion = final_winners[0] if final_winners else sf_winners[0]

        # --- Phase 5: Polish the champion ---------------------------------
        await self._notify(progress_callback, "Polishing tournament champion...")

        polish_prompt = (
            "Here is a strong response selected through competitive evaluation. "
            "Polish it for clarity, completeness, and accuracy without changing "
            f"its core content:\n\n{champion}"
        )
        polished = await client.generate(polish_prompt, model=polish_model)
        steps.append(_trace("polish", polished))

        if not polished.ok:
            return _ok(self._ID, champion, tuple(steps))

        return _ok(self._ID, polished.text, tuple(steps))

    async def _run_round(
        self,
        contenders: list[str],
        round_name: str,
        client: LLMClient,
        model: str,
    ) -> tuple[list[str], list[StepTrace]]:
        """Run one tournament round: pair up contenders, judge each pair."""
        pairs = [
            (contenders[i], contenders[i + 1])
            for i in range(0, len(contenders) - 1, 2)
        ]

        judge_coros = tuple(
            client.generate(
                "Compare these two responses to the same task. "
                "Which is better? Reply with ONLY: 1 or 2\n\n"
                f"Response 1:\n{a}\n\n"
                f"Response 2:\n{b}",
                model=model,
                temperature=0.1,
            )
            for a, b in pairs
        )
        judge_results: tuple[LLMResponse, ...] = await asyncio.gather(
            *judge_coros,
        )

        winners: list[str] = []
        round_steps: list[StepTrace] = []
        for idx, ((a, b), resp) in enumerate(zip(pairs, judge_results)):
            round_steps.append(
                _trace(f"{round_name}_match_{idx + 1}", resp),
            )
            if resp.ok:
                choice = _parse_choice(resp.text)
                if choice == 2:
                    winners.append(b)
                else:
                    winners.append(a)
            else:
                winners.append(max(a, b, key=len))

        return winners, round_steps
