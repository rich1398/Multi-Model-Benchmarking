"""LLM-as-judge scoring system for Occursus Benchmark."""

from __future__ import annotations

import asyncio
import json
import re

from models import AppConfig, JudgeResult
from llm_client import LLMClient


JUDGE_PROMPT = """You are a strict but fair judge evaluating an AI-generated answer.

TASK: {prompt}

RUBRIC: {rubric}

ANSWER TO EVALUATE:
{response}

Score this answer 0-100 against the rubric. Be discriminating and use the full range.
0-34 = seriously wrong, unsafe, or failed the task.
35-54 = weak or partially incorrect.
55-69 = mixed quality with important gaps.
70-84 = strong but clearly imperfect.
85-94 = excellent but with a noticeable weakness.
95-100 = exceptional and nearly flawless.

Reply with ONLY valid JSON: {{"score": N, "reasoning": "one-sentence justification"}}"""


async def judge_response(
    prompt: str,
    rubric: str,
    response: str,
    client: LLMClient,
    config: AppConfig,
) -> JudgeResult:
    if not response or not response.strip():
        return JudgeResult(
            score=0,
            reasoning="No response to evaluate (empty or missing).",
            ok=False,
            error="Empty response",
            backend=config.judge.backend,
            model=config.judge.model,
            parsed_ok=False,
        )

    judge_prompt = JUDGE_PROMPT.format(
        prompt=prompt,
        rubric=rubric,
        response=response,
    )

    resp = await client.generate(
        judge_prompt,
        model=config.judge.model,
        provider=config.judge.backend,
        temperature=config.judge.temperature,
    )

    if not resp.ok:
        return JudgeResult(
            score=0,
            reasoning=f"Judge failed: {resp.error}",
            ok=False,
            error=resp.error,
            backend=config.judge.backend,
            model=config.judge.model,
            parsed_ok=False,
        )

    return _parse_judge_response(
        resp.text,
        backend=config.judge.backend,
        model=config.judge.model,
    )


async def dual_judge_response(
    prompt: str,
    rubric: str,
    response: str,
    client: LLMClient,
    judge_models: list[dict[str, str]],
    temperature: float = 0.1,
) -> JudgeResult:
    """Call multiple judges concurrently (blind), average their scores."""
    if not response or not response.strip():
        jm = judge_models[0] if judge_models else {"provider": "unknown", "model": "unknown"}
        return JudgeResult(
            score=0,
            reasoning="No response to evaluate (empty or missing).",
            ok=False,
            error="Empty response",
            backend=jm["provider"],
            model=jm["model"],
            parsed_ok=False,
        )

    judge_prompt = JUDGE_PROMPT.format(prompt=prompt, rubric=rubric, response=response)

    coros = [
        client.generate(
            judge_prompt,
            model=jm["model"],
            provider=jm["provider"],
            temperature=temperature,
        )
        for jm in judge_models
    ]
    results = await asyncio.gather(*coros, return_exceptions=True)

    parsed: list[JudgeResult] = []
    for i, resp in enumerate(results):
        jm = judge_models[i]
        if isinstance(resp, Exception):
            parsed.append(JudgeResult(
                score=0, reasoning=f"Judge error: {resp}",
                ok=False, error=str(resp),
                backend=jm["provider"], model=jm["model"], parsed_ok=False,
            ))
            continue
        if not resp.ok:
            parsed.append(JudgeResult(
                score=0, reasoning=f"Judge failed: {resp.error}",
                ok=False, error=resp.error,
                backend=jm["provider"], model=jm["model"], parsed_ok=False,
            ))
            continue
        parsed.append(_parse_judge_response(resp.text, backend=jm["provider"], model=jm["model"]))

    ok_results = [r for r in parsed if r.ok and r.parsed_ok]
    if not ok_results:
        return parsed[0] if parsed else JudgeResult(
            score=0, reasoning="No judges available",
            ok=False, error="No judges available",
            backend="none", model="none", parsed_ok=False,
        )

    avg_score = round(sum(r.score for r in ok_results) / len(ok_results))
    combined_reasoning = " | ".join(
        f"[{r.backend}/{r.model}: {r.score}/100] {r.reasoning}" for r in ok_results
    )
    model_names = ", ".join(r.model for r in ok_results)
    backend_names = ", ".join(r.backend for r in ok_results)

    return JudgeResult(
        score=avg_score,
        reasoning=combined_reasoning,
        ok=True,
        backend=backend_names,
        model=model_names,
        parsed_ok=True,
    )


def _parse_judge_response(text: str, *, backend: str, model: str) -> JudgeResult:
    json_match = re.search(r"\{[^}]+\}", text)
    if not json_match:
        return JudgeResult(
            score=0,
            reasoning=f"Judge returned unparseable response: {text[:200]}",
            ok=False,
            error="Could not parse judge JSON",
            backend=backend,
            model=model,
            parsed_ok=False,
        )

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return JudgeResult(
            score=0,
            reasoning=f"Invalid JSON from judge: {text[:200]}",
            ok=False,
            error="Invalid JSON from judge",
            backend=backend,
            model=model,
            parsed_ok=False,
        )

    raw_score = data.get("score", 0)
    try:
        score = int(raw_score)
    except (ValueError, TypeError):
        score = 0

    score = max(0, min(100, score))

    reasoning = str(data.get("reasoning", data.get("brief", "")))

    if score == 0 and not reasoning:
        reasoning = "Judge returned score 0"

    return JudgeResult(
        score=score,
        reasoning=reasoning,
        ok=True,
        backend=backend,
        model=model,
        parsed_ok=True,
    )
