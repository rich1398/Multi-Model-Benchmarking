"""Regex-based task classifier for adaptive temperature."""

from __future__ import annotations

import re

_PATTERNS: dict[str, str] = {
    "factual": r"capital of|who (is|was)|when (was|did)|what is the|true that|is it true",
    "code": r"write.*(function|program|query|code)|refactor|SQL|Python|palindrome|implement|flask|fastapi",
    "analytical": r"compare|analy[sz]e|design.*system|explain.*difference|trade.?off|pros and cons|architecture|microservice|monolith",
    "constraint": r"constraint|must (satisfy|contain|include|be exactly)|word count|cannot use|exactly \d+|hard constraint|acrostic",
    "reasoning": r"calculat|prove|shortest|optimize|maximum area|probability|how many|EBITDA|fencing",
    "creative": r"write.*(story|paragraph|poem|essay|opening|debate)|creative|compelling|narrative|science fiction",
}

_TEMPS: dict[str, float] = {
    "factual": 0.1,
    "code": 0.15,
    "analytical": 0.3,
    "constraint": 0.2,
    "reasoning": 0.35,
    "creative": 0.7,
}


def classify_task(prompt: str) -> tuple[str, float]:
    """Return (category, recommended_temperature)."""
    lower = prompt.lower()
    for category, pattern in _PATTERNS.items():
        if re.search(pattern, lower):
            return category, _TEMPS[category]
    return "default", 0.4
