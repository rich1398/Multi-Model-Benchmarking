"""Cost estimation for LLM API calls."""

from __future__ import annotations

# (input_$/1M_tokens, output_$/1M_tokens) — last updated 2025-06
PRICING: dict[str, dict[str, tuple[float, float]]] = {
    "anthropic": {
        "claude-sonnet-4-20250514": (3.0, 15.0),
        "claude-opus-4-1-20250805": (15.0, 75.0),
        "claude-haiku-3-5-20241022": (0.80, 4.0),
    },
    "openai": {
        "gpt-4o": (2.50, 10.0),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4.1": (2.0, 8.0),
        "gpt-4.1-mini": (0.40, 1.60),
        "o3-mini": (1.10, 4.40),
    },
    "gemini": {
        "gemini-2.5-flash": (0.15, 0.60),
        "gemini-2.5-pro": (1.25, 10.0),
        "gemini-2.0-flash": (0.10, 0.40),
    },
    "ollama": {},
}

# Subscription CLI calls are free
SUBSCRIPTION_PROVIDERS = {"claude_sub", "chatgpt_sub", "gemini_sub"}


def estimate_call_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Return estimated cost in USD. Returns 0 for Ollama/subscription."""
    if provider in SUBSCRIPTION_PROVIDERS:
        return 0.0
    model_pricing = PRICING.get(provider, {}).get(model)
    if not model_pricing:
        return 0.0
    input_rate, output_rate = model_pricing
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def estimate_run_cost(
    pipeline_count: int,
    task_count: int,
    avg_calls_per_pipeline: float,
    repeat_count: int,
    subscription_mode: bool,
) -> float:
    """Rough pre-run cost estimate in USD."""
    if subscription_mode:
        return 0.0
    total_cells = pipeline_count * task_count * repeat_count
    avg_tokens_per_call = 2000
    cost_per_call = 0.02  # rough average across providers
    return round(total_cells * avg_calls_per_pipeline * cost_per_call, 2)
