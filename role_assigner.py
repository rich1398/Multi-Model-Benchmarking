"""Auto-assign models to pipeline roles based on the enabled model set."""

from __future__ import annotations

from llm_client import resolve_provider

_GENERATION_PRIORITY = ("anthropic", "openai", "gemini", "ollama")
_JUDGE_PROVIDERS = ("anthropic", "openai")


def _pick_model(
    by_provider: dict[str, list[str]],
    provider_order: tuple[str, ...],
    *,
    exclude_models: set[str] | None = None,
) -> tuple[str, str]:
    excluded = exclude_models or set()
    for prov in provider_order:
        for model in by_provider.get(prov, []):
            if model and model not in excluded:
                return model, prov
    return "", ""


def auto_assign_roles(
    enabled_models: list[dict[str, str]],
) -> dict[str, object]:
    """Given enabled models, return role_models dict, judge_models list, and primary model."""
    by_provider: dict[str, list[str]] = {}
    for entry in enabled_models:
        prov = entry.get("provider") or resolve_provider(entry.get("model", ""))
        model = entry.get("model", "")
        by_provider.setdefault(prov, []).append(model)

    primary_model, primary_provider = _pick_model(by_provider, _GENERATION_PRIORITY)
    if not primary_model and enabled_models:
        primary_model = enabled_models[0]["model"]
        primary_provider = enabled_models[0].get("provider", "ollama")

    # Build a list of distinct models for even distribution
    distinct: list[tuple[str, str]] = [(primary_model, primary_provider)]
    seen = {primary_model}
    for prov in ("openai", "gemini", "anthropic", "ollama"):
        for m in by_provider.get(prov, []):
            if m and m not in seen:
                distinct.append((m, prov))
                seen.add(m)

    model_a = distinct[0][0]  # primary (Claude)
    model_b = distinct[1][0] if len(distinct) > 1 else model_a  # secondary (GPT)
    model_c = distinct[2][0] if len(distinct) > 2 else model_b  # tertiary (Gemini)

    # Distribute roles evenly: 2 roles per provider when 3 cloud models available
    # generator + arbiter = model_a (strongest, quality-critical)
    # generator_alt + critic = model_b (different perspective, adversarial)
    # synthesizer + reviewer = model_c (independent synthesis and review)
    role_models: dict[str, str] = {
        "generator": model_a,
        "generator_alt": model_b,
        "critic": model_b,
        "synthesizer": model_c,
        "reviewer": model_c,
        "arbiter": model_a,
    }

    judge_models: list[dict[str, str]] = []
    for prov in _JUDGE_PROVIDERS:
        if prov in by_provider and by_provider[prov]:
            judge_models.append({"provider": prov, "model": by_provider[prov][0]})
    if not judge_models:
        judge_models.append({"provider": primary_provider, "model": primary_model})

    return {
        "role_models": role_models,
        "judge_models": judge_models,
        "primary_model": primary_model,
        "primary_provider": primary_provider,
    }
