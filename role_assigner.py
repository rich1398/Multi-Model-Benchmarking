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

    generator_alt, _ = _pick_model(
        by_provider,
        ("openai", "gemini", "ollama", "anthropic"),
        exclude_models={primary_model},
    )
    if not generator_alt:
        generator_alt = primary_model

    # Reviewer is a high-frequency reliability-sensitive role in the deep pipelines.
    # Prefer a non-Gemini reviewer when a stable alternative is available, and do
    # not force Ollama into this slot just because many models are enabled.
    reviewer_model, _ = _pick_model(
        by_provider,
        ("openai", "anthropic", "ollama"),
        exclude_models={primary_model},
    )
    if not reviewer_model:
        reviewer_model = primary_model
    if not reviewer_model:
        reviewer_model, _ = _pick_model(by_provider, ("gemini",))

    critic_model, _ = _pick_model(
        by_provider,
        ("gemini", "openai", "anthropic", "ollama"),
        exclude_models={primary_model, reviewer_model},
    )
    if not critic_model:
        critic_model = generator_alt or reviewer_model or primary_model

    role_models: dict[str, str] = {
        "generator": primary_model,
        "generator_alt": generator_alt,
        "critic": critic_model,
        "synthesizer": primary_model,
        "reviewer": reviewer_model or primary_model,
        "arbiter": primary_model,
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
