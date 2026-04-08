"""Configuration loader for Occursus-Claude."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from models import AppConfig, JudgeConfig, ModelChoice, Persona, ProviderConfig, SavedSettings

SETTINGS_FILE = Path("config/settings.local.json")


def load_config(config_path: str = "config/occursus_claude.yaml") -> AppConfig:
    load_dotenv()

    path = Path(config_path)
    if not path.exists():
        return _defaults()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return _parse_config(raw)


def _parse_config(raw: dict) -> AppConfig:
    saved = load_saved_settings()
    models_list: list[ProviderConfig] = []

    models_section = raw.get("models", {})
    for provider_name, entries in models_section.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            api_key = _resolve_api_key(provider_name, saved)
            base_url = entry.get("base_url", _default_base_url(provider_name, saved))
            models_list.append(ProviderConfig(
                name=entry.get("name", ""),
                base_url=base_url,
                api_key=api_key,
                provider=provider_name,
            ))

    judge_raw = raw.get("judge", {})
    judge = JudgeConfig(
        backend=judge_raw.get("backend", "ollama"),
        model=judge_raw.get("model", "llama3.2"),
        temperature=float(judge_raw.get("temperature", 0.1)),
    )

    defaults = raw.get("defaults", {})
    personas = _parse_personas(raw.get("personas", {}))

    ollama_base = saved.ollama_base_url or os.environ.get(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )

    return AppConfig(
        models=tuple(models_list),
        judge=judge,
        default_model=defaults.get("default_model", "llama3.2"),
        default_provider=defaults.get("default_provider", "ollama"),
        ollama_base_url=ollama_base,
        timeout_seconds=int(defaults.get("timeout_seconds", 180)),
        max_concurrent=int(defaults.get("max_concurrent", 3)),
        tasks_file=raw.get("tasks_file", "tasks/core_tasks.json"),
        results_dir=raw.get("results_dir", "results/"),
        personas=tuple(personas),
    )


def _parse_personas(raw: dict) -> list[Persona]:
    result = []
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        result.append(Persona(
            name=name,
            temperature=float(cfg.get("temperature", 0.5)),
            system_prompt=cfg.get("system_prompt", "").strip(),
        ))
    return result


def _resolve_api_key(provider: str, saved: SavedSettings | None = None) -> str:
    if saved:
        key_map = {
            "openai": saved.openai_api_key,
            "anthropic": saved.anthropic_api_key,
            "gemini": saved.gemini_api_key,
        }
        saved_key = key_map.get(provider, "")
        if saved_key:
            return saved_key

    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = env_map.get(provider, "")
    if env_var:
        return os.environ.get(env_var, "")
    return ""


def _default_base_url(provider: str, saved: SavedSettings | None = None) -> str:
    if saved:
        url_map = {
            "ollama": saved.ollama_base_url,
            "openai": saved.openai_base_url,
            "anthropic": saved.anthropic_base_url,
            "gemini": saved.gemini_base_url,
        }
        saved_url = url_map.get(provider, "")
        if saved_url:
            return saved_url

    defaults = {
        "ollama": "http://localhost:11434",
        "anthropic": "https://api.anthropic.com",
        "openai": "https://api.openai.com",
        "gemini": "https://generativelanguage.googleapis.com",
    }
    return defaults.get(provider, "")


def load_saved_settings() -> SavedSettings:
    if not SETTINGS_FILE.is_file():
        return SavedSettings()
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        gen_models = tuple(
            ModelChoice(
                provider=m.get("provider", "ollama"),
                model=m.get("model", ""),
                label=m.get("label", ""),
            )
            for m in raw.get("selected_generation_models", [])
        )
        return SavedSettings(
            ollama_base_url=raw.get("ollama_base_url", "http://localhost:11434"),
            openai_base_url=raw.get("openai_base_url", "https://api.openai.com"),
            anthropic_base_url=raw.get("anthropic_base_url", "https://api.anthropic.com"),
            gemini_base_url=raw.get("gemini_base_url", "https://generativelanguage.googleapis.com"),
            openai_api_key=raw.get("openai_api_key", ""),
            anthropic_api_key=raw.get("anthropic_api_key", ""),
            gemini_api_key=raw.get("gemini_api_key", ""),
            selected_generation_models=gen_models,
        )
    except Exception:
        return SavedSettings()


def save_saved_settings(settings: SavedSettings) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(settings)
    data["selected_generation_models"] = [
        asdict(m) for m in settings.selected_generation_models
    ]
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _defaults() -> AppConfig:
    return AppConfig(
        models=(ProviderConfig(
            name="llama3.2",
            base_url="http://localhost:11434",
            provider="ollama",
        ),),
        tasks_file="tasks/core_tasks.json",
    )
