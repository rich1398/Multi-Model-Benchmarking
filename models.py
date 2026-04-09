"""Immutable data models for Occursus Benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMResponse:
    text: str = ""
    ok: bool = True
    error: str = ""
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""
    tokens_used: int = 0


@dataclass(frozen=True)
class StepTrace:
    phase: str = ""
    model: str = ""
    latency_ms: float = 0.0
    tokens: int = 0
    text_preview: str = ""


@dataclass(frozen=True)
class PipelineResult:
    pipeline_id: str = ""
    final_text: str = ""
    ok: bool = True
    error: str = ""
    steps: tuple[StepTrace, ...] = ()
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    llm_calls: int = 0


@dataclass(frozen=True)
class TaskDef:
    id: str = ""
    prompt: str = ""
    rubric: str = ""
    category: str = ""
    difficulty: str = "medium"


@dataclass(frozen=True)
class JudgeResult:
    score: int = 0
    reasoning: str = ""
    ok: bool = True
    error: str = ""
    backend: str = ""
    model: str = ""
    parsed_ok: bool = True


@dataclass(frozen=True)
class PipelineSpec:
    id: str = ""
    name: str = ""
    description: str = ""
    tier: int = 1
    estimated_calls: int = 1
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Persona:
    name: str = ""
    temperature: float = 0.5
    system_prompt: str = ""


@dataclass(frozen=True)
class ProviderConfig:
    name: str = ""
    base_url: str = ""
    api_key: str = ""
    provider: str = "ollama"


@dataclass(frozen=True)
class JudgeConfig:
    backend: str = "ollama"
    model: str = "llama3.2"
    temperature: float = 0.1


@dataclass(frozen=True)
class ModelChoice:
    provider: str = "ollama"
    model: str = ""
    label: str = ""


@dataclass(frozen=True)
class SavedSettings:
    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str = "https://api.openai.com"
    anthropic_base_url: str = "https://api.anthropic.com"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    selected_generation_models: tuple[ModelChoice, ...] = ()


@dataclass(frozen=True)
class AppConfig:
    models: tuple[ProviderConfig, ...] = ()
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    default_model: str = "llama3.2"
    default_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    timeout_seconds: int = 180
    max_concurrent: int = 3
    tasks_file: str = "tasks/tasks.json"
    results_dir: str = "results/"
    personas: tuple[Persona, ...] = ()
    role_models: dict[str, str] = field(default_factory=dict)
    # Enhancement toggles
    cot_enabled: bool = False
    token_budget_enabled: bool = False
    adaptive_temp_enabled: bool = False
    repeat_count: int = 1
    cost_tracking_enabled: bool = True
    subscription_mode: bool = False
    max_output_tokens: int = 4096
