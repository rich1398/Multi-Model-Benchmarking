"""Unified async LLM client for Ollama, Anthropic, OpenAI, Gemini, and subscription CLIs (Occursus Benchmark)."""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import httpx
import ollama as ollama_sdk

from models import AppConfig, LLMResponse

MAX_RETRIES = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}

# Subscription CLI infrastructure
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_BILLING_PATTERNS = ("balance is too low", "credit balance", "billing", "quota exceeded", "rate limit")


def _resolve_cli(name: str) -> str | None:
    """Resolve CLI executable. On Windows, .cmd/.bat files are valid."""
    path = shutil.which(name)
    if path and sys.platform == "win32" and path.lower().endswith((".cmd", ".bat")):
        exe = shutil.which(f"{name}.exe")
        if exe:
            return exe
    return path


def _build_cmd(cli_path: str, args: list[str]) -> tuple[str, bool]:
    """Build subprocess command. Returns (command_string_or_list, use_shell).

    On Windows, .cmd/.bat files must run with shell=True to avoid
    deadlocks in frozen PyInstaller executables.
    """
    if sys.platform == "win32" and cli_path.lower().endswith((".cmd", ".bat")):
        # Use shell=True with a quoted command string for .cmd files
        parts = [f'"{cli_path}"'] + [f'"{a}"' if " " in a else a for a in args]
        return " ".join(parts), True
    return [cli_path] + args, False


def _headless_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env.update({"CI": "true", "FORCE_COLOR": "0", "TERM": "dumb", "NO_UPDATE_NOTIFIER": "true"})
    return env


def _sub_timeout(prompt: str) -> int:
    """Adaptive timeout for subscription CLI calls based on prompt length."""
    base = 300  # 5 minutes minimum (thesis prompts need this)
    extra = len(prompt) // 500 * 30  # +30s per ~500 chars of prompt
    return min(base + extra, 900)  # cap at 15 minutes


def _kill_tree(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=5)
            return
        except Exception:
            pass
    proc.kill()


def resolve_provider(model: str) -> str:
    """Determine provider from model name. Returns 'ollama' for unrecognized models."""
    name = (model or "").strip().lower()
    if not name:
        return "ollama"
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if name.startswith("gemini"):
        return "gemini"
    return "ollama"
class LLMClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._ollama = ollama_sdk.AsyncClient(host=config.ollama_base_url)
        self._http = httpx.AsyncClient(timeout=config.timeout_seconds)
        self._global_semaphore = asyncio.Semaphore(max(1, config.max_concurrent))
        self._provider_semaphores: dict[str, asyncio.Semaphore] = {
            "ollama": asyncio.Semaphore(max(1, config.max_concurrent)),
            "openai": asyncio.Semaphore(max(1, config.max_concurrent)),
            "anthropic": asyncio.Semaphore(max(1, config.max_concurrent)),
            "gemini": asyncio.Semaphore(1),
        }
        # Subscription CLIs: 1 concurrent call per CLI, but all 3 CLIs can run in parallel
        self._sub_semaphores: dict[str, asyncio.Semaphore] = {
            "anthropic": asyncio.Semaphore(1),
            "openai": asyncio.Semaphore(1),
            "gemini": asyncio.Semaphore(1),
        }
        self.subscription_mode: bool = getattr(config, "subscription_mode", False)

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        model = model or self._config.default_model
        provider = provider or resolve_provider(model)

        # Subscription mode: redirect cloud providers to CLI
        # Each CLI gets its own semaphore (1 at a time per CLI)
        # but different CLIs run in parallel (no global semaphore)
        if self.subscription_mode and provider in ("anthropic", "openai", "gemini"):
            sub_sem = self._sub_semaphores.get(provider, asyncio.Semaphore(1))
            async with sub_sem:
                if provider == "anthropic":
                    return await self._call_claude_sub(prompt, model, temperature, system_prompt)
                elif provider == "openai":
                    return await self._call_chatgpt_sub(prompt, model, temperature, system_prompt)
                elif provider == "gemini":
                    return await self._call_gemini_sub(prompt, model, temperature, system_prompt)

        provider_sem = self._provider_semaphores.get(
            provider,
            asyncio.Semaphore(max(1, self._config.max_concurrent)),
        )
        async with self._global_semaphore:
            async with provider_sem:
                if provider == "ollama":
                    return await self._call_ollama(
                        prompt, model, temperature, system_prompt
                    )
                elif provider == "anthropic":
                    return await self._call_anthropic(
                        prompt, model, temperature, system_prompt, max_tokens
                    )
                elif provider == "openai":
                    return await self._call_openai(
                        prompt, model, temperature, system_prompt
                    )
                elif provider == "gemini":
                    return await self._call_gemini(
                        prompt, model, temperature, system_prompt, max_tokens
                    )
                else:
                    return LLMResponse(
                        ok=False,
                        error=f"Unknown provider: {provider}",
                        model=model,
                        provider=provider,
                    )

    def _looks_like_embedding_model(self, model: str, family: str = "") -> bool:
        name = (model or "").strip().lower()
        fam = (family or "").strip().lower()
        markers = (
            "embed",
            "embedding",
            "nomic-embed",
            "bge-",
            "e5-",
            "snowflake-arctic-embed",
        )
        family_markers = ("bert", "clip", "embedding")
        return any(marker in name for marker in markers) or any(
            marker in fam for marker in family_markers
        )

    def _matches_provider_family(self, provider: str, model: str) -> tuple[bool, str]:
        provider = (provider or "").strip().lower()
        name = (model or "").strip().lower()
        if not name:
            return False, "No model configured."

        local_markers = (
            "llama",
            "mistral",
            "gemma",
            "qwen",
            "nomic",
            "deepseek",
            "phi",
            "mixtral",
        )

        if provider == "anthropic":
            if not name.startswith("claude"):
                return False, (
                    f'Model "{model}" does not look like an Anthropic Claude model.'
                )
            return True, ""

        if provider == "openai":
            if name.startswith("claude"):
                return False, (
                    f'Model "{model}" looks like an Anthropic model, not an OpenAI model.'
                )
            if ":" in name or any(marker in name for marker in local_markers):
                return False, (
                    f'Model "{model}" looks like a local/Ollama model, not an OpenAI model.'
                )
            return True, ""

        if provider == "gemini":
            if name.startswith("claude") or name.startswith("gpt"):
                return False, (
                    f'Model "{model}" does not look like a Google Gemini model.'
                )
            return True, ""

        return True, ""

    @staticmethod
    def _retry_delay(
        attempt: int,
        *,
        response: httpx.Response | None = None,
        provider: str = "",
    ) -> float:
        retry_after = response.headers.get("Retry-After", "") if response is not None else ""
        if retry_after:
            try:
                return max(1.0, min(float(retry_after), 30.0))
            except ValueError:
                pass
        base_delay = 1.0 if provider != "gemini" else 2.0
        max_delay = 12.0 if provider != "gemini" else 30.0
        jitter = random.uniform(0.0, 0.5)
        return min(base_delay * (2 ** attempt) + jitter, max_delay)

    async def _post_with_retry(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        provider: str = "",
        max_retries: int | None = None,
    ) -> httpx.Response:
        attempts = max_retries or (5 if provider == "gemini" else MAX_RETRIES)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = await self._http.post(url, headers=headers, json=json)
                if resp.status_code not in RETRY_STATUSES:
                    resp.raise_for_status()
                    return resp
                if attempt < attempts - 1:
                    await asyncio.sleep(
                        self._retry_delay(attempt, response=resp, provider=provider)
                    )
                else:
                    resp.raise_for_status()
            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                httpx.ReadError,
            ) as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    await asyncio.sleep(
                        self._retry_delay(attempt, provider=provider)
                    )
                else:
                    raise
        raise last_exc  # type: ignore[misc]

    async def _call_ollama(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
        system_prompt: str | None,
    ) -> LLMResponse:
        start = time.perf_counter()
        try:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            options: dict[str, Any] = {}
            if temperature is not None:
                options["temperature"] = temperature

            resp = await self._ollama.chat(
                model=model,
                messages=messages,
                options=options if options else None,
            )

            latency = (time.perf_counter() - start) * 1000
            text = resp.get("message", {}).get("content", "").strip()
            tokens = resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0)

            if not text:
                return LLMResponse(
                    ok=False, error="Empty Ollama response",
                    latency_ms=latency, model=model, provider="ollama",
                )

            return LLMResponse(
                text=text, ok=True, latency_ms=latency,
                model=model, provider="ollama", tokens_used=tokens,
            )
        except ollama_sdk.ResponseError as e:
            latency = (time.perf_counter() - start) * 1000
            return LLMResponse(
                ok=False, error=f"Ollama error: {e}",
                latency_ms=latency, model=model, provider="ollama",
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return LLMResponse(
                ok=False, error=f"Ollama connection error: {e}",
                latency_ms=latency, model=model, provider="ollama",
            )

    async def _call_anthropic(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
        system_prompt: str | None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        api_key = self._find_api_key("anthropic")
        if not api_key:
            return LLMResponse(
                ok=False, error="ANTHROPIC_API_KEY not set",
                model=model, provider="anthropic",
            )

        try:
            body: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens or 4096,
                "messages": [{"role": "user", "content": prompt}],
            }
            if temperature is not None:
                body["temperature"] = temperature
            if system_prompt:
                body["system"] = system_prompt

            base_url = self._find_base_url("anthropic")
            resp = await self._post_with_retry(
                f"{base_url}/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
                provider="anthropic",
            )
            latency = (time.perf_counter() - start) * 1000
            data = resp.json()

            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            text = text.strip()

            tokens = data.get("usage", {})
            total_tokens = tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0)

            if not text:
                return LLMResponse(
                    ok=False, error="Empty Anthropic response",
                    latency_ms=latency, model=model, provider="anthropic",
                )

            return LLMResponse(
                text=text, ok=True, latency_ms=latency,
                model=model, provider="anthropic", tokens_used=total_tokens,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return LLMResponse(
                ok=False, error=f"Anthropic error: {e}",
                latency_ms=latency, model=model, provider="anthropic",
            )

    async def _call_openai(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
        system_prompt: str | None,
    ) -> LLMResponse:
        start = time.perf_counter()
        api_key = self._find_api_key("openai")
        if not api_key:
            return LLMResponse(
                ok=False, error="OPENAI_API_KEY not set",
                model=model, provider="openai",
            )

        try:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            body: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if temperature is not None:
                body["temperature"] = temperature

            base_url = self._find_base_url("openai")
            resp = await self._post_with_retry(
                f"{base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                provider="openai",
            )
            latency = (time.perf_counter() - start) * 1000
            data = resp.json()

            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content", "").strip()
            tokens = data.get("usage", {}).get("total_tokens", 0)

            if not text:
                return LLMResponse(
                    ok=False, error="Empty OpenAI response",
                    latency_ms=latency, model=model, provider="openai",
                )

            return LLMResponse(
                text=text, ok=True, latency_ms=latency,
                model=model, provider="openai", tokens_used=tokens,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return LLMResponse(
                ok=False, error=f"OpenAI error: {e}",
                latency_ms=latency, model=model, provider="openai",
            )

    async def _call_gemini(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
        system_prompt: str | None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        api_key = self._find_api_key("gemini")
        if not api_key:
            return LLMResponse(
                ok=False, error="GEMINI_API_KEY not set",
                model=model, provider="gemini",
            )

        try:
            user_text = prompt
            payload: dict[str, Any] = {
                "contents": [{"parts": [{"text": user_text}]}],
            }
            gen_config: dict[str, Any] = {}
            if temperature is not None:
                gen_config["temperature"] = temperature
            gen_config["maxOutputTokens"] = max_tokens or 4096
            payload["generationConfig"] = gen_config

            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

            base_url = self._find_base_url("gemini")
            normalized_model = model.split("/", 1)[1] if model.startswith("models/") else model
            resp = await self._post_with_retry(
                f"{base_url}/v1beta/models/{normalized_model}:generateContent",
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                provider="gemini",
                max_retries=5,
            )
            latency = (time.perf_counter() - start) * 1000
            data = resp.json()

            candidates = data.get("candidates") or []
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        text += part["text"]
            text = text.strip()

            usage = data.get("usageMetadata") or {}
            input_tokens = int(usage.get("promptTokenCount") or 0)
            output_tokens = int(usage.get("candidatesTokenCount") or 0)
            total_tokens = input_tokens + output_tokens

            if not text:
                return LLMResponse(
                    ok=False, error="Empty Gemini response",
                    latency_ms=latency, model=normalized_model, provider="gemini",
                )

            return LLMResponse(
                text=text, ok=True, latency_ms=latency,
                model=normalized_model, provider="gemini", tokens_used=total_tokens,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return LLMResponse(
                ok=False, error=f"Gemini error: {e}",
                latency_ms=latency, model=model, provider="gemini",
            )

    def _find_api_key(self, provider: str) -> str:
        for m in self._config.models:
            if m.provider == provider and m.api_key:
                return m.api_key
        return ""

    def _find_base_url(self, provider: str) -> str:
        for m in self._config.models:
            if m.provider == provider and m.base_url:
                return m.base_url.rstrip("/")
        defaults = {
            "openai": "https://api.openai.com",
            "anthropic": "https://api.anthropic.com",
            "gemini": "https://generativelanguage.googleapis.com",
        }
        return defaults.get(provider, "")

    async def list_ollama_model_details(self) -> list[dict[str, Any]]:
        try:
            resp = await self._ollama.list()
            models_list = getattr(resp, "models", None) or resp.get("models", [])
            result: list[dict[str, Any]] = []
            for m in models_list:
                name = getattr(m, "model", "") or (
                    m.get("model", "") if hasattr(m, "get") else ""
                )
                if not name:
                    continue

                details = getattr(m, "details", None)
                family = getattr(details, "family", "") if details else ""
                parameter_size = getattr(details, "parameter_size", "") if details else ""
                embedding = self._looks_like_embedding_model(name, family)
                result.append(
                    {
                        "name": name,
                        "family": family,
                        "parameter_size": parameter_size,
                        "chat_capable": not embedding,
                        "reason": (
                            "Embedding models cannot be used for chat benchmarks."
                            if embedding else ""
                        ),
                    }
                )
            return result
        except Exception:
            return []

    async def list_ollama_models(self) -> list[str]:
        models = await self.list_ollama_model_details()
        return [m["name"] for m in models if m.get("chat_capable")]

    async def test_provider(
        self, provider: str, *, api_key: str = "", base_url: str = ""
    ) -> dict[str, Any]:
        if provider == "ollama":
            try:
                models = await self.list_ollama_models()
                return {
                    "ok": True,
                    "provider": provider,
                    "models": models[:25],
                    "message": f"Ollama reachable with {len(models)} models.",
                }
            except Exception as e:
                return {"ok": False, "provider": provider, "message": str(e)}

        if provider == "openai":
            key = api_key or self._find_api_key("openai")
            url = (base_url or self._find_base_url("openai")).rstrip("/")
            if not key:
                return {"ok": False, "provider": provider, "message": "No API key."}
            try:
                resp = await self._http.get(
                    f"{url}/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                models = [m["id"] for m in data.get("data", [])][:25]
                return {
                    "ok": True, "provider": provider, "models": models,
                    "message": f"OpenAI reachable with {len(models)} models.",
                }
            except Exception as e:
                return {"ok": False, "provider": provider, "message": str(e)}

        if provider == "anthropic":
            key = api_key or self._find_api_key("anthropic")
            url = (base_url or self._find_base_url("anthropic")).rstrip("/")
            if not key:
                return {"ok": False, "provider": provider, "message": "No API key."}
            try:
                resp = await self._http.get(
                    f"{url}/v1/models",
                    headers={
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                models = [m["id"] for m in data.get("data", [])][:25]
                return {
                    "ok": True, "provider": provider, "models": models,
                    "message": f"Anthropic reachable with {len(models)} models.",
                }
            except Exception as e:
                return {"ok": False, "provider": provider, "message": str(e)}

        if provider == "gemini":
            key = api_key or self._find_api_key("gemini")
            url = (base_url or self._find_base_url("gemini")).rstrip("/")
            if not key:
                return {"ok": False, "provider": provider, "message": "No API key."}
            try:
                resp = await self._http.get(
                    f"{url}/v1beta/models",
                    headers={"x-goog-api-key": key},
                )
                resp.raise_for_status()
                data = resp.json()
                models = []
                for item in data.get("models", []):
                    name = item.get("name", "")
                    if name.startswith("models/"):
                        models.append(name.split("/", 1)[1])
                return {
                    "ok": True, "provider": provider, "models": models[:25],
                    "message": f"Gemini reachable with {len(models)} models.",
                }
            except Exception as e:
                return {"ok": False, "provider": provider, "message": str(e)}

        return {"ok": False, "provider": provider, "message": f"Unknown provider: {provider}"}

    async def validate_model(
        self,
        model: str,
        *,
        provider: str = "ollama",
        purpose: str = "generation",
    ) -> tuple[bool, str]:
        model = (model or "").strip()
        provider = (provider or "ollama").strip().lower()
        if not model:
            return False, f"No model configured for {purpose}."

        if provider == "ollama":
            catalog = await self.list_ollama_model_details()
            if not catalog:
                return False, "Ollama is unreachable or returned no models."

            chosen = next((m for m in catalog if m["name"] == model), None)
            if not chosen:
                requested_base = model.split(":", 1)[0]
                chosen = next(
                    (m for m in catalog if str(m.get("name", "")).split(":", 1)[0] == requested_base),
                    None,
                )
            if not chosen:
                available = ", ".join(m["name"] for m in catalog[:8])
                suffix = f" Available: {available}" if available else ""
                return False, f'Model "{model}" is not installed in Ollama.{suffix}'
            if not chosen.get("chat_capable", True):
                return False, chosen.get("reason") or (
                    f'Model "{model}" is not suitable for {purpose}.'
                )
            return True, ""

        if provider == "openai":
            if not self._find_api_key("openai"):
                return False, "OPENAI_API_KEY not set."
            family_ok, family_reason = self._matches_provider_family("openai", model)
            if not family_ok:
                return False, family_reason
            return True, ""

        if provider == "anthropic":
            if not self._find_api_key("anthropic"):
                return False, "ANTHROPIC_API_KEY not set."
            family_ok, family_reason = self._matches_provider_family("anthropic", model)
            if not family_ok:
                return False, family_reason
            return True, ""

        if provider == "gemini":
            if not self._find_api_key("gemini"):
                return False, "GEMINI_API_KEY not set."
            family_ok, family_reason = self._matches_provider_family("gemini", model)
            if not family_ok:
                return False, family_reason
            return True, ""

        return False, f"Unknown provider: {provider}"

    async def preflight_check(
        self,
        *,
        generation_model: str,
        role_models: dict[str, str] | None = None,
        judge_model: str | None = None,
        judge_provider: str = "ollama",
    ) -> dict[str, Any]:
        role_models = role_models or {}
        checks: list[dict[str, str | bool]] = []

        ok, reason = await self.validate_model(
            generation_model,
            provider=self._config.default_provider,
            purpose="base generation",
        )
        checks.append(
            {
                "scope": "base_model",
                "model": generation_model,
                "provider": self._config.default_provider,
                "ok": ok,
                "reason": reason,
            }
        )

        for role, model in sorted(role_models.items()):
            if not str(model).strip():
                continue
            role_ok, role_reason = await self.validate_model(
                str(model),
                provider=self._config.default_provider,
                purpose=f"{role} role",
            )
            checks.append(
                {
                    "scope": role,
                    "model": str(model),
                    "provider": self._config.default_provider,
                    "ok": role_ok,
                    "reason": role_reason,
                }
            )

        if judge_model:
            judge_ok, judge_reason = await self.validate_model(
                judge_model,
                provider=judge_provider,
                purpose="judge",
            )
            checks.append(
                {
                    "scope": "judge",
                    "model": judge_model,
                    "provider": judge_provider,
                    "ok": judge_ok,
                    "reason": judge_reason,
                }
            )

        errors = [str(c["reason"]) for c in checks if not c["ok"] and c["reason"]]
        return {"ok": not errors, "checks": checks, "errors": errors}

    async def preflight_check_v2(
        self,
        enabled_models: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Validate each enabled model against its correct provider."""
        checks: list[dict[str, str | bool]] = []
        for entry in enabled_models:
            model_name = entry.get("model", "")
            provider = entry.get("provider") or resolve_provider(model_name)
            ok, reason = await self.validate_model(
                model_name, provider=provider, purpose="benchmark",
            )
            checks.append({
                "scope": f"{provider}/{model_name}",
                "model": model_name,
                "provider": provider,
                "ok": ok,
                "reason": reason,
            })
        errors = [str(c["reason"]) for c in checks if not c["ok"] and c["reason"]]
        return {"ok": not errors, "checks": checks, "errors": errors}

    # ------------------------------------------------------------------
    # Subscription CLI providers (routes through paid subscriptions)
    # ------------------------------------------------------------------

    async def _run_cli(
        self,
        cmd,
        stdin_data: str | None,
        timeout: int,
        model_name: str,
        provider_name: str,
        env: dict[str, str],
        use_shell: bool = False,
    ) -> LLMResponse:
        def _sync() -> LLMResponse:
            start = time.perf_counter()
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE if stdin_data else subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    shell=use_shell,
                    creationflags=0 if use_shell else _CREATE_NO_WINDOW,
                    cwd=tempfile.gettempdir(),
                )
                hard_limit = timeout + 30
                try:
                    stdout, stderr = proc.communicate(input=stdin_data, timeout=hard_limit)
                except subprocess.TimeoutExpired:
                    _kill_tree(proc)
                    proc.wait(timeout=5)
                    latency = (time.perf_counter() - start) * 1000
                    return LLMResponse(
                        model=model_name, ok=False, text="",
                        latency_ms=latency, error=f"Timed out after {timeout}s",
                        provider=provider_name,
                    )
                latency = (time.perf_counter() - start) * 1000
                if proc.returncode != 0:
                    detail = ((stderr or "").strip() or (stdout or "").strip() or f"exit {proc.returncode}")[:500]
                    return LLMResponse(
                        model=model_name, ok=False, text="",
                        latency_ms=latency, error=f"CLI error (rc={proc.returncode}): {detail}",
                        provider=provider_name,
                    )
                text = (stdout or "").strip()
                tokens = max(1, len(text) // 4)
                return LLMResponse(
                    text=text, ok=bool(text), latency_ms=latency,
                    model=model_name, provider=provider_name,
                    tokens_used=tokens, error="" if text else "Empty CLI response",
                )
            except FileNotFoundError:
                return LLMResponse(
                    model=model_name, ok=False, text="",
                    error=f"CLI not found: {cmd[0]}", provider=provider_name,
                )
            except Exception as e:
                latency = (time.perf_counter() - start) * 1000
                return LLMResponse(
                    model=model_name, ok=False, text="",
                    latency_ms=latency, error=str(e), provider=provider_name,
                )
        return await asyncio.to_thread(_sync)

    async def _call_claude_sub(
        self, prompt: str, model: str, temperature: float | None, system_prompt: str | None,
    ) -> LLMResponse:
        claude_path = _resolve_cli("claude")
        if not claude_path:
            return LLMResponse(ok=False, error="claude CLI not found in PATH", model="Claude (sub)", provider="claude_sub")
        cmd_args = ["-p", "-", "--model", "opus"]
        if system_prompt:
            cmd_args.extend(["--system-prompt", system_prompt])
        cmd, use_shell = _build_cmd(claude_path, cmd_args)
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        for attempt in range(3):
            result = await self._run_cli(cmd, prompt, _sub_timeout(prompt), "Claude (sub)", "claude_sub", env, use_shell=use_shell)
            if result.ok or not any(p in (result.error or "").lower() for p in _BILLING_PATTERNS):
                return result
            await asyncio.sleep(5)
        return result  # type: ignore[possibly-undefined]

    async def _call_chatgpt_sub(
        self, prompt: str, model: str, temperature: float | None, system_prompt: str | None,
    ) -> LLMResponse:
        codex_path = _resolve_cli("codex")
        if not codex_path:
            return LLMResponse(ok=False, error="codex CLI not found in PATH", model="ChatGPT (sub)", provider="chatgpt_sub")
        outfile = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        outfile.close()
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            cmd, use_shell = _build_cmd(codex_path, ["exec", "-", "--sandbox", "read-only", "--ephemeral", "--skip-git-repo-check", "-o", outfile.name])
            result = await self._run_cli(cmd, full_prompt, _sub_timeout(full_prompt), "ChatGPT (sub)", "chatgpt_sub", _headless_env(), use_shell=use_shell)
            if result.ok and not result.text:
                try:
                    with open(outfile.name, encoding="utf-8") as f:
                        file_text = f.read().strip()
                    if file_text:
                        return LLMResponse(
                            text=file_text, ok=True, latency_ms=result.latency_ms,
                            model="ChatGPT (sub)", provider="chatgpt_sub",
                            tokens_used=max(1, len(file_text) // 4),
                        )
                except Exception:
                    pass
            return result
        finally:
            try:
                os.unlink(outfile.name)
            except OSError:
                pass

    async def _call_gemini_sub(
        self, prompt: str, model: str, temperature: float | None, system_prompt: str | None,
    ) -> LLMResponse:
        gemini_path = _resolve_cli("gemini")
        if not gemini_path:
            return LLMResponse(ok=False, error="gemini CLI not found in PATH", model="Gemini (sub)", provider="gemini_sub")
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        cmd, use_shell = _build_cmd(gemini_path, ["-p", "-", "--model", "gemini-2.5-pro", "--approval-mode", "plan", "--output-format", "text"])
        return await self._run_cli(cmd, full_prompt, _sub_timeout(full_prompt), "Gemini (sub)", "gemini_sub", _headless_env(), use_shell=use_shell)

    async def check_subscription_health(self) -> dict[str, Any]:
        checks = {}
        for name in ("claude", "codex", "gemini"):
            path = _resolve_cli(name)
            checks[name] = {"available": path is not None, "path": path or "not found"}
        return checks

    async def close(self) -> None:
        await self._http.aclose()
