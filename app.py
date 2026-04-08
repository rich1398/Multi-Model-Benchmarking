"""Occursus-Claude — Multi-LLM Benchmarking Application."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import load_config, load_saved_settings, save_saved_settings
from models import AppConfig, JudgeConfig, ModelChoice, SavedSettings, TaskDef
from llm_client import LLMClient, resolve_provider
from role_assigner import auto_assign_roles
from judge import judge_response
from pipelines import get_pipeline, list_pipelines

app = FastAPI(title="Occursus-Claude", version="0.1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

_config: AppConfig | None = None
_client: LLMClient | None = None
_active_runs: dict[str, dict] = {}
_run_queues: dict[str, dict[str, asyncio.Queue]] = {}
_cancel_events: dict[str, asyncio.Event] = {}

ROLE_MODEL_KEYS = (
    "generator",
    "generator_alt",
    "critic",
    "synthesizer",
    "reviewer",
    "judge",
    "arbiter",
)
TASKS_DIR = Path("tasks")


def _get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient(_get_config())
    return _client


def _normalize_role_models(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key in ROLE_MODEL_KEYS:
        value = str(raw.get(key, "")).strip()
        if value:
            out[key] = value
    return out


def _resolve_tasks_file(config: AppConfig, suite: str | None) -> str:
    suite_name = (suite or "").strip()
    if not suite_name:
        return config.tasks_file
    candidate = TASKS_DIR / suite_name
    if candidate.is_file():
        return str(candidate)
    if not suite_name.endswith(".json"):
        candidate = TASKS_DIR / f"{suite_name}.json"
        if candidate.is_file():
            return str(candidate)
    return config.tasks_file


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    success_count = sum(1 for r in results if r.get("ok"))
    failure_count = total - success_count
    judge_failure_count = sum(1 for r in results if not r.get("judge_ok", True))
    total_score = sum(int(r.get("score") or 0) for r in results)
    success_scores = [int(r.get("score") or 0) for r in results if r.get("ok")]
    average_all = round(total_score / total, 2) if total else 0.0
    average_success_only = (
        round(sum(success_scores) / len(success_scores), 2)
        if success_scores else None
    )
    return {
        "total_results": total,
        "success_count": success_count,
        "failure_count": failure_count,
        "judge_failure_count": judge_failure_count,
        "success_rate": round(success_count / total, 4) if total else 0.0,
        "average_score_all_tasks": average_all,
        "average_score_success_only": average_success_only,
        "has_failures": failure_count > 0,
        "valid_for_thesis": (
            total > 0 and failure_count == 0 and judge_failure_count == 0 and success_count >= 2
        ),
    }


def _final_status(results: list[dict[str, Any]]) -> str:
    summary = _summarize_results(results)
    if summary["failure_count"] > 0:
        return "completed_with_failures"
    if summary["judge_failure_count"] > 0:
        return "completed_with_judge_failures"
    return "completed"


def _task_to_run_dict(t: TaskDef) -> dict[str, Any]:
    return {
        "id": t.id,
        "prompt": t.prompt,
        "rubric": t.rubric,
        "category": t.category,
        "difficulty": t.difficulty,
    }


def _benchmark_mode(
    model: str,
    role_models: dict[str, str],
    judge_backend: str,
    judge_model: str,
    default_provider: str,
) -> str:
    base_model = str(model or "").strip()
    if any(str(value).strip() and str(value).strip() != base_model for value in role_models.values()):
        return "multi_model"
    if str(judge_backend).strip() != str(default_provider).strip():
        return "multi_model"
    if str(judge_model).strip() and str(judge_model).strip() != base_model:
        return "multi_model"
    return "single_model_orchestration"


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path("static/index.html")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/models")
async def api_models():
    client = _get_client()
    ollama_models = await client.list_ollama_model_details()
    config = _get_config()

    all_models: list[dict[str, Any]] = []
    for m in ollama_models:
        all_models.append({**m, "provider": "ollama"})

    cloud_catalog = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o3-mini"],
        "anthropic": [
            "claude-sonnet-4-20250514",
            "claude-opus-4-1-20250805",
            "claude-haiku-3-5-20241022",
        ],
        "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    }
    for provider, names in cloud_catalog.items():
        has_key = any(m.provider == provider and m.api_key for m in config.models)
        for name in names:
            all_models.append({
                "name": name,
                "provider": provider,
                "chat_capable": True,
                "reason": "" if has_key else f"No API key for {provider}",
            })

    return {
        "models": all_models,
        "default_model": config.default_model,
        "default_provider": config.default_provider,
        "role_model_keys": list(ROLE_MODEL_KEYS),
        "provider_keys": {
            "openai": any(m.provider == "openai" and m.api_key for m in config.models),
            "anthropic": any(m.provider == "anthropic" and m.api_key for m in config.models),
            "gemini": any(m.provider == "gemini" and m.api_key for m in config.models),
        },
    }


@app.get("/api/settings")
async def api_get_settings():
    saved = load_saved_settings()
    return {
        "ollama_base_url": saved.ollama_base_url,
        "openai_base_url": saved.openai_base_url,
        "anthropic_base_url": saved.anthropic_base_url,
        "gemini_base_url": saved.gemini_base_url,
        "openai_api_key": _mask_key(saved.openai_api_key),
        "anthropic_api_key": _mask_key(saved.anthropic_api_key),
        "gemini_api_key": _mask_key(saved.gemini_api_key),
        "has_openai_key": bool(saved.openai_api_key),
        "has_anthropic_key": bool(saved.anthropic_api_key),
        "has_gemini_key": bool(saved.gemini_api_key),
        "selected_generation_models": [
            {"provider": m.provider, "model": m.model, "label": m.label}
            for m in saved.selected_generation_models
        ],
    }


@app.post("/api/settings")
async def api_save_settings(request: Request):
    global _config, _client
    body = await request.json()

    current = load_saved_settings()
    gen_models = tuple(
        ModelChoice(
            provider=m.get("provider", "ollama"),
            model=m.get("model", ""),
            label=m.get("label", ""),
        )
        for m in body.get("selected_generation_models", [])
    )

    new_settings = SavedSettings(
        ollama_base_url=body.get("ollama_base_url", current.ollama_base_url),
        openai_base_url=body.get("openai_base_url", current.openai_base_url),
        anthropic_base_url=body.get("anthropic_base_url", current.anthropic_base_url),
        gemini_base_url=body.get("gemini_base_url", current.gemini_base_url),
        openai_api_key=_unmask_key(body.get("openai_api_key", ""), current.openai_api_key),
        anthropic_api_key=_unmask_key(body.get("anthropic_api_key", ""), current.anthropic_api_key),
        gemini_api_key=_unmask_key(body.get("gemini_api_key", ""), current.gemini_api_key),
        selected_generation_models=gen_models,
    )
    save_saved_settings(new_settings)

    _config = None
    _client = None
    return {"ok": True, "message": "Settings saved."}


@app.post("/api/settings/test")
async def api_test_provider(request: Request):
    body = await request.json()
    provider = body.get("provider", "")
    api_key = body.get("api_key", "")
    base_url = body.get("base_url", "")
    client = _get_client()
    return await client.test_provider(provider, api_key=api_key, base_url=base_url)


def _mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return ""
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def _unmask_key(submitted: str, current: str) -> str:
    if not submitted:
        return current
    if "*" in submitted:
        return current
    return submitted


@app.get("/api/health")
async def api_health():
    config = _get_config()
    client = _get_client()
    preflight = await client.preflight_check(
        generation_model=config.default_model,
        role_models=config.role_models,
        judge_model=config.judge.model,
        judge_provider=config.judge.backend,
    )
    return preflight


@app.get("/api/pipelines")
async def api_pipelines():
    pipes = list_pipelines()
    return {
        "pipelines": [
            {
                "id": p.spec().id,
                "name": p.spec().name,
                "description": p.spec().description,
                "tier": p.spec().tier,
                "estimated_calls": p.spec().estimated_calls,
                "tags": list(p.spec().tags),
            }
            for p in pipes
        ]
    }


@app.get("/api/tasks")
async def api_tasks(suite: str | None = None):
    config = _get_config()
    tasks = _load_tasks(_resolve_tasks_file(config, suite))
    return {"tasks": [_task_to_dict(t) for t in tasks]}


@app.get("/api/task-suites")
async def api_task_suites():
    suites = []
    for path in sorted(TASKS_DIR.glob("*.json")):
        suites.append({"id": path.name, "name": path.stem.replace("_", " ")})
    return {"suites": suites}


@app.post("/api/run")
async def api_run(request: Request):
    body = await request.json()

    pipeline_ids = body.get("pipelines", ["single"])
    task_ids = body.get("task_ids", None)
    custom_prompt = body.get("custom_prompt", None)
    custom_tasks_json = body.get("custom_tasks", None)
    task_suite = body.get("task_suite", None)

    enabled_models = body.get("enabled_models", [])
    if not enabled_models and body.get("model"):
        model_name = body.get("model")
        model_provider = body.get("model_provider", resolve_provider(model_name))
        enabled_models = [{"provider": model_provider, "model": model_name}]

    assignment = auto_assign_roles(enabled_models)
    role_models = assignment["role_models"]
    judge_models = assignment["judge_models"]
    primary_model = assignment["primary_model"]
    primary_provider = assignment["primary_provider"]

    role_models_legacy = _normalize_role_models(body.get("role_models"))
    if role_models_legacy:
        role_models.update(role_models_legacy)

    config = _get_config()
    client = _get_client()
    run_config = replace(
        config,
        default_provider=primary_provider,
        tasks_file=_resolve_tasks_file(config, task_suite),
        role_models=role_models,
        judge=JudgeConfig(
            backend=judge_models[0]["provider"],
            model=judge_models[0]["model"],
            temperature=config.judge.temperature,
        ),
    )

    try:
        if custom_prompt:
            tasks = [TaskDef(
                id="custom",
                prompt=custom_prompt,
                rubric="Evaluate the quality, accuracy, and completeness of the response.",
                category="custom",
                difficulty="medium",
            )]
        elif custom_tasks_json:
            tasks = _parse_task_list(custom_tasks_json)
        else:
            tasks = _load_tasks(run_config.tasks_file)
    except (ValueError, json.JSONDecodeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    if task_ids:
        tasks = [t for t in tasks if t.id in task_ids]

    if not tasks:
        return JSONResponse({"error": "No tasks selected"}, status_code=400)

    valid_pipelines = []
    for pid in pipeline_ids:
        p = get_pipeline(pid)
        if p:
            valid_pipelines.append(pid)

    if not valid_pipelines:
        return JSONResponse({"error": "No valid pipelines selected"}, status_code=400)

    selected_model = primary_model or run_config.default_model
    preflight = await client.preflight_check_v2(enabled_models)
    if not preflight["ok"]:
        return JSONResponse(
            {
                "error": "Preflight validation failed",
                "details": preflight["errors"],
                "checks": preflight["checks"],
            },
            status_code=400,
        )

    run_id = str(uuid.uuid4())[:8]
    cancel_event = asyncio.Event()
    _cancel_events[run_id] = cancel_event
    _run_queues[run_id] = {}

    _active_runs[run_id] = {
        "status": "running",
        "started_at": time.time(),
        "total": len(tasks) * len(valid_pipelines),
        "completed": 0,
        "results": [],
        "summary": _summarize_results([]),
        "config": {
            "model": selected_model,
            "pipelines": valid_pipelines,
            "task_count": len(tasks),
            "task_suite": Path(run_config.tasks_file).name,
            "role_models": role_models,
            "judge_model": run_config.judge.model,
            "judge_backend": run_config.judge.backend,
            "benchmark_mode": _benchmark_mode(
                selected_model,
                role_models,
                run_config.judge.backend,
                run_config.judge.model,
                run_config.default_provider,
            ),
            "tasks": [_task_to_run_dict(t) for t in tasks],
            "preflight": preflight,
        },
    }

    asyncio.create_task(_run_benchmark(
        run_id, tasks, valid_pipelines, selected_model, run_config, client, cancel_event,
        judge_models=judge_models,
    ))

    return {"run_id": run_id, "total": len(tasks) * len(valid_pipelines)}


@app.get("/api/stream/{run_id}")
async def api_stream(run_id: str):
    if run_id not in _active_runs:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    client_id = str(uuid.uuid4())[:8]
    queue: asyncio.Queue = asyncio.Queue()

    if run_id not in _run_queues:
        _run_queues[run_id] = {}
    _run_queues[run_id][client_id] = queue

    async def event_stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                except asyncio.TimeoutError:
                    yield f"event: ping\ndata: {{}}\n\n"
                    continue

                if event.get("type") == "done":
                    yield f"event: done\ndata: {json.dumps(event)}\n\n"
                    break

                event_type = event.get("type", "progress")
                yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
        finally:
            _run_queues.get(run_id, {}).pop(client_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/api/run/{run_id}")
async def api_cancel(run_id: str):
    event = _cancel_events.get(run_id)
    if event:
        event.set()
        return {"status": "cancelling"}
    return JSONResponse({"error": "Run not found"}, status_code=404)


@app.post("/api/rerun-failed/{run_id}")
async def api_rerun_failed(run_id: str):
    run = _active_runs.get(run_id)
    if not run:
        try:
            from db import get_run_results
            run = await get_run_results(run_id)
        except Exception:
            run = None
    if not run:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    failed_rows = [row for row in run.get("results", []) if not row.get("ok")]
    if not failed_rows:
        return JSONResponse({"error": "This run has no failed cells to rerun"}, status_code=400)

    config = _get_config()
    client = _get_client()
    run_cfg = run.get("config", {}) or {}
    role_models = _normalize_role_models(run_cfg.get("role_models"))
    rerun_config = replace(
        config,
        tasks_file=_resolve_tasks_file(config, run_cfg.get("task_suite")),
        role_models=role_models,
        judge=JudgeConfig(
            backend=str(run_cfg.get("judge_backend") or config.judge.backend),
            model=str(run_cfg.get("judge_model") or config.judge.model),
            temperature=config.judge.temperature,
        ),
    )

    selected_model = str(run_cfg.get("model") or rerun_config.default_model)
    preflight = await client.preflight_check(
        generation_model=selected_model,
        role_models=role_models,
        judge_model=rerun_config.judge.model,
        judge_provider=rerun_config.judge.backend,
    )
    if not preflight["ok"]:
        return JSONResponse(
            {
                "error": "Preflight validation failed",
                "details": preflight["errors"],
                "checks": preflight["checks"],
            },
            status_code=400,
        )

    task_lookup = {
        t.id: t
        for t in _load_tasks(rerun_config.tasks_file)
    }
    for task_raw in run_cfg.get("tasks", []):
        task = TaskDef(
            id=str(task_raw.get("id", "")),
            prompt=str(task_raw.get("prompt", "")),
            rubric=str(task_raw.get("rubric", "")),
            category=str(task_raw.get("category", "general")),
            difficulty=str(task_raw.get("difficulty", "medium")),
        )
        task_lookup[task.id] = task

    cell_plan: list[tuple[TaskDef, str]] = []
    for row in failed_rows:
        task = task_lookup.get(str(row.get("task_id", "")))
        if task is None:
            task = TaskDef(
                id=str(row.get("task_id", "")),
                prompt=str(row.get("task_prompt", "")),
                rubric=str(row.get("task_rubric", "")),
                category=str(row.get("task_category", "general")),
                difficulty=str(row.get("task_difficulty", "medium")),
            )
        cell_plan.append((task, str(row.get("pipeline_id", ""))))

    new_run_id = str(uuid.uuid4())[:8]
    cancel_event = asyncio.Event()
    _cancel_events[new_run_id] = cancel_event
    _run_queues[new_run_id] = {}
    _active_runs[new_run_id] = {
        "status": "running",
        "started_at": time.time(),
        "total": len(cell_plan),
        "completed": 0,
        "results": [],
        "summary": _summarize_results([]),
        "config": {
            "model": selected_model,
            "pipelines": sorted({pipeline_id for _, pipeline_id in cell_plan}),
            "task_count": len({task.id for task, _ in cell_plan}),
            "task_suite": Path(rerun_config.tasks_file).name,
            "role_models": role_models,
            "judge_model": rerun_config.judge.model,
            "judge_backend": rerun_config.judge.backend,
            "benchmark_mode": _benchmark_mode(
                selected_model,
                role_models,
                rerun_config.judge.backend,
                rerun_config.judge.model,
                rerun_config.default_provider,
            ),
            "tasks": [_task_to_run_dict(task) for task, _ in cell_plan],
            "preflight": preflight,
            "rerun_of": run_id,
            "rerun_failed_only": True,
        },
    }

    asyncio.create_task(
        _run_benchmark(
            new_run_id,
            [],
            [],
            selected_model,
            rerun_config,
            client,
            cancel_event,
            cell_plan=cell_plan,
        )
    )
    return {"run_id": new_run_id, "total": len(cell_plan)}


@app.get("/api/results")
async def api_results_list():
    try:
        from db import list_runs
        runs = await list_runs()
        return {"runs": runs}
    except Exception:
        runs = []
        for rid, data in _active_runs.items():
            summary = data.get("summary", _summarize_results(data.get("results", [])))
            runs.append({
                "run_id": rid,
                "status": data["status"],
                "started_at": data["started_at"],
                "total": data["total"],
                "completed": data["completed"],
                "summary": summary,
                "success_rate": summary.get("success_rate"),
                "average_score_all_tasks": summary.get("average_score_all_tasks"),
                "average_score_success_only": summary.get("average_score_success_only"),
                "failure_count": summary.get("failure_count"),
                "judge_failure_count": summary.get("judge_failure_count"),
            })
        return {"runs": runs}


@app.get("/api/results/{run_id}")
async def api_results_detail(run_id: str):
    run = _active_runs.get(run_id)
    if not run:
        try:
            from db import get_run_results
            stored = await get_run_results(run_id)
            if stored is None:
                return JSONResponse({"error": "Run not found"}, status_code=404)
            return stored
        except Exception:
            return JSONResponse({"error": "Run not found"}, status_code=404)
    run["summary"] = _summarize_results(run.get("results", []))
    return run


@app.get("/api/export/{run_id}")
async def api_export(run_id: str, format: str = "json"):
    run = _active_runs.get(run_id)
    if not run:
        try:
            from db import get_run_results
            run = await get_run_results(run_id)
        except Exception:
            run = None
    if not run:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    results = run.get("results", [])

    if format == "csv":
        lines = [
            "task_id,pipeline,ok,score,judge_ok,judge_backend,judge_model,"
            "judge_reasoning,error,judge_error,wall_ms,llm_calls,tokens"
        ]
        for r in results:
            line = ",".join([
                _csv_escape(r.get("task_id", "")),
                _csv_escape(r.get("pipeline_id", "")),
                str(bool(r.get("ok", True))),
                str(r.get("score", 0)),
                str(bool(r.get("judge_ok", True))),
                _csv_escape(r.get("judge_backend", "")),
                _csv_escape(r.get("judge_model", "")),
                _csv_escape(r.get("judge_reasoning", "")),
                _csv_escape(r.get("error", "")),
                _csv_escape(r.get("judge_error", "")),
                str(int(r.get("wall_ms", 0))),
                str(r.get("llm_calls", 0)),
                str(r.get("total_tokens", r.get("tokens", 0))),
            ])
            lines.append(line)
        csv_text = "\n".join(lines)
        return StreamingResponse(
            iter([csv_text]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=occursus_{run_id}.csv"},
        )

    return JSONResponse({
        "run_id": run_id,
        "config": run.get("config", {}),
        "summary": run.get("summary", _summarize_results(results)),
        "results": results,
    })


async def _run_benchmark(
    run_id: str,
    tasks: list[TaskDef],
    pipeline_ids: list[str],
    model: str | None,
    config: AppConfig,
    client: LLMClient,
    cancel_event: asyncio.Event,
    cell_plan: list[tuple[TaskDef, str]] | None = None,
    judge_models: list[dict[str, str]] | None = None,
) -> None:
    run = _active_runs[run_id]
    planned_cells = cell_plan or [
        (task, pid) for task in tasks for pid in pipeline_ids
    ]
    run["total"] = len(planned_cells)

    try:
        for task, pid in planned_cells:
            if cancel_event.is_set():
                run["status"] = "cancelled"
                run["finished_at"] = time.time()
                run["summary"] = _summarize_results(run["results"])
                try:
                    from db import save_run
                    await save_run(run_id, run)
                except Exception:
                    pass
                await _broadcast(
                    run_id,
                    {
                        "type": "done",
                        "status": "cancelled",
                        "run_id": run_id,
                        "summary": run["summary"],
                    },
                )
                return

            pipeline = get_pipeline(pid)
            if not pipeline:
                continue

            await _broadcast(run_id, {
                "type": "progress",
                "task_id": task.id,
                "pipeline_id": pid,
                "message": f"Running {pid} on {task.id}...",
            })

            start = time.perf_counter()

            async def progress_cb(msg: str):
                await _broadcast(run_id, {
                    "type": "progress",
                    "task_id": task.id,
                    "pipeline_id": pid,
                    "message": msg,
                })

            try:
                result = await pipeline.execute(
                    task.prompt, client, config,
                    model=model, progress_callback=progress_cb,
                )
            except Exception as e:
                from models import PipelineResult
                result = PipelineResult(
                    pipeline_id=pid, ok=False, error=str(e),
                )

            wall_ms = (time.perf_counter() - start) * 1000

            if result.ok and result.final_text:
                if judge_models:
                    from judge import dual_judge_response
                    judge_result = await dual_judge_response(
                        task.prompt, task.rubric, result.final_text,
                        client, judge_models,
                        temperature=config.judge.temperature,
                    )
                else:
                    judge_result = await judge_response(
                        task.prompt, task.rubric, result.final_text,
                        client, config,
                    )
                score = judge_result.score if judge_result.ok else 0
                judge_reasoning = judge_result.reasoning
                judge_ok = judge_result.ok
                judge_error = judge_result.error
                judge_backend_value = judge_result.backend
                judge_model_value = judge_result.model
            else:
                score = 0
                judge_reasoning = f"Pipeline failed: {result.error}"
                judge_ok = True
                judge_error = ""
                judge_backend_value = config.judge.backend
                judge_model_value = config.judge.model

            entry = {
                "task_id": task.id,
                "task_prompt": task.prompt,
                "task_rubric": task.rubric,
                "task_category": task.category,
                "task_difficulty": task.difficulty,
                "pipeline_id": pid,
                "ok": result.ok,
                "final_text": result.final_text,
                "error": result.error,
                "score": score,
                "judge_reasoning": judge_reasoning,
                "judge_ok": judge_ok,
                "judge_error": judge_error,
                "judge_backend": judge_backend_value,
                "judge_model": judge_model_value,
                    "wall_ms": wall_ms,
                    "llm_calls": result.llm_calls,
                    "tokens": result.total_tokens,
                    "total_tokens": result.total_tokens,
                    "steps": [
                    {
                        "phase": s.phase,
                        "model": s.model,
                        "latency_ms": s.latency_ms,
                        "tokens": s.tokens,
                        "text_preview": s.text_preview,
                    }
                    for s in result.steps
                ],
            }

            run["results"].append(entry)
            run["completed"] += 1
            run["summary"] = _summarize_results(run["results"])

            await _broadcast(run_id, {
                "type": "result",
                "task_id": task.id,
                "pipeline_id": pid,
                "score": score,
                "judge_reasoning": judge_reasoning,
                "wall_ms": wall_ms,
                "llm_calls": result.llm_calls,
                "tokens": result.total_tokens,
                "total_tokens": result.total_tokens,
                "ok": result.ok,
                "error": result.error,
                "judge_ok": judge_ok,
                "judge_error": judge_error,
                "judge_backend": judge_backend_value,
                "judge_model": judge_model_value,
                "final_text": result.final_text,
                "steps": entry["steps"],
                "completed": run["completed"],
                "total": run["total"],
                "summary": run["summary"],
            })

        run["status"] = _final_status(run["results"])
        run["finished_at"] = time.time()
        run["summary"] = _summarize_results(run["results"])

        try:
            from db import save_run
            await save_run(run_id, run)
        except Exception:
            pass

        await _broadcast(run_id, {
            "type": "done",
            "status": run["status"],
            "run_id": run_id,
            "summary": run["summary"],
        })

    except Exception as e:
        run["status"] = "error"
        run["error"] = str(e)
        run["finished_at"] = time.time()
        run["summary"] = _summarize_results(run["results"])
        try:
            from db import save_run
            await save_run(run_id, run)
        except Exception:
            pass
        await _broadcast(run_id, {
            "type": "done",
            "status": "error",
            "error": str(e),
            "run_id": run_id,
            "summary": run["summary"],
        })
    finally:
        _cancel_events.pop(run_id, None)


async def _broadcast(run_id: str, event: dict) -> None:
    queues = _run_queues.get(run_id, {})
    for q in queues.values():
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


def _load_tasks(tasks_file: str) -> list[TaskDef]:
    path = Path(tasks_file)
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [
        TaskDef(
            id=t.get("id", ""),
            prompt=t.get("prompt", ""),
            rubric=t.get("rubric", ""),
            category=t.get("category", "general"),
            difficulty=t.get("difficulty", "medium"),
        )
        for t in raw
    ]


def _parse_task_list(raw: Any) -> list[TaskDef]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        raise ValueError("custom tasks must be a JSON array")
    return [
        TaskDef(
            id=t.get("id", f"custom_{i}"),
            prompt=t.get("prompt", ""),
            rubric=t.get("rubric", "Evaluate quality and accuracy."),
            category=t.get("category", "custom"),
            difficulty=t.get("difficulty", "medium"),
        )
        for i, t in enumerate(raw)
    ]


def _task_to_dict(t: TaskDef) -> dict:
    return {
        "id": t.id,
        "prompt": t.prompt,
        "rubric": t.rubric,
        "category": t.category,
        "difficulty": t.difficulty,
    }


def _csv_escape(value: str) -> str:
    value = str(value).replace('"', '""')
    if "," in value or '"' in value or "\n" in value:
        return f'"{value}"'
    return value


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
